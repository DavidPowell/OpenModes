# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
#  OpenModes - An eigenmode solver for open electromagnetic resonantors
#  Copyright (C) 2013-2015 David Powell
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <http://www.gnu.org/licenses/>.
# -----------------------------------------------------------------------------
"Classes for arrays where one or more axes can be indexed by Part objects"

from __future__ import division

import numpy as np
from openmodes.parts import Part
import numbers
import collections
import six


def part_ranges_lowest(parent_part, basis_container):
    """Construct the slice objects for the parent part, iterating down only
    to the specified lowest level"""

    lowest = getattr(basis_container, 'lowest_parts', None)

    # get the size of each child part
    sizes = [len(basis_container[part]) for part in parent_part.iter_lowest(lowest)]

    offsets = np.cumsum([0]+sizes)

    ranges = {}

    # index of single parts to get the offsets
    lowest_part_num = 0

    # iterate with parents last, so they can get data from their child objects
    for part in parent_part.iter_lowest(lowest, parent_order='after'):
        if part in lowest:
            ranges[part] = slice(offsets[lowest_part_num], offsets[lowest_part_num+1])
            lowest_part_num += 1
        else:
            # take slice information from first and last child
            start = ranges[part.children[0]].start
            stop = ranges[part.children[-1]].stop
            ranges[part] = slice(start, stop)

    return ranges


def part_ranges(parent_part, basis_container):
    "Construct the slice objects for the parent part and all of its children"
    if hasattr(basis_container, 'lowest_parts'):
        return part_ranges_lowest(parent_part, basis_container)

    # get the size of each child part
    sizes = [len(basis_container[part]) for part in parent_part.iter_single()]
    offsets = np.cumsum([0]+sizes)

    ranges = {}

    # index of single parts to get the offsets
    single_part_num = 0

    # iterate with parents last, so they can get data from their child objects
    for part in parent_part.iter_all(parent_first=False):
        if hasattr(part, 'children'):
            # take slice information from first and last child
            start = ranges[part.children[0]].start
            stop = ranges[part.children[-1]].stop
            ranges[part] = slice(start, stop)
        else:
            ranges[part] = slice(offsets[single_part_num], offsets[single_part_num+1])
            single_part_num += 1

    return ranges


def build_lookup(index_data):
    "Create the lookup table for a LookupArray"
    lookup = []
    shape = []
    for index in index_data:
        if isinstance(index, collections.Iterable) and isinstance(index[0], Part):
            # a hierarchy of parts
            basis_container = index[1]
            part = index[0]
            ranges = part_ranges(part, basis_container)
            lookup.append((ranges, basis_container, part))
            shape.append(ranges[part].stop)
        elif isinstance(index, numbers.Integral):
            # an integer for a specific length
            lookup.append(None)
            shape.append(index)
        else:
            # Assumed to be a tuple of strings, although any sequence of
            # immutable objects should be okay.
            lookup.append({y: x for (x, y) in enumerate(index)})
            shape.append(len(index))

    return lookup, shape


class LookupArray(np.ndarray):
    """
    A subclass of a numpy array, where for certain dimensions, Part objects
    or strings can be used to index array elements.

    For explanation of subclassing numpy arrays, see
    http://docs.scipy.org/doc/numpy/user/basics.subclassing.html

    The following functionality of numpy arrays may cause problems, so they
    should be used with "extreme caution":
        - transpose
        - adding new axes by indexing with np.newaxis/None
        - flattening
        - anything other than C ordering
        - Functions which reduce dimensions
        - Indexing with ...
    """

    def __new__(subtype, index_data=None, lookup=None, shape=None, dtype=float):
        """Construct an empty vector which can be indexed by parts

        Parameters
        ----------
        index_data : tuple, optional
            Tuple elements can be integer, for a fixed length,
            a tuple (Part, BasisContainer), for hierarchical indexing by Parts,
            or a tuple of strings, for quantities
        lookup, shape: tuple, optional
            Instead of providing index_data, these elements can be provided
            directly if the lookup table is known in advance
        dtype : dtype, optional
            The numpy data type of the vector
        """

        if lookup is None or shape is None:
            lookup, shape = build_lookup(index_data)
        obj = np.ndarray.__new__(subtype, shape, dtype)
        obj.lookup = lookup

        # Finally, we must return the newly created object:
        return obj

    def __array_finalize__(self, obj):
        "Function is called when creating array from view as well"
        if obj is None:
            return

        # set default values for the custom attributes
        self.lookup = getattr(obj, 'lookup', None)

    def __setstate__(self, state):
        """Allow additional attributes of this array type to be unpickled

        Note that some metadata may be lost when unpickling."""
        base_state, extended_state = state
        super(LookupArray, self).__setstate__(base_state)
        self.lookup, = extended_state

    def __reduce__(self):
        """Allow additional attributes of this array type to be pickled

        Note that some metadata may be lost when unpickling."""
        base_reduce = list(super(LookupArray, self).__reduce__(self))
        full_state = (base_reduce[2], (self.lookup,))
        base_reduce[2] = full_state
        return tuple(base_reduce)

    # Under python 3.x, these members will not be called. However, they should
    # not cause any trouble.
    def __getslice__(self, start, stop):
        "Needed due to CPython bug"
        return self.__getitem__(slice(start, stop))

    def __setslice__(self, start, stop, val):
        "Needed due to CPython bug"
        self.__setitem__(slice(start, stop), val)

    def __getitem__(self, idx):
        """Gets an item or items from the array. Any of the indices may be the
        name of a range, in addition to all the usual fancy indexing options"""

        if not isinstance(idx, tuple):
            # force a single index to be a tuple
            idx = idx,

        new_idx = []
        sub_lookup = []

        # try to lookup every part of the index to convert to a range
        for entry_num, entry in enumerate(idx):
            if isinstance(entry, Part):
                # Need to pass this metadata to the sub-array for its
                # lookup table
                this_lookup, container, parent_part = self.lookup[entry_num]
                sub_lookup.append((part_ranges(entry, container), container, entry))
                new_idx.append(this_lookup[entry])
            elif isinstance(entry, six.string_types):
                # If a string has been passed, then this dimension will have
                # been flattened out, so no metadata is needed
                this_lookup = self.lookup[entry_num]
                new_idx.append(this_lookup[entry])
            else:
                new_idx.append(entry)

                if not isinstance(entry, numbers.Integral):
                    # Integers mean a dimension is dropped, in all other
                    # cases it is kept
                    if entry is None:
                        # TODO: what if None/np.newaxis is passed?
                        raise NotImplementedError
                    elif isinstance(entry, slice) and entry == slice(None):
                        # If slicing the whole dimension, metadata can be kept
                        sub_lookup.append(self.lookup[entry_num])
                    else:
                        # In all other cases metadata is lost
                        sub_lookup.append(None)

        # now add lookup data for all the non-indexed dimensions
        sub_lookup = sub_lookup+self.lookup[len(idx):]

        try:
            result = super(LookupArray, self).__getitem__(tuple(new_idx))
        except IndexError as exc:
            message = "Invalid index %s" % str(idx)
            exc.args = (message,)+tuple(str(n) for n in exc.args[1:])
            raise

        # May get a LookupArray or an array scalar back
        if isinstance(result, LookupArray):
            result.lookup = sub_lookup

        return result

    def __setitem__(self, idx, value):
        """Gets an item or items in the array. Any of the indices may be the
        name of a range, in addition to all the usual fancy indexing options"""
        if not isinstance(idx, tuple):
            # force a single index to be a tuple
            idx = idx,

        new_idx = []

        # try to lookup every part of the index to convert to a range
        for entry_num, entry in enumerate(idx):
            if isinstance(entry, Part):
                this_lookup, container, parent_part = self.lookup[entry_num]
                new_idx.append(this_lookup[entry])
            elif isinstance(entry, six.string_types):
                this_lookup = self.lookup[entry_num]
                new_idx.append(this_lookup[entry])
            else:
                new_idx.append(entry)

        try:
            super(LookupArray, self).__setitem__(tuple(new_idx), value)
        except IndexError as exc:
            message = "Invalid index %s" % idx
            exc.args = (message,)+tuple(str(n) for n in exc.args[1:])
            raise

    def transpose(self, **args):
        raise NotImplementedError

    @property
    def T(self):
        result = super(LookupArray, self).T
        assert(type(result) == LookupArray)
        result.lookup = list(reversed(self.lookup))
        return result

    def simple_view(self):
        """Return a view where quantity dimensions (with string keys) are
        collapsed into the subsequent dimension. View is of type ndarray."""
        new_shape = []

        for dim_n, lu_n in zip(reversed(self.shape), reversed(self.lookup)):
            if type(lu_n) == dict and type(list(lu_n.keys())[0]) == str:
                new_shape[-1] *= dim_n
            else:
                new_shape.append(dim_n)
        new_shape.reverse()
        return self.reshape(new_shape).view(np.ndarray)

    def dot(self, other):
        """Matrix/vector multiplication with another LookupArray"""
        if not isinstance(other, LookupArray):
            assert(self.shape[-1] == other.shape[0])
            new_lookup = self.lookup[:-1]+(None,)*(other.ndims-1)
            new_shape = self.shape[:-1]+other.shape[1:]
        elif self.lookup[-2:] == other.lookup[:2]:
            new_lookup = self.lookup[:-2]+other.lookup[2:]
            new_shape = self.shape[:-2]+other.shape[2:]
            other = other.simple_view()
        else:
            raise NotImplementedError

        new_array = LookupArray(lookup=new_lookup, shape=new_shape,
                                dtype=np.promote_types(self.dtype, other.dtype))
        new_array.simple_view()[:] = np.dot(self.simple_view(), other)
        return new_array


def view_lookuparray(original, index_data):
    """Convert an array to a LookupArray, where possible avoiding copying"""
    lookup, shape = build_lookup(index_data)
    result = original.reshape(shape).view(LookupArray)
    result.lookup = lookup
    return result


def loop_star_indices(x):
    """Return the indices into the array corresponding to the loop and star
    parts. The array must have been constructed using loop/star basis
    functions.

    Parameters
    ----------
    x: LookupArray
        Must have either 2 dimensions, both of which must be indexable
        by Parts

    Returns
    -------
    indices_loop, indices_star: list of ndarray
        For each dimension n, indices_loop[n] is an array indexing the loop
        part, and indices_star[n] is an array indexing the star part.
    """

    indices_loop = []
    indices_star = []

    for lookup_num, lookup in enumerate(x.lookup):
        if isinstance(list(lookup[0].keys())[0], Part):
            # This index is a lookup for Parts, so find all the SingleParts
            # along this index and add the relevant ranges to the indexing
            # array

            loop_list = []
            star_list = []
            # First find the parent part, the one covering the largest range
            part_list = list(lookup[0].keys())
            parent_part = part_list[np.argmax(lookup[n].stop-lookup[n].start
                                              for n in part_list)]

            # now iterate over all SingleParts of this parent part
            for part in parent_part.iter_single():
                part_start = lookup[0][part].start

                bf = lookup[1][part]
                loop_range = bf.loop_range
                loop_list.append(np.arange(loop_range.start+part_start,
                                           loop_range.stop+part_start))

                star_range = bf.star_range
                star_list.append(np.arange(star_range.start+part_start,
                                           star_range.stop+part_start))

            # If this is not the last axis, then add the necessary number of
            # extra dimensions to each array so that they will be broadcast
            # correctly when the caller goes to use them
            new_shape = (-1,)+(1,)*(x.ndim-lookup_num-1)
            loop_array = np.hstack(loop_list).reshape(new_shape)
            star_array = np.hstack(star_list).reshape(new_shape)

            indices_loop.append(loop_array)
            indices_star.append(star_array)
        else:
            # This index is not for parts, so just take the whole axis
            indices_loop.append(slice(None))
            indices_star.append(slice(None))

    return indices_loop, indices_star
