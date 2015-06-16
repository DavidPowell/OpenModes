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


def part_ranges(parent_part, basis_container):
    "Construct the slice objects for the parent part and all of its children"
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


def build_lookup(index_data, basis_container):
    "Create the lookup table for a LookupArray"
    lookup = []
    shape = []
    for index in index_data:
        if isinstance(index, Part):
            # a hierarchy of parts
            ranges = part_ranges(index, basis_container)
            lookup.append(ranges)
            shape.append(ranges[index].stop)
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
    """

    def __new__(subtype, index_data, basis_container, dtype=float):
        """Construct an empty vector which can be indexed by parts

        Parameters
        ----------
        index_data : tuple
            Tuple elements can be integer, for a fixed length, a Part, for
            hierarchical indexing by Parts, or a tuple of strings.
        basis_container : BasisContainer
            The container with basis functions for each sub-part
        dtype : dtype
            The numpy data type of the vector
        """

        lookup, shape = build_lookup(index_data, basis_container)
        obj = np.ndarray.__new__(subtype, shape, dtype)
        obj.lookup = lookup
        obj.basis_container = basis_container

        # Finally, we must return the newly created object:
        return obj

    def __array_finalize__(self, obj):
        "Function is called when creating array from view as well"
        if obj is None:
            return

        # set default values for the custom attributes
        self.basis_container = getattr(obj, 'basis_container', None)
        self.lookup = getattr(obj, 'lookup', None)

    def __setstate__(self, state):
        """Allow additional attributes of this array type to be unpickled

        Note that some metadata may be lost when unpickling."""
        base_state, extended_state = state
        super(LookupArray, self).__setstate__(base_state)
        self.lookup, self.basis_container = extended_state

    def __reduce__(self):
        """Allow additional attributes of this array type to be pickled

        Note that some metadata may be lost when unpickling."""
        base_reduce = list(super(LookupArray, self).__reduce__(self))
        full_state = (base_reduce[2], self.lookup, self.basis_container)
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
            try:
                this_lookup = self.lookup[entry_num]
                new_idx.append(this_lookup[entry])

                if isinstance(entry, Part):
                    # Need to pass this metadata to the sub-array for its
                    # lookup table
                    sub_lookup.append(part_ranges(entry, self.basis_container))

                # If a string has been passed, then this dimension will have
                # been flattened out, so no metadata is needed

            except (KeyError, TypeError):
                new_idx.append(entry)

                if not isinstance(entry, numbers.Integral):
                    # Integers mean a dimension is dropped, in all other cases
                    # the dimension is kept but the dimension metadata is lost

                    # TODO: what if None/np.newaxis is passed?
                    if entry is None:
                        raise NotImplementedError

                    sub_lookup.append(None)

        # now add lookup data for all the non-indexed dimensions
        sub_lookup = sub_lookup+self.lookup[len(idx):]

        try:
            result = super(LookupArray, self).__getitem__(tuple(new_idx))
        except IndexError as exc:
            message = "Invalid index %s" % idx
            exc.args = (message,)+exc.args[1:]
            raise

        # May get a LookupArray or an array scalar back
        if isinstance(result, LookupArray):
            result.lookup = sub_lookup
            result.basis_container = self.basis_container

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
            try:
                this_lookup = self.lookup[entry_num]
                new_idx.append(this_lookup[entry])

            except (KeyError, TypeError):
                new_idx.append(entry)

        try:
            super(LookupArray, self).__setitem__(tuple(new_idx), value)
        except IndexError as exc:
            message = "Invalid index %s" % idx
            exc.args = (message,)+exc.args[1:]
            raise

    def transpose(self, **args):
        raise NotImplementedError


def view_lookuparray(original, index_data, basis_container):
    """Convert an array to a LookupArray, where possible avoiding copying"""
    lookup, shape = build_lookup(index_data, basis_container)
    result = original.reshape(shape).view(LookupArray)
    result.lookup = lookup
    result.basis_container = basis_container
    return result
