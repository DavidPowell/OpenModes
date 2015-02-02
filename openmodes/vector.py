# -*- coding: utf-8 -*-
#-----------------------------------------------------------------------------
#  OpenModes - An eigenmode solver for open electromagnetic resonantors
#  Copyright (C) 2013 David Powell
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
#-----------------------------------------------------------------------------
"Classes for holding current and fields as vectors"

from __future__ import division

import numpy as np


def build_index_arrays(parent_part, basis_container):
    """Calculate the index arrays for each part with respect to the vector

    Parameters
    ----------
    parent_part : Part
        The part which includes the whole range of this vector
    basis_container : BasisContainer
        The container with all the basis functions
    """

    # First go through all the SingleParts, and work out the size of the
    # complete vector, and all the sections within it
    sections = []
    for part in parent_part.iter_single():
        basis = basis_container[part]
        sections.append(basis.sections)

    num_sections = len(sections[0])
    # insert zeros at the start to be the offset of the first part
    sections.insert(0, [0 for n in xrange(num_sections)])

    # first index is section, second is the part
    sections = np.array(sections).T

    # the offset of each part's sections in the final vector
    offsets = np.cumsum(sections).reshape(sections.shape)

    # Now iterate through all parts, getting the sections for higher parts
    # from their child parts
    index_arrays = {}
    single_part_num = 0
    for part in parent_part.iter_all(parent_first=False):
        if hasattr(part, 'children'):
            # build up the index array from the children
            index_arrays[part] = np.hstack(index_arrays[child]
                                           for child in part.children)
            # sort to prevent unwanted reordering
            index_arrays[part].sort()
        else:
            part_index = []
            # Is a SinglePart, so generate its index array from the sections
            for sec_num in xrange(num_sections):
                part_index.append(np.arange(offsets[sec_num, single_part_num],
                                            offsets[sec_num, single_part_num+1]))
            index_arrays[part] = np.hstack(part_index)
            single_part_num += 1

    total_length = offsets[-1, -1]
    return index_arrays, total_length


class VectorParts(np.ndarray):
    """
    A subclass of a numpy array, where elements along the last dimension are
    specified by a Part

    For explanation of subclassing numpy arrays, see
    http://docs.scipy.org/doc/numpy/user/basics.subclassing.html
    """

    def __new__(subtype, parent_part, basis_container, dtype=float, cols=None):
        """Construct an empty vector which can be indexed by the parts

        Parameters
        ----------
        parent_part : Part
            The part which contains everything in the vector
        basis_class : BasisContainer
            The container with basis functions for each sub-part
        dtype : dtype
            The numpy data type of the vector
        cols : integer, optional
            If specified, then this array will have multiple columns, which
            will not be associated with any names
        """

        # knowing the sizes of all sections, work out the location of each part
        # within the data
        index_arrays, total_length = build_index_arrays(parent_part,
                                                        basis_container)

        if cols is None:
            shape = (total_length,)
        else:
            shape = (total_length, cols)

        obj = np.ndarray.__new__(subtype, shape, dtype)

        obj.parent_part = parent_part
        obj.index_arrays = index_arrays
        obj.basis_container = basis_container

        # Finally, we must return the newly created object:
        return obj

    def __array_finalize__(self, obj):
        "Function is called when creating array from view as well"
        if obj is None:
            return

        # set default values for the custom attributes
        self.basis_container = obj.basis_container
        self.parent_part = getattr(obj, 'parent_part', None)
        self.index_arrays = getattr(obj, 'index_arrays', {})

    def __setstate__(self, state):
        """Allow additional attributes of this array type to be unpickled

        Note that some metadata may be lost when unpickling."""
        nd_state, own_state = state
        super(VectorParts, self).__setstate__(nd_state)
        self.parent_part, self.index_arrays, self.basis_container = own_state

    def __reduce__(self):
        """Allow additional attributes of this array type to be pickled

        Note that some metadata may be lost when unpickling."""
        object_state = list(super(VectorParts, self).__reduce__(self))
        vectorparts_state = (self.parent_part, self.index_arrays,
                             self.basis_container)
        object_state[2] = [object_state[2], vectorparts_state]
        return tuple(object_state)

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

        part = None
        try:
            new_idx.append(self.index_arrays[idx[0]])
            part = idx[0]
        except (KeyError, TypeError):
            # index item not found in dictionary
            new_idx.append(idx[0])
        if len(idx) > 1:
            new_idx.extend(idx[1:])
        try:
            result = super(VectorParts, self).__getitem__(tuple(new_idx))
        except IndexError:
            raise IndexError("Invalid index")

        if isinstance(result, VectorParts):
            # If the result is a VectorPart, then set the metadata to the
            # appropriate values
            if idx[0] == slice(None):
                # indexing this full dimension, just pass on the parent
                # part and indexing dictionary directly
                result.parent_part = self.parent_part
                result.index_arrays = self.index_arrays
            elif part is not None:
                # A certain part was selected. Update the index arrays accordingly
                result.parent_part = part
                index_arrays, total_length = build_index_arrays(part, self.basis_container)
                result.index_arrays = index_arrays
        return result

    def __setitem__(self, idx, value):
        """Gets an item or items in the array. Any of the indices may be the
        name of a range, in addition to all the usual fancy indexing options"""
        if not isinstance(idx, tuple):
            # force a single index to be a tuple
            idx = idx,
        new_idx = []

        try:
            new_idx.append(self.index_arrays[idx[0]])
        except (KeyError, TypeError):
            # index item not found in dictionary
            new_idx.append(idx[0])
        if len(idx) > 1:
            new_idx.extend(idx[1:])
        super(VectorParts, self).__setitem__(tuple(new_idx), value)

    def weight(self, part_modes, normalise_modes=None):
        """Weight the vector using a set of modes for each part

        Parameters
        ----------
        part_modes : sequence of tuples of (part, modes_vec)
            For each part `modes_vec` will be used as modes in which to base
            the reduced model. Multiple modes should be represented by mutiple
            columns of `modes_vec`.
        normalise_modes : string, optional
            If `None`, no normalisation will be performed
            If 'complex', then the sum of each vector squared will be
            normalised to 1, without taking the magnitude of each element

        Returns
        -------
        V : ndarray
            the reduced vector

        Note that indirectly including a part more than once in part_modes
        (e.g. by including it and its parent part) will yield invalid results.
        """

        V_red = []

        for part, modes in part_modes:
            if normalise_modes == 'complex':
                modes = modes/np.sqrt(np.sum(modes**2, axis=0))
            V_red.append(modes.T.dot(self[part]))

        return np.hstack(V_red)


if __name__ == "__main__":
    import openmodes
    from openmodes.constants import c
    import os.path as osp

    mesh_tol = 0.5e-3
    name = 'SRR'
    sim = openmodes.Simulation(name='vector_test',
                               basis_class=openmodes.basis.LoopStarBasis,
                               log_display_level=20)
    mesh = sim.load_mesh(osp.join(openmodes.geometry_dir, name+'.geo'),
                         mesh_tol=mesh_tol)
    part = sim.place_part()
    part1 = sim.place_part(mesh, part)
    part2 = sim.place_part(mesh, part)
    part3 = sim.place_part(mesh)

    e_inc = np.array([1, 0, 0], dtype=np.complex128)
    k_hat = np.array([0, 0, 1], dtype=np.complex128)

    s = 2j*np.pi*1e9

    V = sim.source_plane_wave(e_inc, s/c*k_hat)
    #V[part] = -5
    V[part1] = 66
    V[part2] = 7

    #a = V[part1]
    print V[part][part2]
