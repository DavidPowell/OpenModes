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

from itertools import islice
import numpy as np

from openmodes.basis import get_basis_functions
from openmodes.namedarray import NamedArray

class VectorParts(object):
    def __init__(self, parts, vectors, basis, operator):
        self.parts = parts
        self.vectors = vectors
        self.basis = basis
        self.operator = operator
        #self.dtype = self.vectors[0].dtype

        # The initial data could be available at the root or leaves of the
        # tree, in which cases we want to split or combine the data
        # respectively when asked for it at an intermediate level
        if parts in vectors:
            self.combine_vectors = False
        else:
            for part in parts.iter_single():
                if part not in vectors:
                    raise ValueError("Missing data for some parts")
            self.combine_vectors = True

    def __getitem__(self, part):
        """Allow the component of this vector to be retrieved which corresponds
        to some particular part

        Parameters
        ----------
        index : Part
            The part for which to retrieve the relevant parts of the vector
        """

        # allow vector[:] to return the full vector
        if part == slice(None):
            part = self.parts

        try:
            return self.vectors[part]
        except KeyError:
            if part not in self.parts:
                raise KeyError("Invalid part specified")

        if self.combine_vectors:
            # combine lower-level vectors
            combined = []
            old_sections = []
            iterators = []
            for sub_part in part.iter_single():
                combined.append(self.vectors[sub_part])
                old_sections.append(self.operator.sections(self.basis[sub_part]))
                iterators.append(i for i in combined[-1])

            if len(combined) == 1:
                # don't bother combining if we only have a single matrix
                new_vector = combined[0]
            else:
                # work out the size of the new vector, and each of its sections
                new_sections = zip(old_sections)
                new_vector = [] #np.empty(sum(new_sections), self.dtype)
                for new_section in new_sections:
                    for orig_num, orig_iter in zip(new_sections, iterators):
                        new_vector.extend(islice(orig_iter, orig_num))
                new_vector = np.array(new_vector)

            self.vectors[sub_part] = new_vector
        else:
            raise NotImplementedError
            # split a high-level vector
            parent_part = part.parent
            while parent_part not in self.vectors:
                #if parent_part is None:
                #    raise ValueError("Cannot find vector to split")
                parent_part = parent_part.parent
                
                

        return new_vector

    def dot(self, x):
        "Dot product with another array or vector"
        return self[:].dot(x[:])


def empty_vector_parts(parent, basis_functions, dtype, cols=None):
    """Construct an empty vector which can be indexed by the parts
    
    Parameters
    ----------
    parent : Part
        The part which contains everything in the vector
    basis_functions : dictionary
        For each SinglePart, the corresponding basis functions should be in
        this dictionary
    dtype : dtype
        The numpy data type of the vector
    cols : integer, optional
        If specified, then this array will have multiple columns, which will
        not be associated with any names
    """

    # First go through all the SingleParts, and work out the size of the
    # complete vector, and all the sections within it
    sections = []
    for part in parent.iter_single():
        sections.append(basis_functions[part].sections)

    num_sections = len(sections[0])
    # insert zeros at the start to be the offset of the first part
    sections.insert(0, [0 for n in xrange(num_sections)])

    # first index is section, second is the part
    sections = np.array(sections).T

    # the offset of each part's sections in the final vector
    offsets = np.cumsum(sections).reshape(sections.shape)

    index_arrays = {}
    single_part_num = 0
    # knowing the sizes of all sections, work out the location of each part
    # within the data
    for part in parent.iter_all(parent_first=False):
        if hasattr(part, 'parts'):
            # build up the index array from the children
            index_arrays[part] = np.hstack(index_arrays[child] 
                                           for child in part.parts)
        else:
            part_index = []
            # this is a SinglePart, so generate its index array from the sections
            for sec_num in xrange(num_sections):
                part_index.append(np.arange(offsets[sec_num, single_part_num],
                                            offsets[sec_num, single_part_num+1]))
            index_arrays[part] = np.hstack(part_index)
            single_part_num += 1

    if cols is None:
        return NamedArray(offsets[-1, -1], index_arrays, dtype=dtype)
    else:
        return NamedArray((offsets[-1, -1], cols), (index_arrays, {}),
                          dtype=dtype)

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
    part1 = sim.place_part(mesh)
    part2 = sim.place_part(mesh)

    e_inc = np.array([1, 0, 0], dtype=np.complex128)
    k_hat = np.array([0, 0, 1], dtype=np.complex128)

    s = 2j*np.pi*1e9

    def vdot(self, x):
        "Conjugated dot product with another array or vector"
        return np.dot(self[:], x[:])
    V = sim.source_plane_wave(e_inc, s/c*k_hat)
    V[part1] = 66
    V[part2] = 7
    print V
    