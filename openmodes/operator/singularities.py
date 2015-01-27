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

"""Routines for dealing with singular integrals, where for convenience the
quantities for both EFIE and MFIE may be calculated simultaneously"""

import numpy as np
import openmodes.core
from openmodes.taylor_duffy import taylor_duffy


class MultiSparse(object):
    """A sparse matrix class for holding multiple arrays with the same
    sparsity pattern."""
    def __init__(self, subarrays):
        """
        Parameters
        ----------
        subarrays: list of tuple (dtype, shape)
            The data type and shape of each of the arrays stored. Shape
            refers to the individual stored elements, and should be set to
            None if each element is a scalar.
        """
        self.rows = {}
        self.subarrays = subarrays

    def __setitem__(self, index, item):
        """Add an item, which will be stored in a dictionary of dictionaries.
        Item is a tuple, with elements corresponding to previously passed
        subarrays list. Each item may be a arbitrary type, inlcluding a multi
        dimensional array
        """

        row, col = index
        try:
            self.rows[row][col] = item
        except KeyError:
            self.rows[row] = {col: item}

    def __getitem__(self, index):
        return self.rows[index[0]][index[1]]

    def __len__(self):
        return sum(len(row_dict) for row, row_dict in self.rows.iteritems())

    def iteritems(self):
        "Iterate through all items"
        for row, row_dict in self.rows.iteritems():
            for col, item in row_dict.iteritems():
                yield ((row, col), item)

    def to_csr(self, order='C'):
        """Convert the matrix to compressed sparse row format, with
        common index arrays

        Parameters
        ----------
            order: string, optional
                The order in which to create the dense arrays, should be 'C'
                or 'F'

        Returns
        -------
        array1,...arrayN : ndarray
            Each of the data arrays
        indices : ndarray
            The indices within each row
        indptr : ndarray
            The pointer to each row's indices
        """

        num_objs = len(self)
        indices = np.empty(num_objs, dtype=np.int32, order=order)
        indptr = [0]

        data_arrays = []

        for (dtype, shape) in self.subarrays:
            if shape is None:
                data_arrays.append(np.empty(shape=num_objs, dtype=dtype,
                                   order=order))
            else:
                data_arrays.append(np.empty(shape=(num_objs,)+shape,
                                   dtype=dtype, order=order))

        data_index = 0
        num_rows = max(self.rows.keys())+1

        for row in xrange(num_rows):
            if row in self.rows:
                # the row exists, so process it
                for col, item in self.rows[row].iteritems():
                    for sub_count, sub_item in enumerate(item):
                        data_arrays[sub_count][data_index] = sub_item

                    indices[data_index] = col
                    data_index += 1

            # regardless of whether the row exists, update the index pointer
            indptr.append(data_index)

        # now put all subarrays and indices into a single dictionary
        return (data_arrays+[indices,
                             np.array(indptr, dtype=np.int32, order=order)])

cached_singular_terms = {}


def singular_impedance_rwg_efie_homogeneous(basis, integration_rule):
    """Precalculate the singular impedance terms for an object

    Parameters
    ----------
    quadrature_rule : tuple of 2 ndarrays
        The barycentric coordinates and weights of the quadrature to
        use for the non-analytical neighbour terms.

    Returns
    -------
    singular_terms : SingularSparse object
        The sparse array of singular impedance terms

    """
    unique_id = ("EFIE", "RWG", basis.id, integration_rule.id)
    if unique_id in cached_singular_terms:
        #print "singular terms retrieved from cache"
        return cached_singular_terms[unique_id]
    else:
        sharing_nodes = basis.mesh.triangles_sharing_nodes()

        # Precalculate the singular integration rules for faces, which depend
        # on the observation point

        # slightly inefficient reordering and resizing of nodes array
        polygons = np.ascontiguousarray(basis.mesh.polygons)
        nodes = np.ascontiguousarray(basis.mesh.nodes, dtype=np.float64)
        num_faces = len(polygons)

        nodes_c = np.ascontiguousarray(nodes, dtype=np.float64)
        polygons_c = np.ascontiguousarray(polygons)

        singular_terms = MultiSparse(((np.float64, None),     # phi
                                      (np.float64, (3, 3))))  # A
        # find the neighbouring triangles (including self terms) to integrate
        # singular part
        for p in xrange(0, num_faces):  # observer
            sharing_triangles = set()
            for node in polygons[p]:
                sharing_triangles = sharing_triangles.union(sharing_nodes[node])

            # find any neighbouring elements which are touching
            for q in sharing_triangles:  # source
                # at least one node is shared
                    res = taylor_duffy(nodes, polygons[p], polygons[q],
                                       rel_tol=1e-8)
                    singular_terms[p, q] = (res[1]*4*np.pi, res[0]*4*np.pi)

        cached_singular_terms[unique_id] = singular_terms.to_csr(order='F')
        return cached_singular_terms[unique_id]
