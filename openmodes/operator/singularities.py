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

import logging
import numpy as np
import openmodes.core
from openmodes.taylor_duffy import taylor_duffy, OPERATOR_EFIE, OPERATOR_MFIE
from openmodes.basis import LinearTriangleBasis


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


def singular_impedance_rwg(basis, operator, tangential_form, num_terms,
                           rel_tol, normals=None):
    """Precalculate the singular impedance terms for an object

    Parameters
    ----------
    basis: LinearTriangleBasis object
        The basis functions representing the object for which to calculate the
        singularities
    operator: string
        The operator form, either "EFIE" or "MFIE"
    tangential_form: boolean
        If True, the T form operator is taken, otherwise the N form
    num_terms: integer
        The number of singular terms to extract
    rel_tol: float
        The desired relative tolerance of the singular integrals
    normals: ndarray(num_triangles, 3) of float, optional
        The surface normals, required for n x operator forms

    Returns
    -------
    singular_terms : SingularSparse object
        The sparse array of singular impedance terms
    """

    if not isinstance(basis, LinearTriangleBasis):
        raise ValueError("Basis functions are not RWG based")

    # Check if this part's singularities have previously been calculated
    # Note that higher accuracy calculations will not be used if a lower
    # accuracy is requested. This avoids non-deterministic behaviour.
    unique_id = ("RWG", operator, tangential_form, basis.id, rel_tol)
    if unique_id in cached_singular_terms:
        return cached_singular_terms[unique_id]

    sharing_nodes = basis.mesh.triangles_sharing_nodes()

    # slightly inefficient reordering and resizing of nodes array
    polygons = np.ascontiguousarray(basis.mesh.polygons)
    nodes = np.ascontiguousarray(basis.mesh.nodes, dtype=np.float64)
    num_faces = len(polygons)

    nodes_c = np.ascontiguousarray(nodes, dtype=np.float64)
    polygons_c = np.ascontiguousarray(polygons)

    # Choose what to store based on the operator for which the singularities
    # are to be calculated, including T vs N form
    if operator == "EFIE" and tangential_form:
        singular_terms = MultiSparse([(np.float64, (num_terms,)),        # phi
                                      (np.float64, (num_terms, 3, 3))])  # A
        which_operator = OPERATOR_EFIE
    elif operator == "EFIE":
        raise NotImplementedError
    elif operator == "MFIE" and not tangential_form:
        singular_terms = MultiSparse([(np.float64, (num_terms, 3, 3))])  # A
        which_operator = OPERATOR_MFIE
    elif operator == "MFIE":
        raise NotImplementedError
    else:
        raise ValueError("Don't know how to handle singularities for operator "
                         "%s with tangential_form=%s" %
                         (operator, tangential_form))

    logging.info("Integrating singular terms for basis function %s, with "
                 "operator type %s, tangential_form %s, %d terms, "
                 "relative tolerance %e" % (basis, operator, tangential_form,
                                            num_terms, rel_tol))

    # find the neighbouring triangles (including self terms) to integrate
    # singular part
    for p in xrange(0, num_faces):  # observer
        sharing_triangles = set()
        for node in polygons[p]:
            sharing_triangles = sharing_triangles.union(sharing_nodes[node])
        # find any neighbouring elements which are touching
        for q in sharing_triangles:  # source
            if operator == "MFIE":
                # The self triangle terms are not evaluated for MFIE
                if q == p:
                    continue
                normal = normals[p]
            else:
                # normals are not supplied for EFIE
                normal = None
            # at least one node is shared
            res = taylor_duffy(nodes, polygons[p], polygons[q], which_operator,
                               tangential_form, num_terms, rel_tol=rel_tol,
                               normal=normal)
            singular_terms[p, q] = res

    # Arrays are currently put into fortran order, under the assumption
    # that they will mostly be used by fortran routines.
    cached_singular_terms[unique_id] = singular_terms.to_csr(order='F')
    return cached_singular_terms[unique_id]
