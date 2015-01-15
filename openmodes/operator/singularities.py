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


class SingularSparse(object):
    """A sparse matrix class for holding A and phi arrays with the same
    sparsity pattern to store singular triangle impedances"""
    def __init__(self):
        self.rows = {}

    def __setitem__(self, index, item):
        """Add an item, which will be stored in a dictionary of dictionaries.
        Item is assumed to be (A, phi)"""

        row, col = index
        try:
            self.rows[row][col] = item
        except KeyError:
            self.rows[row] = {col: item}

    def iteritems(self):
        "Iterate through all items"
        for row, row_dict in self.rows.iteritems():
            for col, item in row_dict.iteritems():
                yield ((row, col), item)

    def to_csr(self):
        """Convert the matrix to compressed sparse row format, with
        common index array and two data arrays for A and phi"""
        A_data = []
        phi_data = []
        indices = []
        indptr = [0]

        data_index = 0

        num_rows = max(self.rows.keys())+1

        for row in xrange(num_rows):
            if row in self.rows:
                # the row exists, so process it
                for col, item in self.rows[row].iteritems():
                    A_data.append(item[0])
                    phi_data.append(item[1])
                    indices.append(col)

                    data_index = data_index + 1
            # regardless of whether the row exists, update the index pointer
            indptr.append(data_index)

        return (np.array(phi_data, dtype=np.float64, order="F"),
                np.array(A_data, dtype=np.float64, order="F"),
                np.array(indices, dtype=np.int32, order="F"),
                np.array(indptr, dtype=np.int32, order="F"))

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
        polygons = basis.mesh.polygons
        nodes = basis.mesh.nodes
        num_faces = len(polygons)

        singular_terms = SingularSparse()
        # find the neighbouring triangles (including self terms) to integrate
        # singular part
        for p in xrange(0, num_faces):  # observer

            nodes_p = nodes[polygons[p]]

            sharing_triangles = set()
            for node in polygons[p]:
                sharing_triangles = sharing_triangles.union(sharing_nodes[node])

            # find any neighbouring elements which are touching
            for q in sharing_triangles:
                if q == p:
                    # calculate the self term using the exact formula
                    res = openmodes.core.arcioni_singular(nodes_p,)
                    assert(np.all(np.isfinite(res[0])) and np.all(np.isfinite(res[1])))
                    singular_terms[p, p] = res
                else:
                    # at least one node is shared
                    # calculate neighbour integrals semi-numerically
                    res = openmodes.core.face_integrals_hanninen(
                                        nodes[polygons[q]],
                                        integration_rule.xi_eta,
                                        integration_rule.weights, nodes_p)
                    assert(np.all(np.isfinite(res[0])) and np.all(np.isfinite(res[1])))
                    singular_terms[p, q] = res

        cached_singular_terms[unique_id] = singular_terms.to_csr()
        return cached_singular_terms[unique_id]
