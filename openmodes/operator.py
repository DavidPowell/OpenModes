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


import numpy as np

import openmodes.core

from openmodes.constants import epsilon_0, mu_0, pi, c
from openmodes.basis import LinearTriangleBasis, LoopStarBasis
from openmodes.impedance import (EfieImpedanceMatrix,
                                 EfieImpedanceMatrixLoopStar,
                                 ImpedanceParts)
from openmodes.vector import VectorParts


class FreeSpaceGreensFunction(object):
    "Green's function in a homogeneous isotropic medium such as free space"
    def __init__(self, epsilon_r=1, mu_r=1):
        """
        Parameters
        ---------
        epsilon_r, mu_r : real
            Relative permittivity and permeability of the background medium

        A lossy background medium is currently unsupported
        """
        self.epsilon = epsilon_r*epsilon_0
        self.mu = mu_r*mu_0
        self.c = c/np.sqrt(mu_r*epsilon_r)
        self.eta = np.sqrt(self.mu/self.epsilon)


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


def impedance_rwg_efie_free_space(s, integration_rule, basis_o, nodes_o,
                                  basis_s, nodes_s, self_impedance):
    """EFIE derived Impedance matrix for RWG or loop-star basis functions"""

    transform_L_o, transform_S_o = basis_o.transformation_matrices
    num_faces_o = len(basis_o.mesh.polygons)

    if (self_impedance):
        # calculate self impedance

        singular_terms = singular_impedance_rwg_efie_homogeneous(basis_o,
                                                             integration_rule)
        if (np.any(np.isnan(singular_terms[0])) or
                np.any(np.isnan(singular_terms[1]))):
            raise ValueError("NaN returned in singular impedance terms")

        num_faces_s = num_faces_o
        A_faces, phi_faces = openmodes.core.z_efie_faces_self(nodes_o,
                                         basis_o.mesh.polygons, s,
                                         integration_rule.xi_eta,
                                         integration_rule.weights, *singular_terms)

        transform_L_s = transform_L_o
        transform_S_s = transform_S_o

    else:
        # calculate mutual impedance

        num_faces_s = len(basis_s.mesh.polygons)

        A_faces, phi_faces = openmodes.core.z_efie_faces_mutual(nodes_o,
                                basis_o.mesh.polygons, nodes_s,
                                basis_s.mesh.polygons, s, 
                                integration_rule.xi_eta,
                                integration_rule.weights)

        transform_L_s, transform_S_s = basis_s.transformation_matrices

    if np.any(np.isnan(A_faces)) or np.any(np.isnan(phi_faces)):
        raise ValueError("NaN returned in impedance matrix")

    L = transform_L_o.dot(transform_L_s.dot(A_faces.reshape(num_faces_o*3,
                                                            num_faces_s*3,
                                                            order='C').T).T)
    S = transform_S_o.dot(transform_S_s.dot(phi_faces.T).T)

    L *= mu_0/(4*pi)
    S *= 1/(pi*epsilon_0)
    return L, S


class Operator(object):
    "A base class for operator equations"

    def impedance(self, s, parent_o, parent_s):
        """Evaluate the self and mutual impedances of all parts in the
        simulation. Return an `ImpedancePart` object which can calculate
        several derived impedance quantities

        Parameters
        ----------
        s : number
            Complex frequency at which to calculate impedance (in rad/s)
        parent : Part
            Only this part and its sub-parts will be calculated

        Returns
        -------
        impedance_matrices : ImpedanceParts
            The impedance matrix object which can represent the impedance of
            the object in several ways.
        """

        matrices = {}

        # TODO: cache individual part impedances to avoid repetition?
        # May not be worth it because mutual impedances cannot be cached
        # except in specific cases such as arrays, and self terms may be
        # invalidated by green's functions which depend on coordinates

        for part_o in parent_o.iter_single():
            for part_s in parent_s.iter_single():
                if self.reciprocal and (part_s, part_o) in matrices:
                    # use reciprocity to avoid repeated calculation
                    res = matrices[part_s, part_o].T
                else:
                    res = self.impedance_single_parts(s, part_o, part_s)
                matrices[part_o, part_s] = res

        return ImpedanceParts(s, parent_o, parent_s, matrices, type(res))

    def source_plane_wave(self, e_inc, jk_inc, parent):
        """Evaluate the source vectors due to an incident plane wave, returning
        separate vectors for each part.

        Parameters
        ----------
        e_inc: ndarray
            incident field polarisation in free space
        jk_inc: ndarray
            incident wave vector in free space
        parent: Part
            The part for which to calculate the source vector

        Returns
        -------
        V : list of ndarray
            the source vector for each part
        """

        vector = VectorParts(parent, self.basis_container, dtype=np.complex128)

        for part in parent.iter_single():
            vector[part] = self.source_plane_wave_single_part(part, e_inc, jk_inc)

        return vector


class EfieOperator(Operator):
    """An operator for the electric field integral equation, discretised with
    respect to some set of basis functions. Assumes that Galerkin's method is
    used, such that the testing functions are the same as the basis functions.
    """
    reciprocal = True

    def __init__(self, integration_rule, basis_container,
                 greens_function=FreeSpaceGreensFunction()):
        self.basis_container = basis_container
        self.integration_rule = integration_rule
        self.greens_function = greens_function

    def impedance_single_parts(self, s, part_o, part_s=None):
        """Calculate a self or mutual impedance matrix at a given complex
        frequency

        Parameters
        ----------
        s : complex
            Complex frequency at which to calculate impedance
        part_o : SinglePart
            The observing part, which must be a single part, not a composite
        part_s : SinglePart, optional
            The source part, if not specified will default to observing part
        """

        # if source part is not given, default to observer part
        part_s = part_s or part_o

        basis_o = self.basis_container[part_o]
        basis_s = self.basis_container[part_s]

        if isinstance(self.greens_function, FreeSpaceGreensFunction):
            if isinstance(basis_o, LinearTriangleBasis):
                L, S = impedance_rwg_efie_free_space(s, self.integration_rule,
                                                     basis_o, part_o.nodes,
                                                     basis_s, part_s.nodes,
                                                     part_o == part_s)
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError

        if issubclass(self.basis_container.basis_class, LoopStarBasis):
            return EfieImpedanceMatrixLoopStar(s, L, S, basis_o, basis_s, self,
                                               part_o, part_s)
        else:
            return EfieImpedanceMatrix(s, L, S, basis_o, basis_s, self, part_o,
                                       part_s)

    def source_plane_wave_single_part(self, part, e_inc, jk_inc):
        """Evaluate the source vector due to the incident wave

        Parameters
        ----------
        e_inc: ndarray
            incident field polarisation in free space
        jk_inc: ndarray
            incident wave vector in free space

        Returns
        -------
        V : ndarray
            the source "voltage" vector
        """
        basis = self.basis_container[part]

        if (isinstance(basis, LinearTriangleBasis) and
                isinstance(self.greens_function, FreeSpaceGreensFunction)):

            incident_faces = openmodes.core.v_efie_faces_plane_wave(part.nodes,
                                        basis.mesh.polygons, 
                                        self.integration_rule.xi_eta,
                                        self.integration_rule.weights,
                                        e_inc, jk_inc)

            transform_L, _ = basis.transformation_matrices
            return transform_L.dot(incident_faces.flatten())
        else:
            raise NotImplementedError("%s, %s" % (str(type(basis)),
                                              str(type(self.greens_function))))

    def far_field_radiation(self, s, part, current_vec, direction):
        """Calculate the far-field radiation in a given direction. Note that
        all calculations will be referenced to the global origin. This means
        that the contributions of different parts can be added together if
        their current solutions were calculated consistently.

        Parameters
        ----------
        s : complex
            The complex frequency
        part : SinglePart
            The part for which to calculate far-field radiation.
        current_vec : ndarray
            The current solution defined over basis functions
        direction : (num_direction, 3) ndarray
           The directions in which to calculate radiation as cartesian vectors
        xi_eta : (num_points, 2) ndarray
            The barycentric integration points
        weights : (num_points) ndarray
            The integration weights

        Returns
        -------
        pattern : (num_direction, 3) ndarray
            The radiation pattern in each direction. When multiplied by
            $exp(jkr)/r$, this gives the far-field component of the
            electric field at distance r.
        """

        raise NotImplementedError

        # ensure that all directions are unit vectors
        direction = np.atleast_2d(direction)
        direction /= np.sqrt(np.sum(direction**2, axis=1))

        basis = self.basis_container[part]
        r, currents = basis.interpolate_function(current_vec,
                                                 self.integration_rule,
                                                 nodes=part.mesh.nodes,
                                                 scale_area=False)
