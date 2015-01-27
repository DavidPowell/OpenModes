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
import logging

from openmodes.constants import epsilon_0, mu_0, pi
from openmodes.core import z_efie_faces_self, z_efie_faces_mutual
from openmodes.basis import LinearTriangleBasis, LoopStarBasis
from openmodes.impedance import (EfieImpedanceMatrix,
                                 EfieImpedanceMatrixLoopStar)

from openmodes.operator.operator import Operator, FreeSpaceGreensFunction
from openmodes.operator.singularities import singular_impedance_rwg


def impedance_rwg_efie_free_space(s, integration_rule, basis_o, nodes_o,
                                  basis_s, nodes_s, self_impedance,
                                  singularity_accuracy):
    """EFIE derived Impedance matrix for RWG or loop-star basis functions"""

    transform_L_o, transform_S_o = basis_o.transformation_matrices
    num_faces_o = len(basis_o.mesh.polygons)

    if (self_impedance):
        # calculate self impedance

        singular_terms = singular_impedance_rwg(basis_o, operator="EFIE",
                                                tangential_form=True,
                                                rel_tol=singularity_accuracy)
        if (np.any(np.isnan(singular_terms[0])) or
                np.any(np.isnan(singular_terms[1]))):
            raise ValueError("NaN returned in singular impedance terms")

        num_faces_s = num_faces_o
        A_faces, phi_faces = z_efie_faces_self(nodes_o,
                                               basis_o.mesh.polygons, s,
                                               integration_rule.xi_eta,
                                               integration_rule.weights,
                                               *singular_terms)

        transform_L_s = transform_L_o
        transform_S_s = transform_S_o

    else:
        # calculate mutual impedance

        num_faces_s = len(basis_s.mesh.polygons)

        A_faces, phi_faces = z_efie_faces_mutual(nodes_o,
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


class EfieOperator(Operator):
    """An operator for the electric field integral equation, discretised with
    respect to some set of basis functions. Assumes that Galerkin's method is
    used, such that the testing functions are the same as the basis functions.
    """
    reciprocal = True
    source_field = "electric_field"
    source_cross = False

    def __init__(self, integration_rule, basis_container,
                 greens_function=FreeSpaceGreensFunction(),
                 tangential_form=True, singularity_accuracy=1e-5):
        self.basis_container = basis_container
        self.integration_rule = integration_rule
        self.greens_function = greens_function
        self.singularities_accuracy = singularity_accuracy

        self.tangential_form = tangential_form
        if tangential_form:
            self.reciprocal = False
            self.source_cross = False
        else:
            raise NotImplementedError("Tangential EFIE")

        logging.info("Creating EFIE operator, tangential form: %s"
                     % str(tangential_form))

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
                                                     part_o == part_s,
                                                     self.singularities_accuracy)
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
