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
"""This module contains most of the matrix construction routines which are
fully specific to RWG and related basis functions"""

import numpy as np

from openmodes.operator.singularities import singular_impedance_rwg
from openmodes.core import z_mfie_faces_self, z_mfie_faces_mutual
from openmodes.core import z_efie_faces_self, z_efie_faces_mutual
from openmodes.constants import epsilon_0, mu_0, pi, c


def impedance_curl_G(s, integration_rule, basis_o, nodes_o, basis_s, nodes_s,
                     normals, self_impedance, epsilon, mu, num_singular_terms,
                     singularity_accuracy, tangential_form):
    """Calculates the impedance matrix corresponding to the equation:
    fm . curl(G) . fn
    for RWG and related basis functions"""

    transform_o, _ = basis_o.transformation_matrices
    num_faces_o = len(basis_o.mesh.polygons)

    gamma_0 = s/c*np.sqrt(epsilon*mu)

    if self_impedance:
        # calculate self impedance

        singular_terms = singular_impedance_rwg(basis_o, operator="MFIE",
                                                tangential_form=tangential_form,
                                                num_terms=num_singular_terms,
                                                rel_tol=singularity_accuracy,
                                                normals=normals)

        if np.any(np.isnan(singular_terms[0])):
            raise ValueError("NaN returned in singular impedance terms")

        num_faces_s = num_faces_o
        Z_faces = z_mfie_faces_self(nodes_o, basis_o.mesh.polygons,
                                    basis_o.mesh.polygon_areas, gamma_0,
                                    integration_rule.xi_eta,
                                    integration_rule.weights, normals,
                                    tangential_form, *singular_terms)

        transform_s = transform_o

    else:
        # calculate mutual impedance
        num_faces_s = len(basis_s.mesh.polygons)

        Z_faces = z_mfie_faces_mutual(nodes_o, basis_o.mesh.polygons,
                                      nodes_s, basis_s.mesh.polygons,
                                      gamma_0, integration_rule.xi_eta,
                                      integration_rule.weights, normals,
                                      tangential_form)

        transform_s, _ = basis_s.transformation_matrices

    if np.any(np.isnan(Z_faces)):
        raise ValueError("NaN returned in impedance matrix")

    Z = transform_o.dot(transform_s.dot(Z_faces.reshape(num_faces_o*3,
                                                        num_faces_s*3,
                                                        order='C').T).T)
    return Z


def impedance_G(s, integration_rule, basis_o, nodes_o, basis_s, nodes_s,
                self_impedance, epsilon, mu, num_singular_terms,
                singularity_accuracy):
    """Calculates the impedance matrix corresponding to the equation:
    fm . (I + grad grad) G . fn
    for RWG or loop-star basis functions"""

    transform_L_o, transform_S_o = basis_o.transformation_matrices
    num_faces_o = len(basis_o.mesh.polygons)

    gamma_0 = s/c*np.sqrt(epsilon*mu)

    if (self_impedance):
        # calculate self impedance

        singular_terms = singular_impedance_rwg(basis_o, operator="EFIE",
                                                tangential_form=True,
                                                num_terms=num_singular_terms,
                                                rel_tol=singularity_accuracy)
        if (np.any(np.isnan(singular_terms[0])) or
                np.any(np.isnan(singular_terms[1]))):
            raise ValueError("NaN returned in singular impedance terms")

        num_faces_s = num_faces_o
        A_faces, phi_faces = z_efie_faces_self(nodes_o,
                                               basis_o.mesh.polygons, gamma_0,
                                               integration_rule.xi_eta,
                                               integration_rule.weights,
                                               *singular_terms)

        transform_L_s = transform_L_o
        transform_S_s = transform_S_o

    else:
        # calculate mutual impedance

        num_faces_s = len(basis_s.mesh.polygons)

        A_faces, phi_faces = z_efie_faces_mutual(nodes_o,
                                                 basis_o.mesh.polygons,
                                                 nodes_s,
                                                 basis_s.mesh.polygons,
                                                 gamma_0,
                                                 integration_rule.xi_eta,
                                                 integration_rule.weights)

        transform_L_s, transform_S_s = basis_s.transformation_matrices

    if np.any(np.isnan(A_faces)) or np.any(np.isnan(phi_faces)):
        raise ValueError("NaN returned in impedance matrix")

    L = transform_L_o.dot(transform_L_s.dot(A_faces.reshape(num_faces_o*3,
                                                            num_faces_s*3,
                                                            order='C').T).T)
    S = transform_S_o.dot(transform_S_s.dot(phi_faces.T).T)

    L *= mu*mu_0/(4*pi)
    S *= 1/(pi*epsilon*epsilon_0)
    return L, S
