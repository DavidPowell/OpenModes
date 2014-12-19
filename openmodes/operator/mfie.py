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

from openmodes.constants import epsilon_0, mu_0, pi, c
import openmodes.core
from openmodes.basis import LinearTriangleBasis
from openmodes.impedance import ImpedanceMatrix

from openmodes.operator.operator import Operator, FreeSpaceGreensFunction


#def Z_pair(nodes_o, nodes_s, xi_eta, weights, normal, jk_0):
#    num_points = len(weights)
##    for node_o in xrange(3):
##        for node_s in xrange(3):
#    for point_o, (xi_o, eta_o) in enumerate(xi_eta):
#        zeta_o = 1 - xi_o - eta_o
#        r_o = xi_o*nodes_o[:, 0] + eta_o*nodes_o[:, 1] + zeta_o*nodes_o[:, 2]
#        for point_s, (xi_s, eta_s) in enumerate(xi_eta):
#            zeta_s = 1 - xi_s - eta_s
#            r_s = xi_s*nodes_s[:, 0] + eta_s*nodes_s[:, 1] + zeta_s*nodes_s[:, 2]
#            
#            r = r_o - r_s
#            r_mag = np.sqrt(sum(r**2))
#            np.exp(-jk_0*r_mag)/r_mag**3*(1+jk_0*r_mag)
#                    
#
#def z_mfie_faces_self(nodes, triangles, s, xi_eta, weights, normals):
#
#    num_triangles = len(triangles)
#
#    Z_faces = np.empty((num_triangles, 3, num_triangles, 3), np.complex128)
#    jk_0 = s/c
#    
#    for n in xrange(num_triangles):
#        for m in xrange(num_triangles):
#            if m == n:
#                Z_faces[m, :, n, :] = 0.0
#            else:
#                Z_faces[m, :, n, :] = Z_pair(nodes[triangles[m]], nodes[triangles[n]],
#                                         xi_eta, weights, normals[m], jk_0)
#    return Z_faces


def impedance_rwg_mfie_free_space(s, integration_rule, basis_o, nodes_o,
                                  basis_s, nodes_s, normals, self_impedance):
    """EFIE derived Impedance matrix for RWG or loop-star basis functions"""

    transform_o, _ = basis_o.transformation_matrices
    num_faces_o = len(basis_o.mesh.polygons)

    if self_impedance:
        # calculate self impedance

        num_faces_s = num_faces_o
        Z_faces = openmodes.core.z_mfie_faces_self(nodes_o,
                                         basis_o.mesh.polygons, s,
                                         integration_rule.xi_eta,
                                         integration_rule.weights, normals)

        transform_s = transform_o

    else:
        # calculate mutual impedance
        raise NotImplementedError

        num_faces_s = len(basis_s.mesh.polygons)

        transform_Z_s, _ = basis_s.transformation_matrices

    if np.any(np.isnan(Z_faces)):
        raise ValueError("NaN returned in impedance matrix")

    Z = transform_o.dot(transform_s.dot(Z_faces.reshape(num_faces_o*3,
                                                        num_faces_s*3,
                                                        order='C').T).T)#/4/pi
    # Gram matrix is wrong, it is a question of same triangle, not the same basis function!                                                        
    #if self_impedance:
    #    Z = Z/(4*pi) + 0.5*basis_o.gram_matrix

    return Z


class MfieOperator(Operator):
    """An operator for the magnetic field integral equation, discretised with
    respect to some set of basis functions. Assumes that Galerkin's method is
    used, such that the testing functions are the same as the basis functions.
    """
    reciprocal = False
    source_field = "magnetic_field"
    source_cross = False #True

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

        normals = basis_o.mesh.surface_normals

        if isinstance(self.greens_function, FreeSpaceGreensFunction):
            if isinstance(basis_o, LinearTriangleBasis):
                Z = impedance_rwg_mfie_free_space(s, self.integration_rule,
                                                  basis_o, part_o.nodes,
                                                  basis_s, part_s.nodes,
                                                  normals,
                                                  part_o == part_s)
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError

        return ImpedanceMatrix(s, Z, basis_o, basis_s, self, part_o, part_s)
