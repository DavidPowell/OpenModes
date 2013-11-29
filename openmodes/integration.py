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

"""
Integration routines and quadrature rules over triangles

Created on Fri Apr 27 09:57:42 2012

@author: dap124
"""

import numpy as np
import numpy.linalg as la


def get_dunavant_rule(tri_rule):
    """Calculate the symmetric quadrature rule over a triangle as given in
    D. A. Dunavant, Int. J. Numer. Methods Eng. 21, 1129 (1985).

    Parameters
    ----------
    n : integer
        The order of the rule (maximum 20)

    Returns
    -------
    xi_eta : array
        The barycentric coordinates (xi, eta) of the quadrature points
    weights : array
        The weights, normalised to sum to 1/2
    """

    import dunavant

    dunavant_order = dunavant.dunavant_order_num(tri_rule)
    xi_eta_eval, weights = dunavant.dunavant_rule(tri_rule, dunavant_order)
    xi_eta_eval = np.asfortranarray(xi_eta_eval.T)
    # scale the weights to 0.5
    weights = np.asfortranarray((weights[:, None]*0.5/sum(weights)).T)

    return xi_eta_eval, weights


def cartesian_to_barycentric(r, nodes):
    """Convert cartesian coordinates to barycentric (area coordinates) in a
    triangle

    r - Nx2 array of cartesian coordinates
    nodes - 3x2 array of nodes of the triangle
    """

    T = np.array(((nodes[0, 0] - nodes[2, 0], nodes[1, 0] - nodes[2, 0]),
                 (nodes[0, 1] - nodes[2, 1], nodes[1, 1] - nodes[2, 1])))

    bary_coords = np.empty((len(r), 3))
    bary_coords[:, :2] = la.solve(T, (r[:, :2]-nodes[None, 2, :2]).T).T
    bary_coords[:, 2] = 1.0 - bary_coords[:, 1] - bary_coords[:, 0]
    return bary_coords


def triangle_electric_dipole(vertices, xi_eta, weights):
    """Calculate the dipole moment of a triangle with constant unit charge

    Parameters
    ----------
    vertices : ndarray
        the vertices which define the triangle
    xi_eta : ndarray
        the points of the quadrature rule in barycentric form
    weights : ndarray
        the weights of the integration

    Returns
    -------
    p : ndarray
        the electric dipole moment of the triangle
    """

    r = ((vertices[0]-vertices[2])*xi_eta[:, 0, None] +
         (vertices[1]-vertices[2])*xi_eta[:, 1, None] +
         vertices[2])

    return np.sum(weights[0, :, None]*r, axis=0)
