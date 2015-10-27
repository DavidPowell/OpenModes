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
"""Calculate multipole decompositions"""

import numpy as np
import scipy.special

from openmodes.constants import c


def spherical_multipoles(max_l, gamma, points, weights, current, exp_phi=None):
    """Calculate the multipole coefficients in a spherical basis

    Using the formulas from:
    P. Grahn, A. Shevchenko, and M. Kaivola, “Electromagnetic multipole theory
    for optical nanomaterials,” New Journal of Physics,
    vol. 14, no. 9, pp. 093033–093033, Jun. 2012.

    Parameters
    ----------
    max_l : integer
        The maximum order of multipole to consider
    gamma : complex
        The complex wave-number in the background medium
    points : array
        The array of points at which to integrate
    weights : array
        The weights which should be applied to each point, will have units
        of area so that final integral is dimensionally correct
    current : array
        The current vector calculated at each point

    Returns
    -------
    a_e : (max_l+1 x max_l+1) array
        The electric multipole coefficients of order l, m for m > 0
    a_m : (max_l+1 x max_l+1) array
        The magnetic multipole coefficients of order l, m for m > 0
    """

    num_l = max_l + 1
    a_e = np.empty((num_l, num_l), np.complex128)
    a_m = np.empty((num_l, num_l), np.complex128)

    l = np.arange(max_l)

    for point in points:
        # convert cartesian to (r, theta, phi) spherical coordinates
        r = np.sqrt(np.sum(point**2))
        theta = np.arccos(point[2]/r)
        phi = np.arctan2(point[1], point[0])

        # calculate the unit vectors on the sphere
        st = np.sin(theta)
        ct = np.cos(theta)
        sp = np.sin(phi)
        cp = np.cos(phi)
        
        r_hat = point/r
        theta_hat = np.array((ct*cp, ct*sp, -st))
        phi_hat = np.array(-sp, cp, np.zeros(len(points)))

        kr = gamma*r/1j
        jl, djl = scipy.special.sph_jn(kr)
        
        # index l
        ric_plus_second = (l*(l+1))*jl/kr
        ric_der = jl[:-1] + jl/kr
        
        l_terms_e[n] = (c*charge[n]*(r*jl + djl) + jk*np.dot(points[n], current[n])*jl)
        #Y_lmc[n] =
        #cos_theta = points[n, 2]/spherical[n, 0]
        #legendre = scipy.special.lpmn(cos_theta



def cartesian_multipoles(points, charge, current, s, electric_order=1,
                         magnetic_order=1):
    """Calculate the electric and magnetic multipole moments up to the
    specified order

    Parameters
    ----------
    points : ndarray (num_points, 3)
        The points at which charge and current are calculated
    charge : ndarray (num_points)
        The charge at each point
    current : ndarray (num_points, 3)
        The current vector at each point
    weights : ??
    s : number
        complex frequency
    electric_order : integer, optional
        The maximum order of electric multipoles (currently maximum 2)
    magnetic_order : integer, optional
        The maximum order of magnetic multipoles (currently maximum 1)

    Returns
    -------
    p : ndarray
        electric dipole moment
    m : ndarray
        magnetic dipole moment

    Moments are calculated relative to zero coordinate - does not affect
    the electric dipole, but will affect the magnetic dipole moment and
    any other higher-order multipoles

    The moments are 'primitive moments' as defined by Raab and de Lange
    """

    electric_moments = []
    magnetic_moments = []

    # electric dipole moment
    if electric_order >= 1:
        electric_moments.append(np.sum(points[:, :]*charge[:, None], axis=0)/s)

    # electric quadrupole moment
    if electric_order >= 2:
        quad = np.empty((3, 3), np.complex128)
        for i in range(3):
            for j in range(3):
                quad[i, j] = np.sum(points[:, i]*points[:, j]*charge[:])
        electric_moments.append(quad/s)

    if electric_order >= 3:
        raise NotImplementedError("Electric moments only up to quadrupole")

    # magnetic dipole moment
    if magnetic_order >= 1:
        magnetic_moments.append(0.5*np.sum(np.cross(points[:, :], current[:, :]), axis=0))

    if magnetic_order >= 2:
        raise NotImplementedError("Magnetic moments only up to dipole")

    return electric_moments, magnetic_moments
