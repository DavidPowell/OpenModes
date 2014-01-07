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

def spherical_multipoles(max_l, jk, points, charge, current):
    """Calculate the multipole coefficients in a spherical basis

    Using the formulas from Jackson, Sect 9.10, currently neglecting the
    magnetic polarisation.

    Moments are calculated under the assumption of a free-space background.

    Parameters
    ----------
    max_l : integer
        The maximum order of l to consider
    jk : complex
        The complex wave-number
    points : array
        The array of points at which to integrate
    weights : array
        The weights which should be applied to each point, will have units
        of area so that final integral is dimensionally correct
    charge : array
        The charge calculated at each point
    current : array
        The current vector calculated at each point

    Returns
    -------
    a_e : (max_l+1 x max_l+1) array
        The electric multipole coefficients of order l, m for m > 0
    a_m : (max_l+1 x max_l+1) array
        The magnetic multipole coefficients of order l, m for m > 0
    """

    raise NotImplementedError

    num_l = max_l + 1

    # convert cartesian to (r, theta, phi) spherical coordinates
    spherical = np.empty_like(points)
    spherical[:, 0] = np.sqrt(np.sum(points**2, axis=1))
    spherical[:, 1] = np.arccos(points[:, 2]/spherical[:, 0])
    spherical[:, 2] = np.arctan2(points[:, 1], points[:, 0])
    
    
    # calculate the spherical Bessel functions
    
    l_terms_e = np.empty((len(points), num_l), np.complex128)
    l_terms_m = np.empty_like(l_terms_e)
    #Y_lmc = np.empty((len(points), num_l, 2*num_l+1), np.complex128)
    
    for n, r in enumerate(spherical[:, 0]):
        jl, djl = scipy.special.sph_jn(-1j*jk*r)
        l_terms_e[n] = (c*charge[n]*(r*jl + djl) + jk*np.dot(points[n], current[n])*jl)
        #Y_lmc[n] = 
        #cos_theta = points[n, 2]/spherical[n, 0]
        #legendre = scipy.special.lpmn(cos_theta
        
    a_e = np.zeros((num_l, num_l), np.complex128)
    a_m = np.zeros_like(a_e)


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
        for i in xrange(3):
            for j in xrange(3):
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
