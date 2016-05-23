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
from scipy.special import factorial

from openmodes.constants import eta_0


def spherical_multipoles(max_l, k, points, current, current_M, eta=eta_0):
    """Calculate the multipole coefficients in a spherical basis

    Using the formulas from:
    P. Grahn, A. Shevchenko, and M. Kaivola, “Electromagnetic multipole theory
    for optical nanomaterials,” New Journal of Physics,
    vol. 14, no. 9, pp. 093033–093033, Jun. 2012.

    Parameters
    ----------
    max_l : integer
        The maximum order of multipole to consider
    k : complex
        The wave-number in the background medium
    points : array
        The array of points at which to integrate
    current : array
        The current vector calculated at each point. Any weights from
        the integration rule should already be applied.
    current_M : array
        The equivalent magnetic current for surface equivalent description of
        dielectrics

    Returns
    -------
    a_e : (max_l+1 x 2*max_l+1) array
        The electric multipole coefficients of order l, m
    a_m : (max_l+1 x 2*max_l+1) array
        The magnetic multipole coefficients of order l, m
    """

    num_l = max_l + 1
    num_m = 2*max_l + 1

    # indices l, m.
    # Note that for m, negative indices are used for negative m,
    # and values with |m| > l should be ignored
    a_e = np.zeros((num_l, num_m), np.complex128)
    a_m = np.zeros((num_l, num_m), np.complex128)

    l = np.arange(num_l)[:, None]
    m_pos = np.arange(num_l)[None, :]
    m = np.hstack((m_pos, np.arange(-max_l, 0)[None, :]))

    for J, M, point in zip(current, current_M, points):

        # Convert cartesian to (r, theta, phi) spherical coordinates
        # Convention as per Jackson, Figure 3.1
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
        phi_hat = np.array((-sp, cp, 0))

        kr = k*r
        jl, djl = scipy.special.sph_jn(max_l+1, kr)
        # reshape to be size (num_l x 1)
        jl = jl[:, None]
        djl = djl[:, None]

        # Riccati Bessel function plus its second derivative
        ll = l[1:]
        ric_plus_second = np.zeros((max_l+1, 1), np.complex128)
        ric_plus_second[1:] = ll*(ll+1)/(2*ll+1)*(jl[:-2]+jl[2:])
        # First derivative, divided by kr
        ric_der = np.zeros((max_l+1, 1), np.complex128)
        ric_der[1:] = ((ll+1)*jl[:-2]-ll*jl[2:])/(2*ll+1)

        # associated Legendre function and its derivative
        P_lm, dP_lm = scipy.special.lpmn(max_l, max_l, ct)
        P_lm = P_lm.T
        dP_lm = dP_lm.T

        # Calculate negative values of m from positive
        P_lmn, dP_lmn = scipy.special.lpmn(-max_l, max_l, ct)
        P_neg = (-1)**m_pos*factorial(l-m_pos)/factorial(l+m_pos)
        P_lmn = P_neg*P_lm
        dP_lmn = P_neg*dP_lm

        # combine positive and negative P_lmn
        P_lm = np.hstack((P_lm,  P_lmn[:, :0:-1]))
        dP_lm = np.hstack((dP_lm, dP_lmn[:, :0:-1]))

        # theta derivative of P_lm(cos\theta)
        tau_lm = -st*dP_lm
        pi_lm = P_lm*m/st

        exp_imp = np.exp(-1j*m*phi)

        # components of current
        J_r = np.dot(r_hat, J)
        J_theta = np.dot(theta_hat, J)
        J_phi = np.dot(phi_hat, J)
        M_r = np.dot(r_hat, M)
        M_theta = np.dot(theta_hat, M)
        M_phi = np.dot(phi_hat, M)

        a_e += exp_imp*(ric_plus_second*P_lm*J_r + ric_der*(tau_lm*J_theta - 1j*pi_lm*J_phi))
        a_e += exp_imp*jl[:-1]*(1j*pi_lm*M_theta + tau_lm*M_phi)
        a_m += exp_imp*jl[:-1]*(1j*pi_lm*J_theta + tau_lm*J_phi)
        a_m += exp_imp*(ric_plus_second*P_lm*M_r + ric_der*(tau_lm*M_theta - 1j*pi_lm*M_phi))

    # Ignore divide by zero and resulting NaN, which will occur for invalid
    # combinations of l, m
    with np.errstate(invalid='ignore', divide='ignore'):
        common_factor = np.sqrt(eta)*k**2/(2*np.pi)*np.sqrt(factorial(l-m)/factorial(l+m))/np.sqrt(l*(l+1))        
        a_e *= (-1j)**(l-1)*common_factor
        a_m *= (-1j)**(l+1)*common_factor

    return a_e, a_m


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
