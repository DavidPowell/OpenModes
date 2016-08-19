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


def multipole_fixed(max_l, points):
    """Calculate all frequency-independent quantities for the multipole
    decomposition.

    These depend only on the coordinates of the points, and need only be
    calculated once for each object and reused at multiple frequencies"""

    num_l = max_l + 1
    l = np.arange(num_l)[:, None]
    m_pos = np.arange(num_l)[None, :]
    m = np.hstack((m_pos, np.arange(-max_l, 0)[None, :]))

    r = np.sqrt(np.sum(points**2, axis=1))
    theta = np.arccos(points[:, 2]/r)
    phi = np.arctan2(points[:, 1], points[:, 0])

    st = np.sin(theta)
    ct = np.cos(theta)
    sp = np.sin(phi)
    cp = np.cos(phi)

    # calculate the unit vectors on the sphere
    r_hat = points/r[:, None]
    theta_hat = np.stack((ct*cp, ct*sp, -st), axis=-1)
    phi_hat = np.stack((-sp, cp, np.zeros_like(sp)), axis=-1)

    exp_imp = np.exp(-1j*m[None, :, :]*phi[:, None, None])

    P_lm = np.zeros((len(ct), len(l), m.shape[1]))
    dP_lm = np.zeros_like(P_lm)

    for point_count, ct_n in enumerate(ct):

        # associated Legendre function and its derivative
        P_lmp, dP_lmp = scipy.special.lpmn(max_l, max_l, ct_n)
        P_lmp = P_lmp.T
        dP_lmp = dP_lmp.T

        # Calculate negative values of m from positive
        P_lmn, dP_lmn = scipy.special.lpmn(-max_l, max_l, ct_n)
        P_neg = (-1)**m_pos*factorial(l-m_pos)/factorial(l+m_pos)
        P_lmn = P_neg*P_lmp
        dP_lmn = P_neg*dP_lmp

        # combine positive and negative P_lmn
        P_lm[point_count] = np.hstack((P_lmp,  P_lmn[:, :0:-1]))
        dP_lm[point_count] = np.hstack((dP_lmp, dP_lmn[:, :0:-1]))

    # theta derivative of P_lm(cos\theta)
    tau_lm = -st[:, None, None]*dP_lm
    pi_lm = P_lm*m/st[:, None, None]

    return (r, theta, phi, r_hat, theta_hat, phi_hat, P_lm, exp_imp,
            tau_lm, pi_lm)


def spherical_multipoles(max_l, k, points, current, current_M, eta=eta_0,
                         fixed_terms=None):
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
    num_points = len(points)

    l = np.arange(num_l)[:, None]
    m_pos = np.arange(num_l)[None, :]
    m = np.hstack((m_pos, np.arange(-max_l, 0)[None, :]))

    # Use precalculated fixed terms if provided
    try:
        (r, theta, phi, r_hat, theta_hat, phi_hat, P_lm, exp_imp, tau_lm,
         pi_lm) = fixed_terms
    except:
        (r, theta, phi, r_hat, theta_hat, phi_hat, P_lm, exp_imp, tau_lm,
         pi_lm) = multipole_fixed(max_l, points)

    jl = np.empty((num_points, num_l+1, 1))
    djl = np.empty_like(jl)

    # spherical Bessel functions must be calculated per point
    for n in range(num_points):
        jl[n, :, 0], djl[n, :, 0] = scipy.special.sph_jn(max_l, k*r[n])

    ll = l[None, 1:]
    # Riccati Bessel function plus its second derivative
    ric_plus_second = np.zeros((num_points, num_l, 1))
    ric_plus_second[:, 1:] = ll*(ll+1)/(2*ll+1)*(jl[:, :-2]+jl[:, 2:])

    # First derivative, divided by kr
    ric_der = np.zeros_like(ric_plus_second)
    ric_der[:, 1:] = ((ll+1)*jl[:, :-2]-ll*jl[:, 2:])/(2*ll+1)

    # components of current
    J_r = np.sum(r_hat*current, axis=1)[:, None, None]
    J_theta = np.sum(theta_hat*current, axis=1)[:, None, None]
    J_phi = np.sum(phi_hat*current, axis=1)[:, None, None]
    M_r = np.sum(r_hat*current_M, axis=1)[:, None, None]
    M_theta = np.sum(theta_hat*current_M, axis=1)[:, None, None]
    M_phi = np.sum(phi_hat*current_M, axis=1)[:, None, None]

    # indices l, m.
    # Note that for m, negative indices are used for negative m,
    # and values with |m| > l should be ignored
    a_e = np.sum(exp_imp*(ric_plus_second*P_lm*J_r +
                 ric_der*(tau_lm*J_theta - 1j*pi_lm*J_phi) +
                 jl[:, :-1]*(1j*pi_lm*M_theta + tau_lm*M_phi)), axis=0)
    a_m = np.sum(exp_imp*(jl[:, :-1]*(1j*pi_lm*J_theta + tau_lm*J_phi) +
                 ric_plus_second*P_lm*M_r +
                 ric_der*(tau_lm*M_theta - 1j*pi_lm*M_phi)), axis=0)

    # Ignore divide by zero and resulting NaN, which will occur for invalid
    # combinations of l, m
    with np.errstate(invalid='ignore', divide='ignore'):
        common_factor = (np.sqrt(eta)*k**2/(2*np.pi) *
                         np.sqrt(factorial(l-m)/factorial(l+m)) /
                         np.sqrt(l*(l+1)))
        a_e *= (1j)**(l-1)*common_factor
        a_m *= (1j)**(l-1)*common_factor

    return a_e, a_m


def far_fields(a_e, a_m, theta, phi):
    """Calculate far fields from the multipole decomposition

    Excludes the r dependent terms

    From Jackson sections 3.5, 9.6, 9.9

    Parameters
    ----------
    a_e, a_m: ndarray(max_l, 2*max_l+1)
        The multipole coefficients
    theta, phi: scalar
        polar angles of observation

    Returns
    -------
    E, H: ndarray(3)
        The electric and magnetic fields at the observation angle
    """

    # TODO: Allow theta and phi to be arrays
    # TODO: Store X_lm to enable reuse at multiple frequencies
    # TODO: Reuse calculated P_lm from spherical_multipoles

    num_l = a_e.shape[0]
    max_l = num_l - 1

    st = np.sin(theta)
    ct = np.cos(theta)
    sp = np.sin(phi)
    cp = np.cos(phi)
    r_hat = np.stack((st*cp, st*sp, ct), axis=-1)

    l = np.arange(num_l)[:, None]
    m_pos = np.arange(num_l)[None, :]
    m = np.hstack((m_pos, np.arange(-max_l, 0)[None, :]))

    # associated Legendre function and its derivative
    P_lm, dP_lm = scipy.special.lpmn(max_l, max_l, ct)
    P_lm = P_lm.T

    Y_lm_pos = np.sqrt((2*l+1)/(4*np.pi)*factorial(l-m_pos)/factorial(l+m_pos))*P_lm*np.exp(1j*m_pos*phi)
    Y_lm_neg = (-1)**m_pos*Y_lm_pos.conj()
    Y_lm = np.hstack((Y_lm_pos,  Y_lm_neg[:, :0:-1]))

    # Angular momentum operator L acting on Y_lm
    # Note that for m=l and m=-l, roll operator gives a spurious result, but
    # this is cancelled by (m-l) and (m+l) terms respectively.
    # As the calculations include invalid combinations of l and m,
    # warnings should be suppressed for these calculations.
    with np.errstate(invalid='ignore'):
        Y_lm_p = np.sqrt((l-m)*(l+m+1))*np.roll(Y_lm, -1, axis=1)
        Y_lm_m = np.sqrt((l+m)*(l-m+1))*np.roll(Y_lm, 1, axis=1)
        X = np.stack((0.5*(Y_lm_p+Y_lm_m), -0.5j*(Y_lm_p-Y_lm_m), m*Y_lm),
                     axis=-1) / np.sqrt(l*(l+1))[:, :, None]

    H = np.zeros(3, np.complex128)
    for l in range(1, max_l+1):
        for m in range(-l, l+1):
            H += (1j)**(l+1)*(a_e[l, m]*X[l, m] + a_m[l, m]*np.cross(r_hat, X[l, m]))
    E = eta_0*np.cross(H, r_hat)
    return E, H


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
