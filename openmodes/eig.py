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
Routines for solving linear and nonlinear eigenvalue problems
"""

#from openmodes.basis import LoopStarBasis
#from openmodes.impedance import EfieImpedanceMatrixLoopStar
import scipy.linalg as la
import numpy as np


def eig_linearised(Z, num_modes):
    """Solves a linearised approximation to the eigenvalue problem from
    the impedance calculated at some fixed frequency.

    The equation :math:`L = -s^2 S` is solved for `s`

    Parameters
    ----------
    Z : EfieImpedanceMatrixLoopStar
        The impedance matrix calculated in a loop-star basis
    num_modes : int
        The number of modes required.

    Returns
    -------
    s_mode : ndarray, complex
        The resonant frequencies of the modes (in Hz)
        The complex pole `s` corresponding to the mode's eigenfrequency
    j_mode : ndarray, complex
        Columns of this matrix contain the corresponding modal currents
    """

    #if not isinstance(Z, EfieImpedanceMatrixLoopStar):
    #    raise ValueError(
    #        "Loop-star basis functions required for linearised eigenvalues")

    star_range = Z.star_range_o

    if Z.basis_o.num_loops == 0:
        L_red = Z.L
    else:
        loop_range = Z.loop_range_o

        L_conv = la.solve(Z.L[loop_range, loop_range],
                          Z.L[loop_range, star_range])
        L_red = Z.L[star_range, star_range] - np.dot(Z.L[star_range, loop_range],
                                                   L_conv)

    # find eigenvalues, and star part of eigenvectors, for LS combined modes
    w, v_s = la.eig(Z.S[star_range, star_range], -L_red)

    if Z.basis_o.num_loops == 0:
        vr = v_s
    else:
        v_l = -np.dot(L_conv, v_s)
        vr = np.vstack((v_l, v_s))

#    import matplotlib.pyplot as plt
#    plt.loglog(w.real, w.imag, 'x')

    w_freq = np.sqrt(w)  # /2/np.pi
    # make sure real part is negative
    w_freq = np.where(w_freq.real > 0, -w_freq, w_freq)

    #plt.loglog(w_freq.real, w_freq.imag, 'x')

    #w_selected = np.ma.masked_array(w_freq, w_freq.real < w_freq.imag)
    w_selected = np.ma.masked_array(w_freq, abs(w_freq.real) > abs(w_freq.imag))
    which_modes = np.argsort(abs(w_selected.imag))[:num_modes]

    return w_freq[which_modes], vr[:, which_modes]


def eig_newton(func, lambda_0, x_0, lambda_tol=1e-8, max_iter=20,
               func_gives_der=False, args=[], weight='rayleigh symmetric'):
    """Solve a nonlinear eigenvalue problem by Newton iteration

    Parameters
    ----------
    func : function
        The function with input `lambda` which returns the matrix
    lambda_0 : complex
        The starting guess for the eigenvalue
    x_0 : ndarray
        The starting guess for the eigenvector
    lambda_tol : float
        The relative tolerance in the eigenvalue for convergence
    max_iter : int
        The maximum number of iterations to perform
    func_gives_der : boolean, optional
        If `True`, then the function also returns the derivative as the second
        returned value. If `False` finite differences will be used instead,
        which will have reduced accuracy
    args : list, optional
        Any additional arguments to be supplied to `func`
    weight : string, optional
        How to perform the weighting of the eigenvector

        'max element' : The element with largest magnitude will be preserved

        'rayleigh' : Rayleigh iteration for Hermition matrices will be used

        'rayleigh symmetric' : Rayleigh iteration for complex symmetric
        (i.e. non-Hermitian) matrices will be used

    Returns
    -------
    res : dictionary
        A dictionary containing the following members:

        `eigval` : the eigenvalue

        'eigvect' : the eigenvector

        'iter_count' : the number of iterations performed

        'delta_lambda' : the change in the eigenvalue on the final iteration


    See:
    1.  P. Lancaster, Lambda Matrices and Vibrating Systems.
        Oxford: Pergamon, 1966.

    2.  A. Ruhe, “Algorithms for the Nonlinear Eigenvalue Problem,”
        SIAM J. Numer. Anal., vol. 10, no. 4, pp. 674–689, Sep. 1973.

    """

    x_s = x_0
    lambda_s = lambda_0

    converged = False

    if not func_gives_der:
        # evaluate at an arbitrary nearby starting point to allow finite
        # differences to be taken
        #lambda_sm = lambda_0*(1+2*lambda_tol)
        lambda_sm = lambda_0*(1+10j*lambda_tol)
        T_sm = func(lambda_sm, *args)

    for iter_count in xrange(max_iter):
        if func_gives_der:
            T_s, T_ds = func(lambda_s, *args)
        else:
            T_s = func(lambda_s, *args)
            T_ds = (T_s - T_sm)/(lambda_s - lambda_sm)

        u = la.solve(T_s, np.dot(T_ds, x_s))

        # if known_vects is supplied, we should take this into account when
        # finding v
        if weight.lower() == 'max element':
            v_s = np.zeros_like(x_s)
            v_s[np.argmax(abs(x_s))] = 1.0
        elif weight.lower() == 'rayleigh':
            v_s = np.dot(T_s.T, x_s.conj())
        elif weight.lower() == 'rayleigh symmetric':
            v_s = np.dot(T_s.T, x_s)

        lambda_s1 = lambda_s - np.dot(v_s, x_s)/(np.dot(v_s, u))
        x_s1 = u/np.sqrt(np.sum(np.abs(u)**2))

        delta_lambda = abs((lambda_s1 - lambda_s)/lambda_s)
        converged = delta_lambda < lambda_tol

        # update variables for next iteration
        if not func_gives_der:
            lambda_sm = lambda_s
            T_sm = T_s

        lambda_s = lambda_s1
        x_s = x_s1
        #print x_s
        #print lambda_s

        if converged: break

    if not converged:
        raise ValueError("maximum iterations reached, no convergence")

    res = {'eigval': lambda_s, 'eigvec': x_s, 'iter_count': iter_count+1,
           'delta_lambda': delta_lambda}

    return res

def eig_newton_linear(Z, lambda_0, x_0, lambda_tol=1e-8, max_iter=20,
               G=None, weight='rayleigh symmetric'):
    """Solve a linear (generalised) eigenvalue problem by Newton iteration

    Parameters
    ----------
    Z : ndarray
        The matrix
    lambda_0 : complex
        The starting guess for the eigenvalue
    x_0 : ndarray
        The starting guess for the eigenvector
    lambda_tol : float
        The relative tolerance in the eigenvalue for convergence
    max_iter : int
        The maximum number of iterations to perform
    G : ndarray, optional
        The RHS matrix for the generalised problem. If omitted, the identity
        matrix will be used
    weight : string, optional
        How to perform the weighting of the eigenvector

        'max element' : The element with largest magnitude will be preserved

        'rayleigh' : Rayleigh iteration for Hermition matrices will be used

        'rayleigh symmetric' : Rayleigh iteration for complex symmetric
        (i.e. non-Hermitian) matrices will be used

    Returns
    -------
    res : dictionary
        A dictionary containing the following members:

        `eigval` : the eigenvalue

        'eigvect' : the eigenvector

        'iter_count' : the number of iterations performed

        'delta_lambda' : the change in the eigenvalue on the final iteration


    See:
    1.  P. Lancaster, Lambda Matrices and Vibrating Systems.
        Oxford: Pergamon, 1966.

    2.  A. Ruhe, “Algorithms for the Nonlinear Eigenvalue Problem,”
        SIAM J. Numer. Anal., vol. 10, no. 4, pp. 674–689, Sep. 1973.

    """

    x_s = x_0
    lambda_s = lambda_0

    converged = False

    for iter_count in xrange(max_iter):
        if G is not None:
            u = la.solve(Z-lambda_s*G, -np.dot(G, x_s))
        else:
            u = la.lu_solve(Z, x_s)
            
        if weight.lower() == 'max element':
            v_s = np.zeros_like(x_s)
            v_s[np.argmax(abs(x_s))] = 1.0
        elif weight.lower() == 'rayleigh':
            v_s = np.dot(Z.T, x_s.conj())
        elif weight.lower() == 'rayleigh symmetric':
            v_s = np.dot(Z.T, x_s)

        lambda_s1 = lambda_s - np.dot(v_s, x_s)/(np.dot(v_s, u))
        x_s1 = u/np.sqrt(np.sum(np.abs(u)**2))

        delta_lambda = abs((lambda_s1 - lambda_s)/lambda_s)
        converged = delta_lambda < lambda_tol

        lambda_s = lambda_s1
        x_s = x_s1
        #print x_s
        #print lambda_s

        if converged: break

    if not converged:
        raise ValueError("maximum iterations reached, no convergence")

    res = {'eigval': lambda_s, 'eigvec': x_s, 'iter_count': iter_count+1,
           'delta_lambda': delta_lambda}

    return res
