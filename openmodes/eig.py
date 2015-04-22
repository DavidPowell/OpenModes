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

import scipy.linalg as la
import numpy as np
import logging


def eig_linearised(Z, modes):
    """Solves a linearised approximation to the eigenvalue problem from
    the impedance calculated at some fixed frequency.

    The equation :math:`L = -s^2 S` is solved for `s`

    Parameters
    ----------
    Z : EfieImpedanceMatrixLoopStar
        The impedance matrix calculated in a loop-star basis
    modes : ndarray (int)
        A list or array of the mode numbers required

    Returns
    -------
    s_mode : ndarray, complex
        The resonant frequencies of the modes (in Hz)
        The complex pole `s` corresponding to the mode's eigenfrequency
    j_mode : ndarray, complex
        Columns of this matrix contain the corresponding modal currents
    """

    modes = np.asarray(modes)

    # check whether the operator has a null-space to be eliminated
    has_nullspace = (hasattr(Z.md['basis_o'], 'num_loops')
                     and Z.md['basis_o'].num_loops != 0)

    L = Z.matrices['L']
    S = Z.matrices['S']

    if has_nullspace:
        star_range = Z.star_range_o

        loop_range = Z.loop_range_o

        L_conv = la.solve(L[loop_range, loop_range],
                          L[loop_range, star_range])
        L_red = (L[star_range, star_range] -
                 np.dot(L[star_range, loop_range], L_conv))
    else:
        star_range = slice(None)
        L_red = L

    # find eigenvalues, and star part of eigenvectors, for LS combined modes
    w, v_s = la.eig(S[star_range, star_range], -L_red)

    if has_nullspace:
        v_l = -np.dot(L_conv, v_s)
        vr = np.vstack((v_l, v_s))
    else:
        vr = v_s

    w_freq = np.sqrt(w)
    # make sure real part is negative
    w_freq = np.where(w_freq.real > 0, -w_freq, w_freq)

    w_selected = np.ma.masked_array(w_freq, abs(w_freq.real) > abs(w_freq.imag))
    which_modes = np.argsort(abs(w_selected.imag))[modes]

    return w_freq[which_modes], vr[:, which_modes]


def eig_newton(func, lambda_0, x_0, lambda_tol=1e-8, max_iter=20,
               func_gives_der=False, G=None, args=[],
               weight='rayleigh symmetric', y_0=None):
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

        'rayleigh asymmetric' : Rayleigh iteration for general matrices

    y_0 : ndarray, optional
        For 'rayleigh asymmetric weighting', this is required as the initial
        guess for the left eigenvector

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

    if weight.lower == 'rayleigh asymmetric':
        if y_0 is None:
            raise ValueError("Parameter y_0 must be supplied for asymmetric "
                             "case")
        y_s = y_0

    logging.info("Searching for zeros with eig_newton")

    converged = False

    if not func_gives_der:
        # evaluate at an arbitrary nearby starting point to allow finite
        # differences to be taken
        lambda_sm = lambda_0*(1+10j*lambda_tol)
        T_sm = func(lambda_sm, *args)

    for iter_count in range(max_iter):
        if func_gives_der:
            T_s, T_ds = func(lambda_s, *args)
        else:
            T_s = func(lambda_s, *args)
            T_ds = (T_s - T_sm)/(lambda_s - lambda_sm)

        T_s_lu = la.lu_factor(T_s)
        u = la.lu_solve(T_s_lu, np.dot(T_ds, x_s))

        # if known_vects is supplied, we should take this into account when
        # finding v
        if weight.lower() == 'max element':
            v_s = np.zeros_like(x_s)
            v_s[np.argmax(abs(x_s))] = 1.0
        elif weight.lower() == 'rayleigh':
            v_s = np.dot(T_s.T, x_s.conj())
        elif weight.lower() == 'rayleigh symmetric':
            v_s = np.dot(T_s.T, x_s)
        elif weight.lower == 'rayleigh asymmetric':
            y_s = la.lu_solve(T_s_lu, np.dot(T_ds, y_s), trans=1)
            y_s /= np.sqrt(np.sum(abs(y_s)**2))
            v_s = np.dot(T_s.T, y_s)
        else:
            raise ValueError("Unknown weighting method %s" % weight)

        delta_lambda_abs = np.dot(v_s, x_s)/(np.dot(v_s, u))

        delta_lambda = abs(delta_lambda_abs/lambda_s)
        converged = delta_lambda < lambda_tol
        if converged:
            break

        lambda_s1 = lambda_s - delta_lambda_abs
        x_s1 = u/np.sqrt(np.sum(np.abs(u)**2))

        # update variables for next iteration
        if not func_gives_der:
            lambda_sm = lambda_s
            T_sm = T_s

        lambda_s = lambda_s1
        x_s = x_s1
        logging.debug("%+.4e %+.4ej" % (lambda_s.real, lambda_s.imag))

    if not converged:
        raise ValueError("maximum iterations reached, no convergence")

    res = {'eigval': lambda_s, 'eigvec': x_s, 'iter_count': iter_count+1,
           'delta_lambda': delta_lambda}

    if weight.lower == 'rayleigh asymmetric':
        res['eigvec_left'] = y_s

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

    for iter_count in range(max_iter):
        if G is not None:
            u = la.solve(Z-lambda_s*G, -G.dot(x_s))
        else:
            raise NotImplementedError
            # this should have identity matrix?
            u = la.solve(Z, x_s)

        if weight.lower() == 'max element':
            v_s = np.zeros_like(x_s)
            v_s[np.argmax(abs(x_s))] = 1.0
        elif weight.lower() == 'rayleigh':
            v_s = np.dot(np.array(Z-lambda_s*G).T, x_s.conj())
        elif weight.lower() == 'rayleigh symmetric':
            v_s = np.dot(np.array(Z-lambda_s*G).T, x_s)

        lambda_s1 = lambda_s - np.dot(v_s, x_s)/(np.dot(v_s, u))

        if G is None:
            x_s1 = u/np.sqrt(np.sum(np.abs(u)**2))
        else:
            # this assumes the rayleigh complex-symmetric normalisation
            x_s1 = u/np.sqrt(np.sum(u.dot(G.dot(u))))

        #x_s1 = u/np.sqrt(np.sum(u**2))

        delta_lambda = abs((lambda_s1 - lambda_s)/lambda_s)
        converged = delta_lambda < lambda_tol

        lambda_s = lambda_s1
        x_s = x_s1
        #print x_s
        #print lambda_s

        if converged:
            break

    if not converged:
        raise ValueError("maximum iterations reached, no convergence")

    res = {'eigval': lambda_s, 'eigvec': x_s, 'iter_count': iter_count+1,
           'delta_lambda': delta_lambda}

    return res


def eig_newton_bordered(A, w_0, vr_0, vl_0=None, w_tol=1e-8,
                        max_iter=20, B=None):
    """Solve a linear (generalised) eigenvalue problem by Newton iteration

    A.vr = w B.vr

    and optionally also

    vl.A = w vl.B

    Parameters
    ----------
    A : ndarray
        The matrix
    w_0 : complex
        The starting guess for the eigenvalue
    vr_0 : ndarray
        The starting guess for the right eigenvector
    vl_0 : ndarray, optional
        The starting guess for the left eigenvector. If not supplied, vr_0
        will be used, which is only accurate when A = A^T and B = B^T.
    w_tol : float, optional
        The relative tolerance in the eigenvalue for convergence
    max_iter : int, optional
        The maximum number of iterations to perform
    B : ndarray, optional
        The RHS matrix for the generalised problem. If omitted, the identity
        matrix will be used

    Returns
    -------
    res : dictionary
        A dictionary containing the following members:

        'w' : the eigenvalue
        'vr' : the right eigenvector
        'vl' : the left eigenvector
        'iter_count' : the number of iterations performed
        'delta_w' : the change in the eigenvalue on the final iteration

    See:
    1.  P. Lancaster, Lambda Matrices and Vibrating Systems.
        Oxford: Pergamon, 1966.

    2.  A. Ruhe, “Algorithms for the Nonlinear Eigenvalue Problem,”
        SIAM J. Numer. Anal., vol. 10, no. 4, pp. 674–689, Sep. 1973.

    3.  A. L. Andrew, E. K. Chu, and P. Lancaster, “On the numerical solution
        of nonlinear eigenvalue problems,” Computing, vol. 55, no. 2,
        pp. 91–111, Jun. 1995.
    """

    N = A.shape[0]

    if B is None:
        B = np.eye(N)
    elif hasattr(B, 'toarray'):
        # handle the sparse case
        B = B.toarray()
    else:
        B = np.asarray(B)

    vr_s = vr_0/np.sqrt(np.sum(vr_0.dot(B.dot(vr_0))))

    # If left eigenvalue is not passed, assume complex-symmetric matrix
    if vl_0 is None:
        vl_0 = vr_0
        vl_s = vr_s
        symmetric = True
    else:
        vl_s = vl_0/np.sum(vl_0.dot(B.dot(vr_0)))
        symmetric = False

    vr_0 /= np.sqrt(np.sum(np.abs(vr_0)**2))
    vl_0 /= np.sqrt(np.sum(np.abs(vl_0)**2))

    w_s = w_0
    converged = False

    augmented = np.empty((N+1, N+1), dtype=np.complex128)
    augmented[N, N] = 0.0
    augmented[N, :N] = vr_0.conjugate()  # vector c in Andrew notation
    augmented[:N, N] = vl_0.conjugate()  # vector b in Andrew notation

    rhs = np.zeros(N+1, A.dtype)
    rhs[-1] = 1

    for iter_count in range(max_iter):
        # Fill the augmented matrix with the impedance, and the previous
        # estimate of the eigenvector
        augmented[:N, :N] = A-w_s*B

        aug_lu = la.lu_factor(augmented)

        sg = la.lu_solve(aug_lu, rhs)
        vr_s1 = sg[:N]
        
        # the improved eigenvector estimate scaled appropriately
        vr_s1 /= np.sqrt(np.sum(vr_s1.dot(B.dot(vr_s1))))

        if symmetric:
            vl_s1 = vr_s1
        else:
            sg2 = la.lu_solve(aug_lu, rhs, trans=1)
            vl_s1 = sg2[:N]
            vl_s1 /= np.sum(vl_s1.dot(B.dot(vr_s1)))

        # at this stage vr_s.B.vl_s = 1
        w_s1 = vl_s.dot(A.dot(vr_s))

        delta_w = abs((w_s1 - w_s)/w_s)
        converged = delta_w < w_tol

        # update values for next iteration
        w_s = w_s1
        vr_s = vr_s1
        vl_s = vl_s1

        if converged:
            break

    if not converged:
        raise ValueError("maximum iterations reached, no convergence")

    return {'w': w_s, 'vr': vr_s, 'iter_count': iter_count+1,
            'delta_w': delta_w, 'vl': vl_s}


def eig_newton_bordered_nonlinear(func, lambda_0, x_0, lambda_tol=1e-8,
                                  max_iter=20, func_gives_der=False, args=[],
                                  G=None, weight=None):
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

    N = len(x_0)
    augmented = np.empty((N+1, N+1), dtype=np.complex128)
    augmented[N, N] = 0.0

    if G is None:
        G = np.eye(N)

    rhs = np.zeros(N+1, np.complex128)
    rhs[-1] = 1

    #x_s = x_0
    x_s = x_0/np.sqrt(np.sum(x_0.dot(G.dot(x_0))))
    lambda_s = lambda_0

    converged = False

    if not func_gives_der:
        # evaluate at an arbitrary nearby starting point to allow finite
        # differences to be taken
        lambda_sm = lambda_0*(1+10j*lambda_tol)
        #lambda_sm = lambda_0*lambda_tol
        T_sm = func(lambda_sm, *args)

    for iter_count in range(max_iter):
        if func_gives_der:
            T_s, T_ds = func(lambda_s, *args)
        else:
            T_s = func(lambda_s, *args)
            T_ds = (T_s - T_sm)/(lambda_s - lambda_sm)

        augmented[:N, :N] = T_s
        augmented[N, :N] = x_s
        augmented[:N, N] = x_s

        sg = la.solve(augmented, rhs)
        u = sg[:N]

#        u = la.solve(T_s, np.dot(T_ds, x_s))
#
#        # if known_vects is supplied, we should take this into account when
#        # finding v
#        if weight.lower() == 'max element':
#            v_s = np.zeros_like(x_s)
#            v_s[np.argmax(abs(x_s))] = 1.0
#        elif weight.lower() == 'rayleigh':
#            v_s = np.dot(T_s.T, x_s.conj())
#        elif weight.lower() == 'rayleigh symmetric':
#            v_s = np.dot(T_s.T, x_s)

        delta_lambda_abs = x_s.dot(T_s.dot(x_s))/x_s.dot(T_ds.dot(x_s))
        #print delta_lambda_abs

        delta_lambda = abs(delta_lambda_abs/lambda_s)
        #print delta_lambda
        converged = delta_lambda < lambda_tol
        if converged:
            break

        lambda_s1 = lambda_s - delta_lambda_abs
        #x_s1 = u/np.sqrt(np.sum(np.abs(u)**2))
        x_s1 = u/np.sqrt(np.sum(u.dot(G.dot(u))))

        # update variables for next iteration
        if not func_gives_der:
            lambda_sm = lambda_s
            T_sm = T_s

        lambda_s = lambda_s1
        x_s = x_s1
        #print x_s
        #print lambda_s

    if not converged:
        raise ValueError("maximum iterations reached, no convergence")

    res = {'eigval': lambda_s, 'eigvec': x_s, 'iter_count': iter_count+1,
           'delta_lambda': delta_lambda}

    return res


def project_modes(mode_j, E):
    """Take the projection of some field onto mode currents. Mostly useful
    for degenerate modes, in order to make the polarisation of a particular
    mode deterministic

    Parameters
    ----------
    mode_j : ndarray (n_basis, n_modes)
        The modal currents
    E : ndarray (n_basis)
        The solution on which to project

    Returns
    -------
    projected : ndarray(n_basis)
        The projected solution
    """
    projected = mode_j.dot(mode_j.T.dot(E))
    projected /= np.sqrt(np.sum(projected**2))
    return projected
