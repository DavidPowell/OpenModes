# -*- coding: utf-8 -*-
"""
Routines for solving nonlinear eigenvalue problems

[1] A. Ruhe, SIAM Journal on Numerical Analysis I, 674 (1973).

"""

from openmodes.basis import LoopStarBasis
import scipy.linalg as la
import numpy as np

def eig_linearised(L, S, num_modes, basis):
    """Solves a linearised approximation to the eigenvalue problem from
    the impedance calculated at some fixed frequency.
    
    Parameters
    ----------
    L, S : ndarray
        The two components of the impedance matrix. They *must* be
        calculated in the loop-star basis.
    num_modes : int
        The number of modes required.
    basis : LoopStarBasis
        Which object in the system to find modes for. If not specified, 
        then modes of the entire system will be found
        
    Returns
    -------
    s_mode : ndarray, complex
        The resonant frequencies of the modes (in Hz)
        The complex pole s corresponding to the mode's eigenfrequency
    j_mode : ndarray, complex
        Columns of this matrix contain the corresponding modal currents
    """
    
    if not isinstance(basis, LoopStarBasis):
        raise ValueError(
            "Loop-star basis functions required for linearised eigenvalues")
    
    star_range = slice(basis.num_loops, len(basis))

    if basis.num_loops == 0:
        L_red = L
    else:
        loop_range = slice(0, basis.num_loops)
        
        L_conv = la.solve(L[loop_range, loop_range], 
                          L[loop_range, star_range])
        L_red = L[star_range, star_range] - np.dot(L[star_range, loop_range], 
                                                    L_conv)

    # find eigenvalues, and star part of eigenvectors, for LS combined modes
    w, v_s = la.eig(S[star_range, star_range], -L_red)
    
    if basis.num_loops == 0:
        vr = v_s
    else:
        v_l = -np.dot(L_conv, v_s)
        vr = np.vstack((v_l, v_s))
    
#    import matplotlib.pyplot as plt
#    plt.loglog(w.real, w.imag, 'x')    
    
    w_freq = np.sqrt(w)#/2/np.pi

    #plt.loglog(w_freq.real, w_freq.imag, 'x')    

    #w_selected = np.ma.masked_array(w_freq, w_freq.real < w_freq.imag)
    w_selected = np.ma.masked_array(w_freq, abs(w_freq.real) > abs(w_freq.imag))
    which_modes = np.argsort(abs(w_selected.imag))[:num_modes]
    
    return w_freq[which_modes], vr[:, which_modes]


def eig_newton(func, lambda_0, x_0, lambda_tol = 1e-8, max_iter = 20, 
               func_gives_der = False, args = [], weight='rayleigh symmetric'):
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
        
        'eigval' : the eigenvalue
        
        'eigvect' : the eigenvector
        
        'iter_count' : the number of iterations performed
        
        'delta_lambda' : the change in the eigenvalue on the final iteration
    """

    x_s = x_0
    lambda_s = lambda_0

    converged = False   

    if not func_gives_der:
        # evaluate at an arbitrary nearby starting point to allow finite
        # differences to be taken
        lambda_sm = lambda_0*(1+2*lambda_tol) # a point for taking finite differences
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
        converged =  delta_lambda < lambda_tol

        # update variables for next iteration        
        if not func_gives_der:
            lambda_sm = lambda_s
            T_sm = T_s
            
        lambda_s = lambda_s1
        x_s = x_s1
        
        if converged: break
        
    if not converged:
        raise ValueError, "maximum iterations reached, no convergence"
     
    res = {'eigval' : lambda_s, 'eigvec' : x_s, 'iter_count' : iter_count,
           'delta_lambda' : delta_lambda}
     
    return res