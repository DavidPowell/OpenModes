# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 14:39:06 2013

@author: dap124
"""

from openmodes.basis import LoopStarBasis
import scipy.linalg as la
import numpy as np

def linearised_eig(L, S, num_modes, basis):
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
    w, v_s = la.eig(S[star_range, star_range], L_red)
    
    if basis.num_loops == 0:
        vr = v_s
    else:
        v_l = -np.dot(L_conv, v_s)
        vr = np.vstack((v_l, v_s))
    
    w_freq = np.sqrt(w)/2/np.pi
    w_selected = np.ma.masked_array(w_freq, w_freq.real < w_freq.imag)
    which_modes = np.argsort(w_selected.real)[:num_modes]
    
    return np.sqrt(w_freq[which_modes]), vr[:, which_modes]
