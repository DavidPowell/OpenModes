# -*- coding: utf-8 -*-
"""
Integration routines and quadrature rules over triangles

Created on Fri Apr 27 09:57:42 2012

@author: dap124
"""

import numpy as np
import scipy.special
import scipy.linalg as la

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
    weights = np.asfortranarray((weights[:, None]*0.5/sum(weights)).T) # scale the weights to 0.5

    return xi_eta_eval, weights


#TODO: confirm copyright, rewrite or get rid of?
#TODO: check that weighting is correct to 1/2
def triangle_quadrature(n):
    """Degree n quadrature points and weights on a triangle (0,0)-(1,0)-(0,1)"""
  
    x00,w00 = scipy.special.orthogonal.p_roots(n)
    x01,w01 = scipy.special.orthogonal.j_roots(n,1,0)
    x00s = (x00+1)/2
    x01s = (x01+1)/2
    w = np.outer(w01, w00).reshape(-1,1) / 8 # a factor of 2 for the legendres and 4 for the jacobi10
    x = np.outer(x01s, np.ones(x00s.shape)).reshape(-1,1)
    y = np.outer(1-x01s, x00s).reshape(-1,1)
    return np.hstack((x, y)), w

def cartesian_to_barycentric(r, nodes):
    """Convert cartesian coordinates to barycentric (area coordinates) in a triangle
    
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

