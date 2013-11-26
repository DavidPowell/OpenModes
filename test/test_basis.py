# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 13:02:06 2013

@author: dap124
"""

import numpy as np
import matplotlib.pyplot as plt
import os.path as osp

from openmodes.basis import DivRwgBasis, LoopStarBasis
from openmodes.integration import get_dunavant_rule
from openmodes import load_mesh
from openmodes.visualise import write_vtk

def test_interpolate_rwg():

    mesh_tol = 0.5e-3
    srr = load_mesh(osp.join("..", "examples", "geometry", "SRR.geo"), mesh_tol)
    
    basis = DivRwgBasis(srr)
    
    rwg_function = np.zeros(len(basis), np.float64)
    rwg_function[20] = 1
    
    xi_eta, weights = get_dunavant_rule(10)
    
    
    r, res = basis.interpolate_function(rwg_function , xi_eta)
    
    plt.figure(figsize=(6, 6))
    plt.quiver(r[:, 0], r[:, 1], res[:, 0], res[:, 1], scale=0.03)
    #plt.plot(r[:, 0], r[:, 1], 'x')
    plt.show()

def test_interpolate_loop_star():
        
    mesh_tol = 4e-3
    mesh = load_mesh(osp.join("..", "examples", "geometry", "square_plate.geo"), mesh_tol)
    
    #basis = DivRwgBasis(mesh)
    basis = LoopStarBasis(mesh)
    
    which_basis = 22 #basis.num_loops+2
    
    ls_function = np.zeros(len(basis), np.float64)
    ls_function[which_basis] = 1
    
    ls_function[4] = 1
    
    xi_eta, weights = get_dunavant_rule(10)
    
    
    r, res = basis.interpolate_function(ls_function, xi_eta)
    
    the_basis = basis[which_basis]
    
    plus_nodes = mesh.nodes[basis.mesh.polygons[the_basis.tri_p, the_basis.node_p]]
    minus_nodes = mesh.nodes[basis.mesh.polygons[the_basis.tri_m, the_basis.node_m]]
    
    plt.figure(figsize=(6, 6))
    #plt.quiver(r[:, 0], r[:, 1], res[:, 0], res[:, 1], scale=0.05, pivot='middle')
    plt.quiver(r[:, 0], r[:, 1], res[:, 0], res[:, 1], pivot='middle')
    plt.plot(plus_nodes[:, 0], plus_nodes[:, 1], 'x')
    plt.plot(minus_nodes[:, 0], minus_nodes[:, 1], '+')
    #plt.plot(r[:, 0], r[:, 1], 'x')
    plt.show()

def test_irregular_array():

    from core_cython import IrregularIntArray
    
    a = IrregularIntArray([[1, 2, 3], [4, 5], [6, 7, 8, 9]])
    print a[0, 1]
    #try:
    print a[4, 3]
#except Exception as e:
#    print e
#prin


def loop_star_to_vtk():
        
    mesh_tol = 1e-3
    mesh = load_mesh(osp.join("..", "examples", "geometry", "SRR.geo"), mesh_tol)
    
    #basis = DivRwgBasis(mesh)
    basis = LoopStarBasis(mesh)
    print len(basis)    
    
    for basis_count in xrange(len(basis)):
    
        #which_basis = 22 #basis.num_loops+2
        
        I = np.zeros(len(basis), np.float64)
        I[basis_count] = 1
        
        #ls_function[4] = 1
        
        xi_eta, weights = get_dunavant_rule(10)
        
        
        #r, res = basis.interpolate_function(ls_function, xi_eta)

        #xi_eta = xi_eta, 

        face_centre, face_current, face_charge = basis.interpolate_function(I, return_scalar=True)
            
        write_vtk(mesh, mesh.nodes, osp.join("output", "srr-basis-%02d.vtk" % basis_count), 
                  vector_function = face_current, scalar_function=face_charge
                  )        
        
#        the_basis = basis[which_basis]
#        
#        plus_nodes = mesh.nodes[basis.mesh.polygons[the_basis.tri_p, the_basis.node_p]]
#        minus_nodes = mesh.nodes[basis.mesh.polygons[the_basis.tri_m, the_basis.node_m]]
#        
#        plt.figure(figsize=(6, 6))
#        #plt.quiver(r[:, 0], r[:, 1], res[:, 0], res[:, 1], scale=0.05, pivot='middle')
#        plt.quiver(r[:, 0], r[:, 1], res[:, 0], res[:, 1], pivot='middle')
#        plt.plot(plus_nodes[:, 0], plus_nodes[:, 1], 'x')
#        plt.plot(minus_nodes[:, 0], minus_nodes[:, 1], '+')
#        #plt.plot(r[:, 0], r[:, 1], 'x')
#        plt.show()
#

#test_interpolate_rwg()
#test_interpolate_loop_star()
#loop_star_to_vtk()