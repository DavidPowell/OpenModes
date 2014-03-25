# -*- coding: utf-8 -*-
"""
Created on Mon Mar 17 17:37:23 2014

@author: dap124
"""

import os.path as osp

import numpy as np

import openmodes
import openmodes.basis
from openmodes.constants import c
    
    
mesh_tol = 0.5e-3;
period_x = 15e-3;
period_y = 15e-3;
    
sim = openmodes.Simulation(name='Montage', 
                           basis_class=openmodes.basis.LoopStarBasis,
                           log_display_level=20)

srr = sim.place_part(location = [2*period_x, 0, 0])

srr_inner_mesh = sim.load_mesh(osp.join(openmodes.geometry_dir, 'SRR.geo'),
                     mesh_tol=mesh_tol, parameters={'inner_radius' : 2.5e-3,
                                                    'outer_radius' : 4e-3})
srr_inner = sim.place_part(srr_inner_mesh, parent=srr)

srr_outer_mesh = sim.load_mesh(osp.join(openmodes.geometry_dir, 'SRR.geo'),
                     mesh_tol=mesh_tol, parameters={'inner_radius' : 4.5e-3,
                                                    'outer_radius' : 6e-3})
srr_outer = sim.place_part(srr_outer_mesh, parent=srr)
srr_outer.rotate([0, 0, 1], 180)

s = 2j*np.pi*1e9

#current = sim.empty_vector()

# for most of the parts, just calculate the lowest-order mode
#for part in (canonical, v_antenna, srr, horseshoe):
mode_s, current = sim.singularities(s, 1)
#mode_s, mode_j = sim.singularities(s, 1)
#current[srr] = mode_j[:, 0]

#Z = sim.impedance(s)
#k_hat = np.array([1, 0, 0])
#e_inc = np.array([0, 1, 0])
#jk = s/c
#V = sim.source_plane_wave(e_inc, jk*k_hat)
#current = Z.solve(V)

sim.plot_solution(current, 'mayavi') #, compress_scalars=1)
