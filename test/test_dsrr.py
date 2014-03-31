# -*- coding: utf-8 -*-
"""
Created on Mon Mar 17 17:37:23 2014

@author: dap124
"""

import os.path as osp

import numpy as np
import matplotlib.pyplot as plt

import openmodes
import openmodes.basis
from openmodes.constants import c
from openmodes.model import ModelPolyInteraction
    
mesh_tol = 1e-3
    
sim = openmodes.Simulation(name='Test DSRR', 
                           basis_class=openmodes.basis.LoopStarBasis,
                           log_display_level=20)

srr = sim.place_part()

srr_inner_mesh = sim.load_mesh(osp.join(openmodes.geometry_dir, 'SRR.geo'),
                     mesh_tol=mesh_tol, parameters={'inner_radius' : 2.5e-3,
                                                    'outer_radius' : 4e-3})
srr_inner = sim.place_part(srr_inner_mesh, parent=srr)

srr_outer_mesh = sim.load_mesh(osp.join(openmodes.geometry_dir, 'SRR.geo'),
                     mesh_tol=mesh_tol, parameters={'inner_radius' : 4.5e-3,
                                                    'outer_radius' : 6e-3})
srr_outer = sim.place_part(srr_outer_mesh, parent=srr)
srr_outer.rotate([0, 0, 1], 180)


start_s = 2j*np.pi*1e9

s_inner, current_inner = sim.singularities(start_s, 3, srr_inner)
s_outer, current_outer = sim.singularities(start_s, 3, srr_outer)

parts_modes = [(srr_inner, s_inner, current_inner),
               (srr_outer, s_outer, current_outer)]

model = ModelPolyInteraction(sim.operator, parts_modes, logger=sim.logger)

projection_sem = [(srr_inner, current_inner),
                  (srr_outer, current_outer)]


#sim.plot_solution(current, 'mayavi') #, compress_scalars=1)



#I = Z.solve(V)
#I[srr_outer]
#
#z_inner, modes_inner = Z.eigenmodes(srr_inner, 1)
#z_outer, modes_outer = Z.eigenmodes(srr_outer, 1)
#
#
#
#Z_red = Z.project_modes(projection)
#V_red = V.project_modes(projection)
#
#I_red = Z_red.solve(V_red)




k_hat = np.array([1, 0, 0])
e_inc = np.array([0, 1, 0])

num_freqs = 100
freqs = np.linspace(2e9, 10e9, num_freqs)

extinction = np.empty(num_freqs, np.complex128)
extinction_red = np.empty(num_freqs, np.complex128)
extinction_sem = np.empty(num_freqs, np.complex128)

for freq_count, s in sim.iter_freqs(freqs):
    Z = sim.impedance(s)
    jk = s/c
    V = sim.source_plane_wave(e_inc, jk*k_hat)
    extinction[freq_count] = np.vdot(V, Z.solve(V))

    z_inner, modes_inner = Z.eigenmodes(srr_inner, 3)
    z_outer, modes_outer = Z.eigenmodes(srr_outer, 3)
    projection = [(srr_inner, modes_inner), (srr_outer, modes_outer)]
    
    Z_red = Z.project_modes(projection)
    V_red = V.project_modes(projection)
    extinction_red[freq_count] = np.vdot(V_red, Z_red.solve(V_red))
    
    V_sem = V.project_modes(projection_sem)
    extinction_sem[freq_count] = np.vdot(V_sem, model.solve(s, V_sem))


plt.figure()
plt.plot(freqs*1e-9, extinction.real)
plt.plot(freqs*1e-9, extinction_red.real)
plt.plot(freqs*1e-9, extinction_sem.real)
plt.show()
    
