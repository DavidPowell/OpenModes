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

import logging
logging.getLogger().setLevel(logging.INFO)

mesh_tol = 0.3e-3
    
sim = openmodes.Simulation(name='Test DSRR', 
                           basis_class=openmodes.basis.LoopStarBasis)

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

modes = [0, 2] #3
num_modes = len(modes)
s_inner, current_inner = sim.singularities(start_s, modes, srr_inner)
s_outer, current_outer = sim.singularities(start_s, modes, srr_outer)

#s_full, current_full = sim.singularities(start_s, 3, srr)

parts_modes = [(srr_inner, s_inner, current_inner),
               (srr_outer, s_outer, current_outer)]

poly_order = 2
s_max = 2j*np.pi*5e9

model = ModelPolyInteraction(sim.operator, parts_modes, poly_order, s_max)

projection_sem = [(srr_inner, current_inner),
                  (srr_outer, current_outer)]

k_hat = np.array([1, 0, 0])
e_inc = np.array([0, 1, 0])

num_freqs = 200
freqs = np.linspace(3.2e9, 4.5e9, num_freqs)

extinction = np.empty(num_freqs, np.complex128)
extinction_red = np.empty(num_freqs, np.complex128)
extinction_sem = np.empty(num_freqs, np.complex128)

z_sem = np.empty((num_freqs, num_modes), np.complex128)
z_eem = np.empty((num_freqs, num_modes), np.complex128)

for freq_count, s in sim.iter_freqs(freqs):
    Z = sim.impedance(s)
    jk = s/c
    V = sim.source_plane_wave(e_inc, jk*k_hat)
    I = Z.solve(V)
    extinction[freq_count] = np.vdot(V, I)

    z_inner, modes_inner = Z.eigenmodes(srr_inner, start_j=current_inner)
    z_outer, modes_outer = Z.eigenmodes(srr_outer, start_j=current_outer)
    projection = [(srr_inner, modes_inner), (srr_outer, modes_outer)]
    
    #z_full, modes_full = Z.eigenmodes(srr, start_j=current_full)
    #projection = [(srr, current_full)]
    
    Z_red = Z.weight(projection)
    V_red = V.weight(projection)
    I_red = Z_red.solve(V_red)
    extinction_red[freq_count] = np.vdot(V_red, I_red)
    #extinction_red[freq_count] = np.vdot(V.project(projection), I.project(projection))
    
    V_sem = V.weight(projection_sem)
    extinction_sem[freq_count] = np.vdot(V_sem, model.solve(s, V_sem))
    
    z_eem[freq_count] = np.diag(Z_red[:])[:num_modes]
    z_sem[freq_count] = np.diag(model.models[srr_inner, srr_inner].block_impedance(s))[:num_modes]


plt.figure()
plt.plot(freqs*1e-9, extinction.real)
plt.plot(freqs*1e-9, extinction_red.real)
plt.plot(freqs*1e-9, extinction_sem.real)
plt.show()

plt.figure()
plt.plot(freqs*1e-9, extinction.imag)
plt.plot(freqs*1e-9, extinction_red.imag)
plt.plot(freqs*1e-9, extinction_sem.imag)
plt.show()

#plt.figure()
##plt.plot(freqs*1e-9, z_eem[:, 0].real)
##plt.plot(freqs*1e-9, z_sem[:, 0].real, '--')
#plt.plot(freqs*1e-9, z_eem.real)
#plt.plot(freqs*1e-9, z_sem.real, '--')
#plt.show()
#
#plt.figure()
##plt.plot(freqs*1e-9, z_eem[:, 0].imag)
##plt.plot(freqs*1e-9, z_sem[:, 0].imag, '--')
#plt.plot(freqs*1e-9, z_eem.imag)
#plt.plot(freqs*1e-9, z_sem.imag, '--')
#plt.show()
