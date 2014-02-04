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


import os.path as osp

import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la

import openmodes
import openmodes.basis
from openmodes.constants import c

def horseshoe_modes():
    sim = openmodes.Simulation(name='horseshoe_modes', basis_class=openmodes.basis.LoopStarBasis, log_stderr=True)
    shoe = sim.load_mesh(osp.join('..', 'examples', 'geometry', 'horseshoe_rect.geo'), mesh_tol=1e-3)
    
    sim.place_part(shoe)
    
    s_start = 2j*np.pi*10e9
    
    mode_s, mode_j = sim.part_singularities(s_start, 3)
    
    sim.plot_solution([mode_j[0][:, 0]], 'mayavi', compress_scalars=1)
    sim.plot_solution([mode_j[0][:, 1]], 'mayavi', compress_scalars=1)
    sim.plot_solution([mode_j[0][:, 2]], 'mayavi', compress_scalars=1)

def horseshoe_extinction():
    sim = openmodes.Simulation(name='horseshoe_extinction', log_stderr=True,
                               basis_class=openmodes.basis.LoopStarBasis)

    shoe = sim.load_mesh(osp.join('..', 'examples', 'geometry', 'horseshoe_rect.geo'),
                         mesh_tol=2e-3)
    
    sim.place_part(shoe)
    
    num_freqs = 101
    freqs = np.linspace(1e8, 20e9, num_freqs)
    
    extinction = np.empty(num_freqs, np.complex128)
    
    e_inc = np.array([1, 0, 0], dtype=np.complex128)
    k_hat = np.array([0, 0, 1], dtype=np.complex128)
    
    
    for freq_count, s in sim.iter_freqs(freqs):
        Z = sim.impedance(s)[0][0]
        V = sim.source_plane_wave(e_inc, s/c*k_hat)[0]
        
        extinction[freq_count] = np.vdot(V, la.solve(Z[:], V))
        
    plt.figure()
    plt.plot(freqs*1e-9, extinction.real)
    plt.plot(freqs*1e-9, extinction.imag)
    plt.xlabel('f (GHz)')
    plt.show()

def horseshoe_extinction_modes():
    sim = openmodes.Simulation(name='horseshoe_extinction_modes', 
                               basis_class=openmodes.basis.LoopStarBasis,
                               log_stderr=True)
    shoe = sim.load_mesh(osp.join('..', 'examples', 'geometry', 'horseshoe_rect.geo'),
                         mesh_tol=3e-3)
    
    sim.place_part(shoe)
    
    s_start = 2j*np.pi*10e9
    
    num_modes = 3
    mode_s, mode_j = sim.part_singularities(s_start, num_modes)
    
    models = sim.construct_models(mode_s, mode_j)[0]
    
    num_freqs = 101
    freqs = np.linspace(1e8, 20e9, num_freqs)
    
    extinction = np.empty(num_freqs, np.complex128)
    extinction_modes = np.empty((num_freqs, num_modes), np.complex128)
    
    e_inc = np.array([1, 0, 0], dtype=np.complex128)
    k_hat = np.array([0, 0, 1], dtype=np.complex128)
    
    
    for freq_count, s in sim.iter_freqs(freqs):
        Z = sim.impedance(s)[0][0]
        V = sim.source_plane_wave(e_inc, s/c*k_hat)[0]
        
        extinction[freq_count] = np.vdot(V, la.solve(Z[:], V))
        mode_currents = [model.solve(s, V) for model in models]
        extinction_modes[freq_count] = [np.vdot(V, mode_current) for mode_current in mode_currents]
        
    plt.figure()
    plt.plot(freqs*1e-9, extinction.real)
    plt.plot(freqs*1e-9, np.sum(extinction_modes.real, axis=1), '--')
    plt.xlabel('f (GHz)')
    plt.show()



horseshoe_extinction_modes()
