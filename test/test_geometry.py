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

def geometry_extinction_modes(name, freqs, num_modes, mesh_tol, 
                              plot_only=False, parameters={}):
    """Load a geometry file, calculate its modes by searching for
    singularities, and plot them in 3D. Then use the modes to calculate
    extinction, and compare with exact calculation
    """
    
    sim = openmodes.Simulation(name=name, 
                               basis_class=openmodes.basis.LoopStarBasis,
                               log_display_level=20)
    mesh = sim.load_mesh(osp.join(openmodes.geometry_dir, name+'.geo'),
                         mesh_tol=mesh_tol, parameters=parameters)
    sim.place_part(mesh)
    
    s_start = 2j*np.pi*0.5*(freqs[0]+freqs[-1])
    
    mode_s, mode_j = sim.part_singularities(s_start, num_modes)

    for mode in xrange(num_modes):
        sim.plot_solution([mode_j[0][:, mode]], 'mayavi', compress_scalars=1)
   
    if plot_only:
        return
   
    models = sim.construct_models(mode_s, mode_j)[0]
    
    num_freqs = len(freqs)
    
    extinction = np.empty(num_freqs, np.complex128)
    extinction_sem = np.empty((num_freqs, num_modes), np.complex128)
    extinction_eem = np.empty((num_freqs, num_modes), np.complex128)
    
    e_inc = np.array([1, 0, 0], dtype=np.complex128)
    k_hat = np.array([0, 0, 1], dtype=np.complex128)

    z_sem = np.empty((num_freqs, num_modes), np.complex128)
    z_eem = np.empty((num_freqs, num_modes), np.complex128)
    z_eem_direct = np.empty((num_freqs, num_modes), np.complex128)
    
    for freq_count, s in sim.iter_freqs(freqs):
        Z = sim.impedance(s)[0][0]
        V = sim.source_plane_wave(e_inc, s/c*k_hat)[0]
        
        extinction[freq_count] = np.vdot(V, la.solve(Z[:], V))
        
        z_sem[freq_count] = [model.scalar_impedance(s) for model in models]
        extinction_sem[freq_count] = [np.vdot(V, model.solve(s, V)) for model in models]

#        z_eem_direct[freq_count], _ = Z.eigenmodes(num_modes, use_gram=False)
#        z_eem[freq_count], j_eem = Z.eigenmodes(start_j = mode_j[0], use_gram=True)
#        extinction_eem[freq_count] = [np.vdot(V, j_eem[:, mode])*np.dot(V, j_eem[:, mode])/z_eem[freq_count, mode] for mode in xrange(num_modes)]
        
        
    plt.figure(figsize=(10,5))
    plt.subplot(121)
    plt.plot(freqs*1e-9, extinction.real)
    plt.plot(freqs*1e-9, np.sum(extinction_sem.real, axis=1), '--')
    #plt.plot(freqs*1e-9, np.sum(extinction_eem.real, axis=1), '-.')
    plt.xlabel('f (GHz)')
    plt.subplot(122)
    plt.plot(freqs*1e-9, extinction.imag)
    plt.plot(freqs*1e-9, np.sum(extinction_sem.imag, axis=1), '--')
    #plt.plot(freqs*1e-9, np.sum(extinction_eem.imag, axis=1), '-.')
    plt.suptitle("Extinction")
    plt.show()

#    plt.figure(figsize=(10,5))
#    plt.subplot(121)
#    plt.plot(freqs*1e-9, z_eem_direct.real)
#    #plt.ylim(0, 80)
#    plt.xlabel('f (GHz)')
#    plt.subplot(122)
#    plt.plot(freqs*1e-9, z_eem_direct.imag)
#    plt.plot([freqs[0]*1e-9, freqs[-1]*1e-9], [0, 0], 'k')
#    plt.suptitle("EEM impedance without Gram matrix")
#    plt.show()
#
#    plt.figure(figsize=(10,5))
#    plt.subplot(121)
#    plt.plot(freqs*1e-9, z_eem.real)
#    plt.plot(freqs*1e-9, z_sem.real, '--')
#    #plt.ylim(0, 80)
#    plt.xlabel('f (GHz)')
#    plt.subplot(122)
#    plt.plot(freqs*1e-9, z_eem.imag)
#    plt.plot(freqs*1e-9, z_sem.imag, '--')
#    plt.ylim(-100, 100)
##    plt.semilogy(freqs*1e-9, abs(z_eem.imag))
##    plt.semilogy(freqs*1e-9, abs(z_sem.imag), '--')
#    plt.suptitle("SEM and EEM impedance")
#    plt.show()
#
    y_sem = 1/z_sem
    #y_eem = 1/z_eem

    plt.figure(figsize=(10,5))
    plt.subplot(121)
    #plt.plot(freqs*1e-9, y_eem.real)
    plt.plot(freqs*1e-9, y_sem.real, '--')
    plt.xlabel('f (GHz)')
    plt.subplot(122)
    #plt.plot(freqs*1e-9, y_eem.imag)
    plt.plot(freqs*1e-9, y_sem.imag, '--')
    plt.xlabel('f (GHz)')
    plt.suptitle("SEM and EEM admittance")
    plt.show()


#geometry_extinction_modes('horseshoe_rect', np.linspace(1e8, 20e9, 101), 3, 1.5e-3)
#geometry_extinction_modes('sphere', np.linspace(0.2e7, 8e7, 101), 16, 0.2, plot_only=True)
#geometry_extinction_modes('canonical_spiral', np.linspace(1e8, 15e9, 101), 
#                          1, 1e-3, parameters={'arm_length' : 12e-3, 'inner_radius' : 2e-3})
geometry_extinction_modes('v_antenna', np.linspace(1e8, 15e9, 101), 
                          2, 0.7e-3)

