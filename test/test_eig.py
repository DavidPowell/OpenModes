# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 16:18:59 2014

@author: dap124
"""

import os.path as osp
import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt

import openmodes
from openmodes.basis import LoopStarBasis
from openmodes.eig import eig_newton_linear
from openmodes.constants import c

def test1():
    ring1 = openmodes.load_mesh(
                        osp.join(openmodes.geometry_dir, "SRR.geo"),
                        mesh_tol=1e-3)
    
    basis_class=LoopStarBasis
    #basis_class=DivRwgBasis
    
    sim = openmodes.Simulation(basis_class=basis_class, name="test_eig")
    sim.logger.propagate = True
    sim.place_part(ring1, location=[0e-3, 0, 0])
    
    start_freq = 1e10
    start_s = 2j*np.pi*start_freq
    
    num_modes = 4
    
    s_sem, j_sem = sim.part_singularities(start_s, num_modes, use_gram=True)
    s_sem = s_sem[0]
    j_sem = j_sem[0]
    
    impedance = sim.impedance(start_s)
    Z = impedance[0][0][:]
    z_sem = np.diag(j_sem.T.dot(Z.dot(j_sem)))
    #z_modes, j_modes = impedance.eigenmodes(num_modes, use_gram=False)
    
    
    G = impedance[0][0].basis_o.gram_matrix
    
    #print z_sem
    #print la.eigvals(Z, G)
    
    
    for mode in xrange(num_modes):
        print z_sem[mode]
        #res = eig_newton_linear(Z, z_sem[mode], j_sem[:, mode], G=None, weight='max element')
        res = eig_newton_linear(Z, z_sem[mode], j_sem[:, mode], G=G, weight='rayleigh symmetric')
        print res['eigval'], res['iter_count']
    
    
def compare_fit_impedance_scale():
    """"Compare fitting if the scalar impedance is scaled by 1/s.
    """
    
    srr = openmodes.load_mesh(osp.join(openmodes.geometry_dir, "SRR.geo"), 
                              mesh_tol=0.3e-3)
    
    basis_class = LoopStarBasis
    #basis_class = DivRwgBasis
    
    sim = openmodes.Simulation(basis_class=basis_class)
    sim.logger.propagate = True
    srr1 = sim.place_part(srr)
    #srr2 = sim.place_part(srr, location=[0e-3, 0e-3, 2e-3])
    #srr2.rotate([0, 0, 1], 180)
    
    num_freqs = 100
    freqs = np.linspace(-20e9, 20e9, num_freqs)#+1e8j
    e_inc = np.array([1, 0, 0], dtype=np.complex128)
    k_hat = np.array([0, 1, 0], dtype=np.complex128)
    
    num_modes = 2
    
    ext = np.empty(num_freqs, np.complex128)
    ext_modes = np.empty(num_freqs, np.complex128)
    ext_sem = np.empty((num_freqs, num_modes), np.complex128)
    ext_eem = np.empty((num_freqs, num_modes), np.complex128)
    
    if basis_class == LoopStarBasis:
        s_start = 2j*np.pi*10e9
        mode_s, mode_j = sim.part_singularities(s_start, num_modes, 
                                                use_gram=True)
        #print [s/(2*np.pi) for s in mode_s]
        models = sim.construct_models(mode_s, mode_j)[0]    
        #print [model.coefficients for model in models]
    
    z_eem = np.empty((num_freqs, num_modes), np.complex128)
    z_modes = np.empty((num_freqs, num_modes), np.complex128)
    z_sem = np.empty((num_freqs, num_modes), np.complex128)
    
    for freq_count, freq in enumerate(freqs):
        s = 2j*np.pi*freq
        jk_0 = s/c
    
        Z = sim.impedance(s)
        V = sim.source_plane_wave(e_inc, jk_0*k_hat)
        #Z, V = Z.combine_parts(V)
        #Z = Z[0][0]
        #V = V[0]
        #if freq_count == 0 and sim.basis_class==LoopStarBasis:
        #    print Z.basis_o.num_loops, Z.basis_s.num_loops
            
        ext[freq_count] = np.vdot(V[0], Z[0][0].solve(V[0]))
        
        eem_z, eem_j = Z.eigenmodes(num_modes, use_gram=True, start_j=mode_j)
    
        z_eem[freq_count] = eem_z[0]#*s
    
        Z_modes = Z.impedance_modes(num_modes, eem_j)
        V_modes = Z.source_modes(V, num_modes, eem_j)
        
        z_modes[freq_count] = Z_modes[:].diagonal()        
        
        ext_modes[freq_count] = np.vdot(V_modes, Z_modes.solve(V_modes))
    
        
        for mode in xrange(num_modes):
            ext_eem[freq_count, mode] = np.vdot(V[0], eem_j[0][:, mode])*np.dot(V[0], eem_j[0][:, mode])/eem_z[0][mode]
            
            if basis_class == LoopStarBasis:
                ext_sem[freq_count, mode] = np.vdot(V[0], models[mode].solve(s, V[0]))
                z_sem[freq_count, mode] = models[mode].scalar_impedance(s)#*s
    
    #plt.figure(figsize=(8, 5))
    #plt.subplot(121)
    #plt.plot(freqs*1e-9, ext.real)
    ##plt.plot(freqs*1e-9, ext_modes.real)
    #plt.plot(freqs*1e-9, np.sum(ext_sem.real, axis=1))
    #plt.plot(freqs*1e-9, np.sum(ext_eem.real, axis=1), 'x')
    #plt.subplot(122)
    #plt.plot(freqs*1e-9, ext.imag)
    ##plt.plot(freqs*1e-9, ext_modes.imag)
    #plt.plot(freqs*1e-9, np.sum(ext_sem.imag, axis=1))
    #plt.plot(freqs*1e-9, np.sum(ext_eem.imag, axis=1), 'x')
    #plt.show()
    
    #    z_eem = 1.0/z_eem
    #    z_sem = 1.0/z_sem    
    
    plt.figure(figsize=(10, 5))
    plt.subplot(121)
    plt.plot(freqs*1e-9, z_eem.real, 'x')
    #plt.plot(freqs*1e-9, z_modes.real, '--')
    plt.plot(freqs*1e-9, z_sem.real, '--')
    plt.subplot(122)
    #plt.semilogy(freqs*1e-9, abs(z_eem.imag), 'x')
    #plt.semilogy(freqs*1e-9, abs(z_modes.imag), '--')
    #plt.semilogy(freqs*1e-9, abs(z_sem.imag), '--')
    plt.plot(freqs*1e-9, z_eem.imag, 'x')
    plt.plot(freqs*1e-9, z_sem.imag, '--')
    plt.ylim(-200, 200)
    plt.show()
    
compare_fit_impedance_scale()    