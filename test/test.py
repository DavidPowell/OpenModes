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

#from openmodeimport gmsh
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la
import scipy.sparse.linalg as spla

import openmodes
from openmodes.visualise import plot_parts, write_vtk, plot_mayavi
from openmodes.constants import c
from openmodes.basis import DivRwgBasis, LoopStarBasis, get_basis_functions
from openmodes.eig import eig_linearised


def loop_star_linear_eigenmodes():
    """"Solve the linearised eigenvalue problem in a loop-star basis.
    """

    ring1, ring2 = openmodes.load_mesh(
                    osp.join("..", "examples", "geometry", "asymmetric_ring.geo"), mesh_tol=1e-3)

    #ring1 = openmodes.load_mesh(
    #                osp.join("geometry", "SRR.geo"), mesh_tol=0.4e-3)

    basis_class=LoopStarBasis
    #basis_class=DivRwgBasis

    sim = openmodes.Simulation(basis_class=basis_class)
    part1 = sim.place_part(ring1)
    part2 = sim.place_part(ring2)
   
    freq = 7e9
    s = 2j*np.pi*freq

    Z = sim.calculate_impedance(s)
    num_modes = 3

    _, vr0 = eig_linearised(Z[0][0], num_modes)
    _, vr1 = eig_linearised(Z[1][1], num_modes)
    I = [vr0[:, 0], vr1[:, 0]]

    #w, vr = sim.part_singularities(Z, n_modes)
    #I = [vr[0][:, 0], vr[1][:, 0]]

    sim.plot_solution(I, 'mayavi')

def loop_star_combined():
    """"Confirm that combining impedance matrices in a loop-star basis gives
    the same result as with RWG, where combining is straightforward.
    """

    ring1, ring2 = openmodes.load_mesh(
                    osp.join("..", "examples", "geometry", "asymmetric_ring.geo"), mesh_tol=0.5e-3)

    sim_rwg = openmodes.Simulation(basis_class=DivRwgBasis)
    sim_rwg.place_part(ring1)
    sim_rwg.place_part(ring2)

    sim_ls = openmodes.Simulation(basis_class=LoopStarBasis)
    sim_ls.place_part(ring1)
    sim_ls.place_part(ring2)

    num_freqs = 101
    freqs = np.linspace(4e9, 16e9, num_freqs)
    e_inc = np.array([1, 0, 0], dtype=np.complex128)
    k_hat = np.array([0, 1, 0], dtype=np.complex128)

    ext_rwg = np.empty(num_freqs, np.complex128)
    ext_ls = np.empty(num_freqs, np.complex128)

    for freq_count, freq in enumerate(freqs):
        s = 2j*np.pi*freq
        jk_0 = s/c

        Z = sim_rwg.calculate_impedance(s)
        V = sim_rwg.source_plane_wave(e_inc, jk_0*k_hat)
        Z, V = Z.combine_parts(V)
        ext_rwg[freq_count] = np.vdot(V, Z.solve(V))

        Z = sim_ls.calculate_impedance(s)
        V = sim_ls.source_plane_wave(e_inc, jk_0*k_hat)
        Z, V = Z.combine_parts(V)
        if freq_count == 0:
            print Z.basis_o.num_loops, Z.basis_s.num_loops
        ext_ls[freq_count] = np.vdot(V, Z.solve(V))

    plt.figure()
    plt.plot(freqs*1e-9, ext_rwg.real)
    plt.plot(freqs*1e-9, ext_rwg.imag, '--')
    plt.plot(freqs*1e-9, ext_ls.real)
    plt.plot(freqs*1e-9, ext_ls.imag, '--')
    plt.show()


def sem_eem_asrr():
    """"Calculate the singularity expansion of a combined system.
    """

    ring1, ring2 = openmodes.load_mesh(
                    osp.join("..", "examples", "geometry", "asymmetric_ring.geo"), mesh_tol=0.5e-3)

    sim = openmodes.Simulation(basis_class=LoopStarBasis)
    sim.place_part(ring1)
    sim.place_part(ring2)

    num_freqs = 201
    freqs = np.linspace(4e9, 16e9, num_freqs)
    e_inc = np.array([1, 0, 0], dtype=np.complex128)
    k_hat = np.array([0, 1, 0], dtype=np.complex128)

    num_modes = 2

    ext = np.empty(num_freqs, np.complex128)
    ext_sem = np.empty((num_freqs, num_modes), np.complex128)
    ext_eem = np.empty((num_freqs, num_modes), np.complex128)

    s_start = 2j*np.pi*10e9
    mode_s, mode_j = sim.system_singularities(s_start, num_modes)
    models = sim.construct_model_system(mode_s, mode_j)

    for freq_count, freq in enumerate(freqs):
        s = 2j*np.pi*freq
        jk_0 = s/c

        Z = sim.calculate_impedance(s)
        V = sim.source_plane_wave(e_inc, jk_0*k_hat)
        Z, V = Z.combine_parts(V)
        if freq_count == 0:
            print Z.num_loops_o, Z.num_loops_s
            
        ext[freq_count] = np.vdot(V, Z.solve(V))
        
        eem_z, eem_j = Z.eigenmodes(num_modes)
        
        for mode in xrange(num_modes):
            ext_sem[freq_count, mode] = np.vdot(V, models[mode].solve(s, V))
            ext_eem[freq_count, mode] = np.vdot(V, eem_j[:, mode])*np.dot(V, eem_j[:, mode])/eem_z[mode]

    plt.figure()
    plt.plot(freqs*1e-9, ext.real)
    plt.plot(freqs*1e-9, np.sum(ext_sem.real, axis=1))
    plt.plot(freqs*1e-9, np.sum(ext_eem.real, axis=1))
    #plt.plot(freqs*1e-9, ext_modes.real, '--')
    #plt.plot(freqs*1e-9, ext.imag, '--')
    plt.show()
    

def sem_eem_bcsrr():
    """"Calculate the singularity expansion of a combined system.
    """

    srr = openmodes.load_mesh(
                    osp.join("..", "examples", "geometry", "SRR.geo"), mesh_tol=0.7e-3)

    basis_class = LoopStarBasis
    #basis_class = DivRwgBasis

    sim = openmodes.Simulation(basis_class=basis_class)
    srr1 = sim.place_part(srr)
    #srr2 = sim.place_part(srr, location=[0e-3, 0e-3, 2e-3])
    #srr2.rotate([0, 0, 1], 180)

    num_freqs = 201
    freqs = np.linspace(0.5e9, 20e9, num_freqs)#+1e8j
    e_inc = np.array([1, 0, 0], dtype=np.complex128)
    k_hat = np.array([0, 1, 0], dtype=np.complex128)

    num_modes = 6

    ext = np.empty(num_freqs, np.complex128)
    ext_modes = np.empty(num_freqs, np.complex128)
    ext_sem = np.empty((num_freqs, num_modes), np.complex128)
    ext_eem = np.empty((num_freqs, num_modes), np.complex128)

    if basis_class == LoopStarBasis:
        s_start = 2j*np.pi*10e9
        mode_s, mode_j = sim.part_singularities(s_start, num_modes)
        print [s/(2*np.pi) for s in mode_s]
        models = sim.construct_models(mode_s, mode_j)[0]    
        print [model.coefficients for model in models]
    
    z_eem = np.empty((num_freqs, num_modes), np.complex128)
    z_modes = np.empty((num_freqs, num_modes), np.complex128)
    z_sem = np.empty((num_freqs, num_modes), np.complex128)

    for freq_count, freq in enumerate(freqs):
        s = 2j*np.pi*freq
        jk_0 = s/c

        Z = sim.calculate_impedance(s)
        V = sim.source_plane_wave(e_inc, jk_0*k_hat)
        #Z, V = Z.combine_parts(V)
        Z = Z[0][0]
        V = V[0]
        if freq_count == 0 and sim.basis_class==LoopStarBasis:
            print Z.basis_o.num_loops, Z.basis_s.num_loops
            
        ext[freq_count] = np.vdot(V, Z.solve(V))
        
        eem_z, eem_j = Z.eigenmodes(num_modes, True)

        z_eem[freq_count] = eem_z

        Z_modes = Z.impedance_modes(num_modes, eem_j)
        V_modes = Z.source_modes(V, num_modes, eem_j)
        
        z_modes[freq_count] = Z_modes[:].diagonal()        
        
        ext_modes[freq_count] = np.vdot(V_modes, Z_modes.solve(V_modes))

        
        for mode in xrange(num_modes):
            ext_eem[freq_count, mode] = np.vdot(V, eem_j[:, mode])*np.dot(V, eem_j[:, mode])/eem_z[mode]
            
            if basis_class == LoopStarBasis:
                ext_sem[freq_count, mode] = np.vdot(V, models[mode].solve(s, V))
                z_sem[freq_count, mode] = models[mode].scalar_impedance(s)

    plt.figure(figsize=(10, 5))
    plt.subplot(121)
    plt.plot(freqs*1e-9, ext.real)
    plt.plot(freqs*1e-9, ext_modes.real)
    #plt.plot(freqs*1e-9, np.sum(ext_sem.real, axis=1))
    plt.plot(freqs*1e-9, np.sum(ext_eem.real, axis=1))
    plt.subplot(122)
    plt.plot(freqs*1e-9, ext.imag)
    plt.plot(freqs*1e-9, ext_modes.imag)
    #plt.plot(freqs*1e-9, np.sum(ext_sem.imag, axis=1))
    plt.plot(freqs*1e-9, np.sum(ext_eem.imag, axis=1))
    plt.show()
    
#    z_eem = 1.0/z_eem
#    z_sem = 1.0/z_sem    
    
#    plt.figure(figsize=(10, 5))
#    plt.subplot(121)
#    plt.plot(freqs*1e-9, z_eem.real, 'x')
#    plt.plot(freqs*1e-9, z_modes.real, '--')
#    #plt.plot(freqs*1e-9, z_sem.real, '--')
#    plt.subplot(122)
#    plt.semilogy(freqs*1e-9, abs(z_eem.imag), 'x')
#    plt.semilogy(freqs*1e-9, abs(z_modes.imag), '--')
#   # plt.semilogy(freqs*1e-9, abs(z_sem.imag), '--')
#    #plt.plot(freqs*1e-9, z_eem.imag, 'x')
#    #plt.plot(freqs*1e-9, z_sem.imag, '--')
#    plt.show()


def srr_coupling():
    """
    Calculate coupling coefficients for a pair of SRRs
    """

    #filename = osp.join("geometry", "dipole.geo")
    #freqs = np.linspace(4e9, 15e9, 201)
    filename = osp.join("..", "examples", "geometry", "SRR.geo")
    freqs = np.linspace(2e9, 20e9, 201)

    mesh_tol = 2e-3
    
    sim = openmodes.Simulation()
    
    #meshed_name = gmsh.mesh_geometry(filename, mesh_tol)
    #(nodes, triangles), = gmsh.read_mesh(meshed_name)
    
    #srr = openmodes.LibraryPart(nodes, triangles)
    srr = openmodes.load_parts(filename, mesh_tol)
    
    srr1 = sim.place_part(srr)
    srr2 = sim.place_part(srr)    
    
    #srr2 = srr1.duplicate()
    #separation = [500e-3, 0e-3, 0]
    separation = [0e-3, 50e-3, 0]
    #separation = [0e-3, 150e-3, 0]
    #separation = [150e-3, 0e-3, 0]
    #separation = [15e-3, 15e-3, 0]
    #separation = [0, 10e-3, 0]
    srr2.translate(separation)
    #dipole2.rotate([0, 0, 1], 45)
    
    #sim.add_object(srr1)
    #sim.add_object(srr2)
    
    kappa_e_direct = np.empty(len(freqs), np.complex128)
    kappa_m_direct = np.empty(len(freqs), np.complex128)
    
    n_modes = 2 
    
    
    y_srr = np.empty((len(freqs), n_modes), np.complex128)
    
    for count, freq in enumerate(freqs):
        s = 2j*np.pi*freq
        #jk_0 = s/c
    
        L, S = sim.impedance_matrix(s)

        
        L_self = sim.sub_matrix(L, 0)
        S_self = sim.sub_matrix(S, 0)

        Z_self = s*L_self + S_self/s

        z_all, v_all = la.eig(Z_self)
        which_z = np.argsort(abs(z_all.imag))[:n_modes]
        z_direct = z_all[which_z]
        
        # There is some variation of dipole moment with frequency. However
        # for x separation of dipoles, this is automatically accounted for.
        # For y separation direct calculation of mutual L drops much faster
        # than dipole
        #if count == 0:
        v = v_all[:, which_z]
        v /= np.sqrt(np.sum(v**2, axis=0))

        y_srr[count] = 1.0/z_direct

        #w, v = mom.eig_linearised(L, S, n_modes, which_obj = 0)
        #w, vl, vr = sim.get_eigs(L[r1, r1], S[r1, r1], filter_freq, left=True)



        L_mutual = sim.sub_matrix(L, 0, 1)
        S_mutual = sim.sub_matrix(S, 0, 1)

        L_mut = np.dot(v[:, 0], np.dot(L_mutual, v[:, 0]))
        L_self = np.dot(v[:, 0], np.dot(L_self, v[:, 0]))

        #kappa_m_direct[count] = np.dot(v[:, 0], la.lu_solve(L_lu, np.dot(L_mutual, v[:, 0])))
        kappa_m_direct[count] = L_mut/L_self #1j*omega*L_mut#/(1j*omega*L_self)


        S_mut = np.dot(v[:, 0], np.dot(S_mutual, v[:, 0]))
        S_self = np.dot(v[:, 0], np.dot(S_self, v[:, 0]))
        kappa_e_direct[count] = S_mut/S_self#/omega


    plt.figure()
    plt.plot(freqs*1e-9, kappa_e_direct.real, label="direct real")
    plt.plot(freqs*1e-9, kappa_e_direct.imag, label="direct imag")
    plt.legend()
    plt.title("kappa_e")
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(freqs*1e-9, kappa_m_direct.real, label="direct real")
    plt.plot(freqs*1e-9, kappa_m_direct.imag, label="direct imag")
    plt.legend()
    plt.title("kappa_m")
    plt.show()

    plt.figure()
    plt.plot(freqs*1e-9, *singular_terms) #y_srr.real)
    plt.plot(freqs*1e-9, y_srr.imag)
    plt.show()

def srr_extinction():
    """
    Calculate the excinction for a pair of SRRs
    """

    filename = osp.join("..", "examples", "geometry", "SRR.geo")
    freqs = np.linspace(2e9, 20e9, 201)

    mesh_tol = 2e-3
    srr = openmodes.load_parts(filename, mesh_tol)
    
    sim = openmodes.Simulation()    
    srr1 = sim.place_part(srr)

    separation = [20e-3, 0e-3, 0]
    srr2 = sim.place_part(srr, separation)    
    
    #separation = [500e-3, 0e-3, 0]
    #separation = [0e-3, 150e-3, 0]
    #separation = [150e-3, 0e-3, 0]
    #separation = [15e-3, 15e-3, 0]
    #separation = [0, 10e-3, 0]
    srr2.translate(separation)
    
    e_inc = np.array([1, 1, 0])
    k_hat = np.array([0, 0, 1])
    
    
    #y_srr = np.empty((len(freqs), n_modes), np.complex128)
    extinction = np.empty(len(freqs), np.complex128)
    
    for freq_count, freq in enumerate(freqs):
        s = 2j*np.pi*freq
        jk_0 = s/c        
        
        L, S = sim.impedance_matrix(s)

        V = sim.source_term(e_inc, k_hat*jk_0)
        extinction[freq_count] = np.dot(V.conj(), la.solve(s*L + S/s, V))
        
    plt.figure()
    plt.plot(freqs*1e-9,*singular_terms) # extinction.real)
    plt.plot(freqs*1e-9, extinction.imag)
    plt.show()


def test_plotting():

    filename = osp.join("..", "examples", "geometry", "SRR.geo")

    mesh_tol = 1e-3
    srr = openmodes.load_parts(filename, mesh_tol)

    sim = openmodes.Simulation()    
    srr1 = sim.place_part(srr)
    srr2 = sim.place_part(srr)    
    
    separation = [20e-3, 0e-3, 0]
    srr2.translate(separation)


    #set([srr1])
    plot_parts(sim.parts, view_angles=(40, 70))

def test_rwg():
    #from openmodes.mesh import TriangularSurfaceMesh
    #from openmodes.basis import DivRwgBasis
    #from openmodes import gmsh
    from openmodes import load_mesh, Simulation
    #from openmodes.operator import EfieOperator
    #from openmodes import integration
    
    filename = osp.join("..", "examples", "geometry", "SRR.geo")
    
    mesh_tol = 0.4e-3 #0.25e-3
    #meshed_name = gmsh.mesh_geometry(filename, mesh_tol)
    #
    #raw_mesh = gmsh.read_mesh(meshed_name)
    #
    #mesh = TriangularSurfaceMesh(raw_mesh[0])
    
    mesh = load_mesh(filename, mesh_tol)
    
    #plot_parts([mesh], view_angles=(40, 70))
    
    sim = Simulation()    
    
    srr = sim.place_part(mesh)
    
    #op = EfieOperator([part], integration.get_dunavant_rule(5))
    
    freqs = np.linspace(2e9, 20e9, 501)
    
    e_inc = np.array([1, 1, 0], dtype=np.complex128)
    k_hat = np.array([0, 0, 1], dtype=np.complex128)
        
    extinction = np.zeros(len(freqs), np.complex128)
    
    for freq_count, freq in enumerate(freqs):
        s = 2j*np.pi*freq
        jk_inc = s/c*k_hat
        
        L, S = sim.operator.impedance_matrix(s, srr)
        #V = sim.operator.source_plane_wave(sim.parts[0], s, e_inc, jk_inc)
        V = sim.operator.source_plane_wave(srr, e_inc, jk_inc)
    
        extinction[freq_count] = np.dot(V.conj(), la.solve(s*L + S/s, V))
       
    plt.figure()
    plt.plot(freqs*1e-9, extinction.real)
    plt.plot(freqs*1e-9, extinction.imag)
    plt.show()
    
    #basis = DivRwgBasis(mesh)
    #mes
        
def test_sphere():

    filename = osp.join("..", "examples", "geometry", "sphere.geo")
    mesh_tol = 0.4
    
    mesh = openmodes.load_mesh(filename, mesh_tol)    
    #sim = openmodes.Simulation(basis_class = LoopStarBasis)    
    sim = openmodes.Simulation(basis_class = DivRwgBasis)
    
    sphere = sim.place_part(mesh)
    basis = LoopStarBasis(mesh)
    
    k_num = np.linspace(0.1, 3, 51)
    
    freqs = c/(2*np.pi)*k_num
    
    e_inc = np.array([0, 1, 0], dtype=np.complex128)
    k_hat = np.array([0, 0, 1], dtype=np.complex128)
        
    extinction = np.zeros(len(freqs), np.complex128)
    
    a = 1.0
    
    for freq_count, freq in enumerate(freqs):
        s = 2j*np.pi*freq
        jk_inc = s/c*k_hat
        
        L, S = sim.operator.impedance_matrix(s, sphere)
        #V = sim.operator.source_plane_wave(sim.parts[0], s, e_inc, jk_inc)
        V = sim.operator.source_plane_wave(sphere, e_inc, jk_inc)
    
        extinction[freq_count] = np.dot(V.conj(), la.solve(s*L + S/s, V))
       
    from openmodes.constants import eta_0   
       
    plt.figure()
    plt.plot(k_num, extinction.real*eta_0/(np.pi*a**2))
    #plt.plot(k_num, extinction.imag)
    plt.show()

def reduced_impedance():
    "Plot the reduced self impedance"
    ring1, ring2 = openmodes.load_mesh(
                    osp.join("..", "examples", "geometry", "asymmetric_ring.geo"), mesh_tol=1e-3)
    
    #basis_class=LoopStarBasis
    basis_class=DivRwgBasis
    
    sim = openmodes.Simulation(basis_class=basis_class)
    part1 = sim.place_part(ring1)
    part2 = sim.place_part(ring2)
     
    
    e_inc = np.array([0, 1, 0], dtype=np.complex128)
    k_hat = np.array([0, 0, 1], dtype=np.complex128)
    
    
    freqs = np.linspace(1e9, 25e9, 51)
    
    L_list = []
    S_list = []
    
    P = []
    
    num_modes = 3
    
    extinction = np.empty(len(freqs), np.complex128)
    
    for freq in freqs:
        s = 2j*np.pi*freq
    
        impedance = sim.calculate_impedance(s)
        impedance.calculate_eigenmodes(num_modes)
        #P.append(w[0])
        L, S = impedance.impedance_reduced()
        
        
        V = impedance.source_reduced(sim.source_plane_wave(e_inc, k_hat*s/c))
        L_list.append(L)
        S_list.append(S)
    #    L = impedance.L_parts[1][1]
    #    S = impedance.S_parts[1][1]
    #    P.append(np.vdot(V, la.solve(s*L + S/s, V)))
    
    L_list = np.array(L_list)
    S_list = np.array(S_list)
    #
    P = np.array(P)
    
    plt.figure()
    plt.subplot(2, 1, 1)
    #plt.plot(freqs*1e-9, P.real, 'x')
    plt.plot(freqs*1e-9, L_list[:, 0, 0].real, 'x')
    plt.plot(freqs*1e-9, L_list[:, 1, 1].real, 'x')
    plt.plot(freqs*1e-9, L_list[:, 2, 2].real, 'x')
    
    plt.subplot(2, 1, 2)
    
    plt.plot(freqs*1e-9, L_list[:, 0, 0].imag, 'x')
    plt.plot(freqs*1e-9, L_list[:, 1, 1].imag, 'x')
    plt.plot(freqs*1e-9, L_list[:, 2, 2].imag, 'x')
    
    
    plt.show()
    
def coupled_extinction():
    """Calculate extinction for a coupled pair, showing that the reduced
    problem gives the same result as the full problem"""
    
    ring1, ring2 = openmodes.load_mesh(
        osp.join("..", "examples", "geometry", "asymmetric_ring.geo"), mesh_tol=1e-3)
    
    #basis_class=LoopStarBasis
    basis_class=DivRwgBasis
    
    sim = openmodes.Simulation(basis_class=basis_class)
    part1 = sim.place_part(ring1)
    part2 = sim.place_part(ring2)
     
    
    e_inc = np.array([1, 0, 0], dtype=np.complex128)
    k_hat = np.array([0, 0, 1], dtype=np.complex128)
        
    freqs = np.linspace(5e9, 15e9, 201)
    
    num_modes = 3
    
    extinction_red = np.empty(len(freqs), np.complex128)
    extinction_tot = np.empty(len(freqs), np.complex128)

    extinction_parts = np.empty((len(freqs), len(sim.parts)*num_modes), np.complex128)
    
    for freq_count, freq in enumerate(freqs):
        s = 2j*np.pi*freq
    
        impedance = sim.calculate_impedance(s)
        V = sim.source_plane_wave(e_inc, k_hat*s/c)
        
        z_mode, j_mode = impedance.eigenmodes(num_modes)
        impedance_red = impedance.impedance_modes(num_modes, j_mode)
        #L_red, S_red = impedance.impedance_reduced()
        V_red = impedance.source_modes(V, num_modes, j_mode)
        #extinction_red[freq_count] = np.vdot(V_red, la.solve(s*L_red + S_red/s, V_red))
        extinction_red[freq_count] = np.vdot(V_red, impedance_red.solve(V_red))
    
        impedance_combined = impedance.combine_parts()
        V_tot = np.hstack(V)
        #extinction_tot[freq_count] = np.vdot(V_tot, la.solve(s*L_tot + S_tot/s, V_tot))
        extinction_tot[freq_count] = np.vdot(V_tot, impedance_combined.solve(V_tot))
    
        #Z_red = impedance_red.evaluate()
        for i in xrange(len(sim.parts)*num_modes):
            extinction_parts[freq_count, i] = V_red[i].conj()*V_red[i]/impedance_red[i, i]
    
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(freqs*1e-9, extinction_red.real)
    plt.plot(freqs*1e-9, extinction_tot.real)
    plt.subplot(2, 1, 2)
    #plt.plot(freqs*1e-9, extinction_red.imag)
    #plt.plot(freqs*1e-9, extinction_tot.imag)
    plt.plot(freqs*1e-9, extinction_parts.real)
    plt.show()


def test_nonlinear_eig_asrr():
    ring1, ring2 = openmodes.load_mesh(
        osp.join("..", "examples", "geometry", "asymmetric_ring.geo"), mesh_tol=1e-3)
    
    basis_class=LoopStarBasis
    #basis_class=DivRwgBasis
    
    sim = openmodes.Simulation(basis_class=basis_class)
    part1 = sim.place_part(ring1)
    part2 = sim.place_part(ring2)
     
    e_inc = np.array([1.0, 0, 0], dtype=np.complex128)
    k_hat = np.array([0, 0, 1], dtype=np.complex128)
    
    start_freq = 15e9
    start_s = 2j*np.pi*start_freq
    
    s_modes, j_modes = sim.part_singularities(start_s, 1)
    
    print np.array(s_modes)/2/np.pi

def test_nonlinear_eig_srr():
    
    ring1 = openmodes.load_mesh(
            osp.join("..", "examples", "geometry", "SRR.geo"), mesh_tol=2e-3)
    
    
    basis_class=LoopStarBasis
    #basis_class=DivRwgBasis
    
    sim = openmodes.Simulation(basis_class=basis_class)
    part1 = sim.place_part(ring1)
    #part2 = sim.place_part(ring2)
     
    e_inc = np.array([1.0, 1.00, 0], dtype=np.complex128)
    k_hat = np.array([0, 0, 1], dtype=np.complex128)
    
    start_freq = 10e9
    start_s = 2j*np.pi*start_freq
    
    num_modes = 3
    
    s_modes, j_modes = sim.part_singularities(start_s, num_modes)
    
    print np.array(s_modes)/2/np.pi
    
    freqs = np.linspace(1e9, 25e9, 201)
    
#    extinction_red = np.empty((len(freqs), num_modes), np.complex128)
#    extinction_tot = np.empty(len(freqs), np.complex128)
#    
#    extinction_parts = np.empty((len(freqs), len(sim.parts)*num_modes), np.complex128)
#    
    w_list = []
    w2_list = []
    
    models = sim.construct_models(s_modes, j_modes)
    
    #print models[0][0].L
    #print models[0][0].S
    
    for freq_count, freq in enumerate(freqs):
        s = 2j*np.pi*freq
    
        impedance = sim.calculate_impedance(s)
        V = sim.source_plane_wave(e_inc, k_hat*s/c)
        if freq_count == 0:
            print len(V[0])
        
        w, vr = impedance.eigenmodes(num_modes)
        w_list.append(w[0])
        
        w2 = [model.scalar_impedance(s) for model in models[0]]
        w2_list.append(w2)
#        impedance_red = impedance.impedance_modes(num_modes, vr)
#        V_red = impedance.source_reduced(V, num_modes, vr)
#        #extinction_red[freq_count] = np.vdot(V_red, la.solve(s*L_red + S_red/s, V_red))
#        #extinction_red[freq_count] = V_red.conj()*V_red/np.diag(s*L_red + S_red/s)
#        extinction_red[freq_count] = V_red.conj()*V_red/np.diag(impedance_red[:])
#    
#        impedance_tot = impedance.impedance_combined()
#        V_tot = np.hstack(V)
#        #extinction_tot[freq_count] = np.vdot(V_tot, la.solve(s*L_tot + S_tot/s, V_tot))
#        extinction_tot[freq_count] = np.vdot(V_tot, impedance_tot.solve(V_tot))    
    
    w_list = 1.0/np.array(w_list)
    w2_list = 1.0/np.array(w2_list)
    
     
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
#    plt.plot(freqs*1e-9, w_list.real, 'x')
#    plt.plot(freqs*1e-9, w2_list.real)

    plt.semilogy(freqs*1e-9, abs(w_list.real), 'x')
    plt.semilogy(freqs*1e-9, abs(w2_list.real))

    plt.subplot(1, 2, 2)
    #plt.plot(freqs*1e-9, w_list.imag, 'x')
    #plt.plot(freqs*1e-9, w2_list.imag)

    plt.semilogy(freqs*1e-9, abs(w_list.imag), 'x')
    plt.semilogy(freqs*1e-9, abs(w2_list.imag))
    #plt.ylim(-2, 1)
    plt.show()

#plt.figure()
##plt.subplot(2, 1, 1)
#plt.plot(freqs*1e-9, extinction_red.real)
#plt.plot(freqs*1e-9, extinction_tot.real)
#plt.show()    

def vis_eigencurrents():
    ring1 = openmodes.load_mesh(
                        osp.join("..", "examples", "geometry", "SRR_wide.geo"),
                        mesh_tol=0.25e-3)
    
    basis_class=LoopStarBasis
    #basis_class=DivRwgBasis
    
    sim = openmodes.Simulation(basis_class=basis_class)
    sim.place_part(ring1, location=[0e-3, 0, 0])
    sim.place_part(ring1, location=[10e-3, 0, 0])
    sim.place_part(ring1, location=[20e-3, 0, 0])
    sim.place_part(ring1, location=[30e-3, 0, 0])

    start_freq = 2e9
    start_s = 2j*np.pi*start_freq
    
    num_modes = 4
    
    if True:
        s_modes, j_modes = sim.part_singularities(start_s, num_modes)
        #print np.array(s_modes)/2/np.pi
    else:
        impedance = sim.calculate_impedance(start_s)
        z_modes, j_modes = impedance.eigenmodes(num_modes)
    
    I = [j_modes[0][:, n] for n in xrange(num_modes)]
    #sim.plot_solution(I, 'mayavi')
    sim.plot_solution(I, 'vtk', filename='modes.vtk')
    
    return

def fit_mode():
    ring1 = openmodes.load_mesh(
                        osp.join("..", "examples", "geometry", "SRR.geo"),
                        mesh_tol=1e-3)
    
    basis_class=LoopStarBasis
    #basis_class=DivRwgBasis
    
    sim = openmodes.Simulation(basis_class=basis_class)
    part = sim.place_part(ring1)
    
    start_freq = 2e9
    start_s = 2j*np.pi*start_freq
    
    num_modes = 3
    
    #s_modes, j_modes = sim.part_singularities(start_s, num_modes)
    mode_s, mode_j = sim.part_singularities(start_s, num_modes)
    
    scalar_models = sim.construct_models(mode_s, mode_j)
            
    #from openmodes.solver import delta_eig, fit_circuit
    #
    #circuits = []
    #
    #print s_modes[0].imag/2/np.pi
    #
    #for mode_count in xrange(num_modes):
    #
    #    Z_func = lambda s: sim.operator.impedance_matrix(s, part)[:]
    #
    #    s_mode = s_modes[0][mode_count]
    #    z_der = delta_eig(s_mode, j_modes[0][:, mode_count], Z_func)
    #
    #    #circ = fit_circuit(s_modes[0][mode_count], z_der)
    #    #circuits.append((s_mode, fit_circuit(s_mode/s_mode.imag, z_der*s_mode.imag)))
    #    circuits.append((s_mode, fit_circuit(s_mode, z_der)))
    #    #print circ
    
    num_freqs = 101
    freqs = np.linspace(4e9, 20e9, num_freqs)
    
    #scale = circuits[1][0].imag
    #coeffs = circuits[1][1]
    
    #s_vals = 2j*np.pi*freqs/scale
    
    e_inc = np.array([1.0, 1.0, 0], dtype=np.complex128)/np.sqrt(2)
    k_hat = np.array([0, 0, 1], dtype=np.complex128)
    
    extinction = np.empty(num_freqs, np.complex128)
    extinction_modes = np.empty((num_freqs, num_modes), np.complex128)
    extinction_eig = np.empty((num_freqs, num_modes), np.complex128)
    z_mode = np.empty((num_freqs, num_modes), np.complex128)
    
    for freq_count, freq in enumerate(freqs):
        s = 2j*np.pi*freq
        Z = sim.calculate_impedance(s)
        V = sim.source_plane_wave(e_inc, k_hat*s/c)
        extinction[freq_count] = np.vdot(V[0], Z[0,0].solve(V[0]))
    
        z_eig, j_eig = Z[0,0].eigenmodes(num_modes)
    
        extinction_modes[freq_count] = [np.dot(V[0], scalar_models[0][mode].solve(s, V[0])) for mode in xrange(num_modes)]
    
    
    #    for mode in xrange(num_modes):
    #        #s_vals = s/circuits[mode][0].imag
    #        #s_vals = s#/circuits[mode][0].imag
    #        #z_mode = np.dot([1/s_vals, 1.0, s_vals, -s_vals**2], circuits[mode][1])
    #        #z_mode[freq_count, mode] = np.dot([1/s_vals, 1.0, s_vals, -s_vals**2], circuits[mode][1])
    #        #z_mode[freq_count, mode] = np.dot([1/s_vals, 1.0, s_vals, 0], circuits[mode][1])
    #        extinction_modes[freq_count, mode] = np.dot(V[0], scalar_models[0][mode].solve(s, V[0]))
    #        #extinction_modes[freq_count, mode] = np.dot(V[0], j_modes[0][:, mode])*np.dot(V[0].conj(), j_modes[0][:, mode])/z_mode[freq_count, mode]
    #        #extinction_eig[freq_count, mode] = np.dot(V[0], j_eig[:, mode])*np.dot(V[0].conj(), j_eig[:, mode])/z_eig[mode]
    #        #print np.dot(V[0], j_modes[0][:, mode])
    
    
    #s_powers = np.array([1/s_vals, np.ones(num_freqs), s_vals, -s_vals**2]).T
    
    #z = np.dot(s_powers, coeffs)
    #y = 1/z_mode
    
    plt.figure()
    plt.plot(freqs*1e-9, extinction.real)
    #plt.plot(freqs*1e-9, extinction.imag)
    #plt.plot(freqs*1e-9, extinction_modes.real)
    #plt.plot(freqs*1e-9, np.sum(extinction_modes.real, axis=1), '+')
    #plt.plot(freqs*1e-9, np.sum(extinction_eig.real, axis=1), 'x')
    plt.plot(freqs*1e-9, extinction_modes.real, '-')
    #plt.plot(freqs*1e-9, extinction_eig.real, '--')
    #plt.plot(freqs*1e-9, y.real)
    #plt.plot(freqs*1e-9, y.real)
    #plt.plot(freqs*1e-9, y.imag)
    plt.show()

#loop_star_linear_eigenmodes()
#srr_coupling()
#srr_extinction()
#test_plotting()
#compare_source_terms()
#test_rwg()
#coupled_extinction()
#vis_eigencurrents()
#test_nonlinear_eig_srr()
#loop_star_combined()
#sem_eem_asrr()
sem_eem_bcsrr()

