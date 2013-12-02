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
from openmodes.visualise import plot_parts, write_vtk
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
    
    L, S = sim.operator.impedance_matrix(s, part1)
    V = sim.operator.source_plane_wave(part1, np.array([0, 1, 0]), np.array([0, 0, 0]))
    
    #I = la.solve(s*L + S/s, V)
    
    #power = np.dot(V.conj(), I)
    
    n_modes = 3
    
    basis = basis_class(ring1)
    
    w, vr = sim.eig_linearised(part1, L, S, n_modes)
    
    #I = np.zeros(len(basis), np.complex128)
    #I[0] = 1.0
    
    I = vr[:, 0]
    
    #face_current = openmodes.basis.rwg_to_triangle_face(I, len(basis.mesh.polygons), basis.rwg)
    face_centre, face_current, face_charge = basis.interpolate_function(I, return_scalar=True)
    
    
    
    write_vtk(part1.mesh, part1.nodes, osp.join("output", "test.vtk"), 
              vector_function = face_current, scalar_function=face_charge
              )
    
#plt.loglog(abs(w.real), abs(w.imag), 'x')

#    power_modes = np.empty(n_modes, np.complex128)
#
#    for mode in xrange(n_modes):
#
#        S_mode = np.dot(vr[:, mode], np.dot(S, vr[:, mode]))
#        L_mode = np.dot(vr[:, mode], np.dot(L, vr[:, mode]))
#        power_modes[mode] = s*np.dot(vr[:, mode], V)*np.dot(vr[:, mode], V.conj())/(S_mode+s**2*L_mode)
#
#    print power, sum(power_modes)
#    print power_modes
 
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
    
    
    #basis_class=LoopStarBasis
    basis_class=DivRwgBasis
    
    sim = openmodes.Simulation(basis_class=basis_class)
    part1 = sim.place_part(ring1)
    #part2 = sim.place_part(ring2)
     
    e_inc = np.array([1.0, 1.00, 0], dtype=np.complex128)
    k_hat = np.array([0, 0, 1], dtype=np.complex128)
    
    start_freq = 10e9
    start_s = 2j*np.pi*start_freq
    
    num_modes = 3
    
    #s_modes, j_modes = sim.part_singularities(start_s, num_modes)
    #
    #print np.array(s_modes)/2/np.pi
    
    freqs = np.linspace(1e9, 20e9, 201)
    
#    extinction_red = np.empty((len(freqs), num_modes), np.complex128)
#    extinction_tot = np.empty(len(freqs), np.complex128)
#    
#    extinction_parts = np.empty((len(freqs), len(sim.parts)*num_modes), np.complex128)
#    
    w_list = []
    
    for freq_count, freq in enumerate(freqs):
        s = 2j*np.pi*freq
    
        impedance = sim.calculate_impedance(s)
        V = sim.source_plane_wave(e_inc, k_hat*s/c)
        if freq_count == 0:
            print len(V[0])
        
        w, vr = impedance.eigenmodes(num_modes)
        w_list.append(w[0])
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
    
    w_list = np.array(w_list)
    
     
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(freqs*1e-9, w_list.real, 'x')
    plt.subplot(1, 2, 2)
    plt.plot(freqs*1e-9, w_list.imag, 'x')
    #plt.ylim(-2, 1)
    plt.show()

#plt.figure()
##plt.subplot(2, 1, 1)
#plt.plot(freqs*1e-9, extinction_red.real)
#plt.plot(freqs*1e-9, extinction_tot.real)
#plt.show()    

#from mayavi import mlab

from openmodes.visualise import plot_mayavi

def vis_eigencurrents():
    ring1 = openmodes.load_mesh(
                        osp.join("..", "examples", "geometry", "SRR.geo"), mesh_tol=0.5e-3)
    #ring1 = openmodes.load_mesh(osp.join("geometry", "SRR.msh"))
    
    
    basis_class=LoopStarBasis
    #basis_class=DivRwgBasis
    
    sim = openmodes.Simulation(basis_class=basis_class)
    part1 = sim.place_part(ring1)
    part2 = sim.place_part(ring1, location=[10e-3, 0, 0])
    #part2 = sim.place_part(ring2)
     
    start_freq = 2e9
    start_s = 2j*np.pi*start_freq
    
    num_modes = 4
    
    if True:
        s_modes, j_modes = sim.part_singularities(start_s, num_modes)
        #print np.array(s_modes)/2/np.pi
    else:
        impedance = sim.calculate_impedance(start_s)
        z_modes, j_modes = impedance.eigenmodes(num_modes)
    
    basis = get_basis_functions(ring1, basis_class)
    
    nodes = ring1.nodes
    triangle_nodes = ring1.polygons
    
    #plot_mayavi(ring1, nodes)
    #return    
    
    for mode in xrange(num_modes):
        I = j_modes[0][:, mode]
        face_centre, face_current, face_charge = basis.interpolate_function(I, return_scalar=True)
#        x = face_centre[:, 0] 
#        y = face_centre[:, 1] 
#        z = face_centre[:, 2] 
        
        # normalise maximum charge for convenience of plotting
        face_charge /= max(abs(face_charge))

        plot_mayavi(ring1, nodes, face_current, face_charge.real, vector_points=face_centre)

            
        #plot_parts(sim.parts)
#        write_vtk(part1.mesh, part1.nodes, osp.join("output", "srr-mode-%d.vtk" % mode), 
#                  vector_function = face_current, scalar_function=face_charge
#                  )
    
#test_nonlinear_eig_srr()

#loop_star_linear_eigenmodes()
#srr_coupling()
#srr_extinction()
#test_plotting()
#compare_source_terms()
#test_rwg()
#coupled_extinction()
vis_eigencurrents()