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
from __future__ import print_function

import os.path as osp

import numpy as np
from numpy.testing import assert_allclose
import matplotlib.pyplot as plt
import scipy.linalg as la

import openmodes
import openmodes.basis
from openmodes.sources import PlaneWaveSource
from openmodes.constants import c
from openmodes.integration import triangle_centres

from helpers import (read_1d_complex, write_1d_complex,
                     read_2d_real, write_2d_real)

tests_location = osp.split(__file__)[0]
mesh_dir = osp.join(tests_location, 'input', 'test_horseshoe')
reference_dir = osp.join(tests_location, 'reference', 'test_horseshoe')


def assert_allclose_sign(a, b, rtol):
    """Compare two arrays which should be equal, to within a sign ambiguity
    of the whole array (not of each element)"""
    assert(np.all(np.abs(a-b) < rtol*abs(a)) or
           np.all(np.abs(a+b) < rtol*abs(a)))


def test_horseshoe_modes(plot=False, skip_asserts=False,
                         write_reference=False):
    "Modes of horseshoe"
    sim = openmodes.Simulation(name='horseshoe_modes',
                               basis_class=openmodes.basis.LoopStarBasis)
    shoe = sim.load_mesh(osp.join(mesh_dir, 'horseshoe_rect.msh'))
    sim.place_part(shoe)

    s_start = 2j*np.pi*10e9

    mode_s, mode_j = sim.singularities(s_start, 3)
    print("Singularities found at", mode_s)

    if write_reference:
        write_1d_complex(osp.join(reference_dir, 'eigenvector_0.txt'),
                         mode_j["J", :, 0])
        write_1d_complex(osp.join(reference_dir, 'eigenvector_1.txt'),
                         mode_j["J", :, 1])
        write_1d_complex(osp.join(reference_dir, 'eigenvector_2.txt'),
                         mode_j["J", :, 2])

    j_0_ref = read_1d_complex(osp.join(reference_dir, 'eigenvector_0.txt'))
    j_1_ref = read_1d_complex(osp.join(reference_dir, 'eigenvector_1.txt'))
    j_2_ref = read_1d_complex(osp.join(reference_dir, 'eigenvector_2.txt'))

    if not skip_asserts:
        assert_allclose(mode_s, [-2.585729e+09 + 3.156438e+10j,
                                 -1.887518e+10 + 4.500579e+10j,
                                 -1.991163e+10 + 6.846221e+10j],
                        rtol=1e-3)
        assert_allclose_sign(mode_j["J", :, 0], j_0_ref, rtol=1e-2)
        assert_allclose_sign(mode_j["J", :, 1], j_1_ref, rtol=1e-2)
        assert_allclose_sign(mode_j["J", :, 2], j_2_ref, rtol=1e-2)

    if plot:
        sim.plot_3d(solution=mode_j["J", :, 0], output_format='mayavi',
                    compress_scalars=3)
        sim.plot_3d(solution=mode_j["J", :, 1], output_format='mayavi',
                    compress_scalars=3)
        sim.plot_3d(solution=mode_j["J", :, 2], output_format='mayavi',
                    compress_scalars=3)


def test_surface_normals(plot=False, skip_asserts=False,
                         write_reference=False):
    "Test the surface normals of a horseshoe mesh"
    sim = openmodes.Simulation()
    mesh = sim.load_mesh(osp.join(mesh_dir, 'horseshoe_rect.msh'))
    part = sim.place_part(mesh)
    basis = sim.basis_container[part]

    r, rho = basis.integration_points(mesh.nodes, triangle_centres)
    normals = mesh.surface_normals
    r = r.reshape((-1, 3))

    if write_reference:
        write_2d_real(osp.join(reference_dir, 'surface_r.txt'), r)
        write_2d_real(osp.join(reference_dir, 'surface_normals.txt'), normals)

    r_ref = read_2d_real(osp.join(reference_dir, 'surface_r.txt'))
    normals_ref = read_2d_real(osp.join(reference_dir, 'surface_normals.txt'))

    if not skip_asserts:
        assert_allclose(r, r_ref)
        assert_allclose(normals, normals_ref)

    if plot:
        from mayavi import mlab
        mlab.figure()
        mlab.quiver3d(r[:, 0], r[:, 1], r[:, 2],
                      normals[:, 0], normals[:, 1], normals[:, 2],
                      mode='cone')
        mlab.view(distance='auto')
        mlab.show()


def test_extinction(plot_extinction=False, skip_asserts=False,
                    write_reference=False):
    "Test extinction of a horseshoe"
    sim = openmodes.Simulation(name='horseshoe_extinction',
                               basis_class=openmodes.basis.LoopStarBasis)

    shoe = sim.load_mesh(osp.join(mesh_dir, 'horseshoe_rect.msh'))
    sim.place_part(shoe)

    num_freqs = 101
    freqs = np.linspace(1e8, 20e9, num_freqs)

    extinction = np.empty(num_freqs, np.complex128)

    e_inc = np.array([1, 0, 0], dtype=np.complex128)
    k_hat = np.array([0, 0, 1], dtype=np.complex128)
    pw = PlaneWaveSource(e_inc, k_hat)

    for freq_count, s in sim.iter_freqs(freqs):
        Z = sim.impedance(s)
        V = sim.source_vector(pw, s)
        extinction[freq_count] = np.vdot(V, Z.solve(V))

    if write_reference:
        # generate the reference extinction solution
        write_1d_complex(osp.join(reference_dir, 'extinction.txt'), extinction)

    extinction_ref = read_1d_complex(osp.join(reference_dir, 'extinction.txt'))

    if not skip_asserts:
        assert_allclose(extinction, extinction_ref, rtol=1e-3)

    if plot_extinction:
        # to plot the generated and reference solutions
        plt.figure()
        plt.plot(freqs*1e-9, extinction.real)
        plt.plot(freqs*1e-9, extinction_ref.real, '--')
        plt.plot(freqs*1e-9, extinction.imag)
        plt.plot(freqs*1e-9, extinction_ref.imag, '--')
        plt.xlabel('f (GHz)')
        plt.show()


def horseshoe_extinction_modes():
    sim = openmodes.Simulation(name='horseshoe_extinction_modes',
                               basis_class=openmodes.basis.LoopStarBasis)
    shoe = sim.load_mesh(osp.join('input', 'test_horseshoe',
                                  'horseshoe_rect.msh'))
    sim.place_part(shoe)

    s_start = 2j*np.pi*10e9

    num_modes = 5
    mode_s, mode_j = sim.part_singularities(s_start, num_modes)

    models = sim.construct_models(mode_s, mode_j)[0]

    num_freqs = 101
    freqs = np.linspace(1e8, 20e9, num_freqs)

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

        z_eem_direct[freq_count], _ = Z.eigenmodes(num_modes, use_gram=False)

        z_eem[freq_count], j_eem = Z.eigenmodes(start_j = mode_j[0], use_gram=True)
        extinction_eem[freq_count] = [np.vdot(V, j_eem[:, mode])*np.dot(V, j_eem[:, mode])/z_eem[freq_count, mode] for mode in range(num_modes)]

    plt.figure(figsize=(10, 5))
    plt.subplot(121)
    plt.plot(freqs*1e-9, extinction.real)
    plt.plot(freqs*1e-9, np.sum(extinction_sem.real, axis=1), '--')
    plt.plot(freqs*1e-9, np.sum(extinction_eem.real, axis=1), '-.')
    plt.xlabel('f (GHz)')
    plt.subplot(122)
    plt.plot(freqs*1e-9, extinction.imag)
    plt.plot(freqs*1e-9, np.sum(extinction_sem.imag, axis=1), '--')
    plt.plot(freqs*1e-9, np.sum(extinction_eem.imag, axis=1), '-.')
    plt.suptitle("Extinction")
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.subplot(121)
    plt.plot(freqs*1e-9, z_eem_direct.real)
    #plt.ylim(0, 80)
    plt.xlabel('f (GHz)')
    plt.subplot(122)
    plt.plot(freqs*1e-9, z_eem_direct.imag)
    plt.plot([freqs[0]*1e-9, freqs[-1]*1e-9], [0, 0], 'k')
    plt.suptitle("EEM impedance without Gram matrix")
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.subplot(121)
    plt.plot(freqs*1e-9, z_eem.real)
    plt.plot(freqs*1e-9, z_sem.real, '--')
    #plt.ylim(0, 80)
    plt.xlabel('f (GHz)')
    plt.subplot(122)
    plt.plot(freqs*1e-9, z_eem.imag)
    plt.plot(freqs*1e-9, z_sem.imag, '--')
    plt.ylim(-100, 100)
#    plt.semilogy(freqs*1e-9, abs(z_eem.imag))
#    plt.semilogy(freqs*1e-9, abs(z_sem.imag), '--')
    plt.suptitle("SEM and EEM impedance")
    plt.show()

    y_sem = 1/z_sem
    y_eem = 1/z_eem

    plt.figure(figsize=(10, 5))
    plt.subplot(121)
    plt.plot(freqs*1e-9, y_eem.real)
    plt.plot(freqs*1e-9, y_sem.real, '--')
    plt.xlabel('f (GHz)')
    plt.subplot(122)
    plt.plot(freqs*1e-9, y_eem.imag)
    plt.plot(freqs*1e-9, y_sem.imag, '--')
    plt.xlabel('f (GHz)')
    plt.suptitle("SEM and EEM admittance")
    plt.show()

if __name__ == "__main__":
    #test_extinction_modes()
    test_horseshoe_modes(plot=True, skip_asserts=True)
    test_extinction(plot_extinction=True, skip_asserts=True)
    test_surface_normals(plot=True, skip_asserts=True)
