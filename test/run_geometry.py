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

import logging
logging.getLogger().setLevel(logging.DEBUG)

import openmodes
import openmodes.basis
from openmodes.constants import c
from openmodes.model import ScalarModelLeastSq


def geometry_extinction_modes(name, freqs, num_modes, mesh_tol,
                              plot_currents=False, plot_admittance=True,
                              parameters={}, model_class=ScalarModelLeastSq):
    """Load a geometry file, calculate its modes by searching for
    singularities, and plot them in 3D. Then use the modes to calculate
    extinction, and compare with exact calculation
    """

    sim = openmodes.Simulation(name=name,
                               basis_class=openmodes.basis.LoopStarBasis)
    mesh = sim.load_mesh(osp.join(openmodes.geometry_dir, name+'.geo'),
                         mesh_tol=mesh_tol, parameters=parameters)
    part = sim.place_part(mesh)

    s_start = 2j*np.pi*0.5*(freqs[0]+freqs[-1])

    mode_s, mode_j = sim.singularities(s_start, num_modes, part)

    if plot_currents:
        for mode in xrange(num_modes):
            current = sim.empty_vector()
            current[:] = mode_j[:, mode]
            sim.plot_3d(solution=current, output_format='mayavi',
                        compress_scalars=1)

    if not plot_admittance:
        return

    models = sim.construct_models(mode_s, mode_j, part,
                                  model_class=model_class)

    num_freqs = len(freqs)

    extinction = np.empty(num_freqs, np.complex128)
    extinction_sem = np.empty((num_freqs, num_modes), np.complex128)
    extinction_eem = np.empty((num_freqs, num_modes), np.complex128)

    e_inc = np.array([1, 1, 0], dtype=np.complex128)/np.sqrt(2)
    k_hat = np.array([0, 0, 1], dtype=np.complex128)

    z_sem = np.empty((num_freqs, num_modes), np.complex128)
    z_eem = np.empty((num_freqs, num_modes), np.complex128)
#    z_eem_direct = np.empty((num_freqs, num_modes), np.complex128)

    for freq_count, s in sim.iter_freqs(freqs):
        Z = sim.impedance(s)
        V = sim.source_plane_wave(e_inc, s/c*k_hat)
        I = Z.solve(V)
        extinction[freq_count] = np.vdot(V[part], I)

        z_sem[freq_count] = [model.scalar_impedance(s) for model in models]
        extinction_sem[freq_count] = [np.vdot(V, model.solve(s, V))
                                      for model in models]

#        z_eem_direct[freq_count], _ = Z.eigenmodes(num_modes, use_gram=False)
        z_eem[freq_count], j_eem = Z.eigenmodes(start_j=mode_j, use_gram=True)
        extinction_eem[freq_count] = [np.vdot(V, j_eem[:, mode]) *
                                      V.dot(j_eem[:, mode]) / z_eem[freq_count, mode]
                                      for mode in xrange(num_modes)]

    plt.figure(figsize=(10, 5))
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

    y_sem = 1/z_sem
    y_eem = 1/z_eem

    z_sem *= 2j*np.pi*freqs[:, None]
    z_eem *= 2j*np.pi*freqs[:, None]

    plt.figure(figsize=(10, 5))
    plt.subplot(121)
    plt.plot(freqs*1e-9, z_eem.real)
    plt.plot(freqs*1e-9, z_sem.real, '--')
    plt.xlabel('f (GHz)')
    plt.subplot(122)
    plt.plot(freqs*1e-9, z_eem.imag)
    plt.plot(freqs*1e-9, z_sem.imag, '--')
    plt.suptitle("SEM and EEM impedance")
    plt.show()

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


geometry_extinction_modes('horseshoe_rect', np.linspace(1e8, 20e9, 101),
                          3, 1.5e-3, plot_currents=True)
#geometry_extinction_modes('sphere', np.linspace(0.2e7, 8e7, 101), 8, 0.2)
#geometry_extinction_modes('canonical_spiral', np.linspace(1e8, 15e9, 101),
#                          3, 1e-3, parameters={'arm_length': 12e-3,
#                                               'inner_radius': 2e-3},
#                          plot_currents=False, plot_admittance=True)
#geometry_extinction_modes('v_antenna', np.linspace(1e8, 15e9, 101),
#                          6, 1.5e-3, model_class=ScalarModelLeastSq)

#geometry_extinction_modes('SRR', np.linspace(1e8, 20e9, 101), 4, 1e-3,
#                          plot_currents=False, plot_admittance=True,
#                          model_class=ScalarModelLeastSq)

#geometry_extinction_modes('cross', np.linspace(1e8, 20e9, 101), 2, 1e-3,
#                          plot_currents=False, plot_admittance=True)

#geometry_extinction_modes('closed_ring', np.linspace(1e8, 15e9, 101),
#                          2, 1e-3, model_class=ScalarModelLeastSq,
#                          plot_currents=True, plot_admittance=False,
#                          parameters={'inner_radius': 3e-3,
#                                      'outer_radius': 6e-3})
