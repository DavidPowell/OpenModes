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
"Test the circle and square geometries"

import os.path as osp

import numpy as np
import matplotlib.pyplot as plt

import logging
logging.getLogger().setLevel(logging.INFO)

import openmodes
import openmodes.basis
from openmodes.constants import c
from openmodes.model import ScalarModelLeastSq

#name = 'circle'
#parameters = {'outer_radius': 20e-3, 'mesh_tol': 4e-3}

name = 'rectangle'
parameters = {'width': 12e-3, 'height': 25e-3, 'mesh_tol': 1e-3}

num_freqs = 101
freqs = np.linspace(1e8, 15e9, num_freqs)
num_modes = 2
mesh_tol = 1e-3
model_class = ScalarModelLeastSq


sim = openmodes.Simulation(name=name,
                           basis_class=openmodes.basis.LoopStarBasis)
mesh = sim.load_mesh(osp.join(openmodes.geometry_dir, name+'.geo'),
                     parameters=parameters)
part = sim.place_part(mesh)

s_start = 2j*np.pi*0.5*(freqs[0]+freqs[-1])

mode_s, mode_j = sim.singularities(s_start, num_modes, part)

models = sim.construct_models(mode_s, mode_j, part, model_class=model_class)

extinction = np.empty(num_freqs, np.complex128)
extinction_sem = np.empty((num_freqs, num_modes), np.complex128)
#extinction_eem = np.empty((num_freqs, num_modes), np.complex128)

e_inc = np.array([1, 1, 0], dtype=np.complex128)/np.sqrt(2)
k_hat = np.array([0, 0, 1], dtype=np.complex128)

z_sem = np.empty((num_freqs, num_modes), np.complex128)
#z_eem = np.empty((num_freqs, num_modes), np.complex128)
#    z_eem_direct = np.empty((num_freqs, num_modes), np.complex128)

for freq_count, s in sim.iter_freqs(freqs):
    Z = sim.impedance(s)
    V = sim.source_plane_wave(e_inc, s/c*k_hat)
    I = Z.solve(V)
    extinction[freq_count] = np.vdot(V[part], I)

    z_sem[freq_count] = [model.scalar_impedance(s) for model in models]
    extinction_sem[freq_count] = [np.vdot(V, model.solve(s, V))
                                  for model in models]

plt.figure(figsize=(10, 5))
plt.subplot(121)
plt.plot(freqs*1e-9, extinction.real)
plt.plot(freqs*1e-9, np.sum(extinction_sem.real, axis=1), '--')
#plt.plot(freqs*1e-9, extinction_sem.real, '--')
plt.xlabel('f (GHz)')
plt.subplot(122)
plt.plot(freqs*1e-9, extinction.imag)
plt.plot(freqs*1e-9, np.sum(extinction_sem.imag, axis=1), '--')
#plt.plot(freqs*1e-9, extinction_sem.imag, '--')
plt.suptitle("Extinction")
plt.show()
