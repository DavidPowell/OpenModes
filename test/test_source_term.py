# -*- coding: utf-8 -*-
"""
Created on Fri Jun 20 11:47:11 2014

@author: dap124
"""

import openmodes
from openmodes.sources import PlaneWaveSource
from openmodes.basis import get_basis_functions
from openmodes.constants import c
import numpy as np
import os.path as osp
import matplotlib.pyplot as plt

import logging
logging.getLogger().setLevel(logging.INFO)

name = "SRR"
mesh_tol = 1e-3

sim = openmodes.Simulation(name="Source test")
mesh = sim.load_mesh(osp.join(openmodes.geometry_dir, name+'.geo'),
                     mesh_tol=mesh_tol)
part = sim.place_part(mesh)

s = 2j*np.pi*1e9

e_inc = [0, 1, 0]

k_dir = np.array([1, 0, 1])

V_old = sim.source_plane_wave(e_inc, k_dir/np.sqrt(sum(abs(k_dir)**2))*s/c)

pw = PlaneWaveSource(e_inc, k_dir)
E_field = lambda r: pw.electric_field(s, r)

basis = get_basis_functions(part.mesh, sim.basis_class)

xi_eta, w = sim.quadrature_rule
V_new = basis.weight_function(E_field, xi_eta, w[0], part.nodes)

plt.figure()
plt.plot(V_old.real)
plt.plot(V_new.real)
plt.show()


plt.figure()
plt.plot(V_old.imag)
plt.plot(V_new.imag)
plt.show()
