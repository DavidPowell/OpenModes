# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 16:18:59 2014

@author: dap124
"""

import os.path as osp
import numpy as np
import scipy.linalg as la

import openmodes
from openmodes.basis import LoopStarBasis
from openmodes.eig import eig_newton_linear

ring1 = openmodes.load_mesh(
                    osp.join("..", "examples", "geometry", "SRR.geo"),
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