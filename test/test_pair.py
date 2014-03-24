# -*- coding: utf-8 -*-
"""
Created on Mon Mar 24 15:55:27 2014

@author: dap124
"""

import matplotlib.pyplot as plt

# the numpy library contains useful mathematical functions
import numpy as np

# import useful python libraries
import os.path as osp

# import the openmodes packages
import openmodes
from openmodes.constants import c, eta_0

sim = openmodes.Simulation(name='example1')

filename = osp.join(openmodes.geometry_dir, "SRR.geo")
mesh_tol = 2e-3
srr = sim.load_mesh(filename, mesh_tol)

srr1 = sim.place_part(srr)
srr2 = sim.place_part(srr)
srr2.rotate(axis = [0, 0, 1], angle = 180)
srr2.translate([0, 0, 2e-3])

s = 2j*np.pi*4.5e9

jk_0 = s/c    

e_inc = np.array([0, 1, 0])
k_hat = np.array([1, 0, 0])

 
Z = sim.impedance(s)
V = sim.source_plane_wave(e_inc, k_hat*jk_0)


Z_single = Z[srr2, srr2]
V_single = V[srr2]
#extinction_single = np.vdot(V_single, Z_single.solve(V_single)).real

#extinction_pair = np.vdot(V, Z.solve(V)).real


