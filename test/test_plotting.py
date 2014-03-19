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


def plot_sem_currents():
    """"Calculate the eigencurrent expansion of a single ring.
    """

    srr = openmodes.load_mesh(
                    osp.join(openmodes.geometry_dir, "SRR_wide.geo"), mesh_tol=0.5e-3)

    basis_class = LoopStarBasis
    #basis_class = DivRwgBasis

    sim = openmodes.Simulation(basis_class=basis_class)
    sim.place_part(srr)

    num_modes = 1

    s_start = 2j*np.pi*10e9
    mode_s, mode_j = sim.part_singularities(s_start, num_modes, use_gram=False)
    print [s/(2*np.pi) for s in mode_s]
    sim.plot_solution([mode_j[0][:, 0]], 'mayavi')
    
plot_sem()
