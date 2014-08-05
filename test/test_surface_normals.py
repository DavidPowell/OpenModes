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
"Test the mesh surface normals"

import os.path as osp
from mayavi.mlab import quiver3d

import openmodes
from openmodes.integration import triangle_centres

name = 'horseshoe_rect'
parameters = {'mesh_tol': 1e-3}

#name = 'sphere'
#parameters = {'radius': 20e-3, 'mesh_tol': 4e-3}

#name = 'rectangle'
#parameters = {'width': 12e-3, 'height': 25e-3, 'mesh_tol': 1e-3}

sim = openmodes.Simulation(name=name)
mesh = sim.load_mesh(osp.join(openmodes.geometry_dir, name+'.geo'),
                     parameters=parameters)
part = sim.place_part(mesh)

basis = sim.basis_container[part]

r, rho = basis.integration_points(mesh.nodes, triangle_centres)
normals = mesh.surface_normals
r = r.reshape((-1, 3))

quiver3d(r[:, 0], r[:, 1], r[:, 2],
         normals[:, 0], normals[:, 1], normals[:, 2],
         mode='cone')
