# -*- coding: utf-8 -*-
"""
Created on Thu Dec 05 14:06:30 2013

@author: dap124
"""

from openmodes.mesh import combine_mesh
from openmodes.parts import Part
from openmodes.visualise import plot_parts

import openmodes
import os.path as osp

def test_combine():
    ring1 = openmodes.load_mesh(
                        osp.join(openmodes.geometry_dir, "SRR_wide.geo"),
                        mesh_tol=0.5e-3)
    
    sim = openmodes.Simulation()
    sim.place_part(ring1, location=[0e-3, 0, 0])
    sim.place_part(ring1, location=[10e-3, 0, 0])
    sim.place_part(ring1, location=[20e-3, 0, 0])
    sim.place_part(ring1, location=[30e-3, 0, 0])

    meshes = [part.mesh for part in sim.parts]
    nodes = [part.nodes for part in sim.parts]
    
    mesh = combine_mesh(meshes, nodes)
    part = Part(mesh)
    plot_parts([part])
    
    
test_combine()
