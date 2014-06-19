# -*- coding: utf-8 -*-
"""
Created on Wed Jun 18 16:12:15 2014

@author: dap124
"""

import openmodes
import openmodes.basis
import os.path as osp
import numpy as np
import pickle
    

def save():
    name = "SRR"
    mesh_tol = 1e-3
    
    sim = openmodes.Simulation(name=name, 
                               log_display_level=20)
    mesh = sim.load_mesh(osp.join(openmodes.geometry_dir, name+'.geo'),
                         mesh_tol=mesh_tol)
    part = sim.place_part(mesh)
    
    s = 2j*np.pi*1e9
    V = sim.source_plane_wave([0, 1, 0], [0, 0, 0])
    
    with open(osp.join("output", "V.pickle"), "wt") as outfile:
        pickle.dump(V, outfile, protocol=0)

def load():        
    with open(osp.join("output", "V.pickle"), "rt") as infile:
        V = pickle.load(infile)
    print V
    part = V.index_arrays.keys()[0]
    print part.nodes

load()
#save()