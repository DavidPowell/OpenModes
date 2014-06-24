# -*- coding: utf-8 -*-
"""
Created on Tue Jun 24 17:18:32 2014

@author: dap124
"""

import openmodes
import os.path as osp
import weakref
import gc

name = "SRR"
mesh_tol = 1e-3

def runit():
    sim = openmodes.Simulation(name=name)
    mesh = sim.load_mesh(osp.join(openmodes.geometry_dir, name+'.geo'),
                         mesh_tol=mesh_tol)
    part = sim.place_part(mesh)
    return weakref.ref(part)

print "creating objects"
w1 = runit()
w2 = runit()
print w1, w2

print "garbage collecting"
gc.collect()
print w1, w2
