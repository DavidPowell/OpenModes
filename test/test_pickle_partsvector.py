# -*- coding: utf-8 -*-
"""
Created on Wed Jun 18 16:12:15 2014

@author: dap124
"""

import openmodes
import openmodes.basis
import os.path as osp
#import numpy as np
import pickle

import logging
logging.getLogger().setLevel(100)  # logging.INFO)


def save():
    name = "SRR"
    mesh_tol = 1e-3

    sim = openmodes.Simulation(name=name)
    mesh = sim.load_mesh(osp.join(openmodes.geometry_dir, name+'.geo'),
                         mesh_tol=mesh_tol)
    part = sim.place_part(mesh)
    for part in sim.parts.iter_all():
        print "Original part", part

    V = sim.source_plane_wave([0, 1, 0], [0, 0, 0])

    with open(osp.join("output", "V.pickle"), "wt") as outfile:
        pickle.dump(V, outfile, protocol=0)


def load():
    with open(osp.join("output", "V.pickle"), "rt") as infile:
        V = pickle.load(infile)

    for part in V.index_arrays.keys():
        print "Unpickled part", part
        if part.parent_ref is None:
            print "No parent"
        else:
            print "With parent", part.parent_ref()

save()
load()
