# -*- coding: utf-8 -*-
"""
Created on Wed Jun 18 16:12:15 2014

@author: dap124
"""
from __future__ import print_function

import openmodes
from openmodes.sources import PlaneWaveSource

import os
import os.path as osp
import dill as pickle
import tempfile
import numpy as np


def test_pickling_references():
    """Test that references to parent objects survive the pickling and
    upickling process"""

    def save():
        name = "SRR"
        mesh_tol = 1e-3

        sim = openmodes.Simulation(name=name)
        mesh = sim.load_mesh(osp.join(openmodes.geometry_dir, name+'.geo'),
                             mesh_tol=mesh_tol)
        parent_part = sim.place_part()
        sim.place_part(mesh, parent=parent_part)
        sim.place_part(mesh, parent=parent_part, location=[10, 10, 10])
        sim.place_part(mesh, location=[10, 10, 10])

        parents_dict = {}
        for part in sim.parts.iter_all():
            print("Original part", part)
            if part.parent_ref is None:
                parents_dict[str(part.id)] = 'None'
            else:
                parents_dict[str(part.id)] = str(part.parent_ref().id)

        pw = PlaneWaveSource([0, 1, 0], [0, 0, 1])
        V = sim.source_vector(pw, 0)

        with tempfile.NamedTemporaryFile(delete=False) as output_file:
            file_name = output_file.name
            pickle.dump(V, output_file, protocol=0)

        return file_name, parents_dict

    def load(file_name):
        with open(file_name, "rb") as infile:
            V = pickle.load(infile)

        parents_dict = {}
        for part in V.lookup[1][0].keys():
            print("Unpickled part", part)
            if part.parent_ref is None:
                parents_dict[str(part.id)] = 'None'
            else:
                parents_dict[str(part.id)] = str(part.parent_ref().id)

        return parents_dict

    file_name, original_parents_dict = save()
    loaded_parents_dict = load(file_name)

    # direct comparison of dictionaries seems to work
    assert(original_parents_dict == loaded_parents_dict)

    print("original parent references", original_parents_dict)
    print("loaded parent references", loaded_parents_dict)
    os.remove(file_name)


def test_empty_array():
    "Confirm that an empty array has the expected dimensions and lookup"
    sim = openmodes.Simulation()
    name = "SRR"
    mesh_tol = 1e-3

    sim = openmodes.Simulation(name=name)
    mesh = sim.load_mesh(osp.join(openmodes.geometry_dir, name+'.geo'),
                         mesh_tol=mesh_tol)
    part1 = sim.place_part(mesh)
    part2 = sim.place_part(mesh, location=[0, 0, 5e-3])

    vec = sim.empty_array()

    s = 2j*np.pi*1e9
    Z = sim.impedance(s)
    pw = PlaneWaveSource([0, 1, 0], [0, 0, 1])
    V = sim.source_vector(pw, 0)
    I = Z.solve(V)

    assert(vec.shape == I.shape)
    assert(vec.lookup == I.lookup)

    vec2 = sim.empty_array(part2)
    assert(vec2.shape == I[:, part2].shape)
    assert(vec2.lookup == I[:, part2].lookup)

if __name__ == "__main__":
    test_pickling_references()
    test_empty_array()
