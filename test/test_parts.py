# -*- coding: utf-8 -*-
"""
Created on Tue Jun 24 17:18:32 2014

@author: dap124
"""

from __future__ import print_function

import openmodes
import os.path as osp
import weakref


def test_part_references():
    """Check that weak and strong references to Parts work as expected"""

    def keep_ref():
        sim = openmodes.Simulation(name=name)
        mesh = sim.load_mesh(osp.join(openmodes.geometry_dir, name+'.geo'),
                             mesh_tol=mesh_tol)
        part = sim.place_part(mesh)
        return weakref.ref(part), part

    def discard_ref():
        sim = openmodes.Simulation(name=name)
        mesh = sim.load_mesh(osp.join(openmodes.geometry_dir, name+'.geo'),
                             mesh_tol=mesh_tol)
        part = sim.place_part(mesh)
        return weakref.ref(part)

    name = "SRR"
    mesh_tol = 1e-3

    # create a reference to a part object which stays in memory
    weakref1, strongref1 = keep_ref()
    print(weakref1, strongref1)
    assert('SinglePart' in repr(weakref1))
    assert('SinglePart' in repr(strongref1))

    # create a reference to a part object which is already discarded
    weakref2 = discard_ref()
    print(weakref2)
    assert('dead' in repr(weakref2))

    # delete the reference to the original part object
    del strongref1
    print(weakref2)
    assert('dead' in repr(weakref1))

if __name__ == "__main__":
    test_part_references()
