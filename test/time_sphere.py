# -*- coding: utf-8 -*-
"""
Created on Thu Feb 26 09:36:46 2015

@author: dap124
"""

from __future__ import print_function

import os.path as osp

import numpy as np

import openmodes
from openmodes.basis import DivRwgBasis
from openmodes.operator import MfieOperator, EfieOperator

import logging
logging.getLogger().setLevel(logging.INFO)

tests_location = osp.split(__file__)[0]
mesh_dir = osp.join(tests_location, 'input', 'test_sphere')
reference_dir = osp.join(tests_location, 'reference', 'test_sphere')

def test_extinction_all(plot_extinction=False, skip_asserts=False,
                        write_reference=False):
    "Extinction of a PEC sphere with EFIE, MFIE, CFIE"

    tests = (("EFIE", EfieOperator, 'extinction_efie.npy'),
             ("MFIE", MfieOperator, 'extinction_mfie.npy'),
             )

    for operator_name, operator_class, reference_filename in tests:

        sim = openmodes.Simulation(name='horseshoe_extinction',
                                   basis_class=DivRwgBasis,
                                   operator_class=operator_class)

        sphere = sim.load_mesh(osp.join(mesh_dir, 'sphere.msh'))
        sim.place_part(sphere)

        s = 2j*np.pi*2e9
        Z = sim.impedance(s)

if __name__ == "__main__":
    test_extinction_all(plot_extinction=True, skip_asserts=True)
