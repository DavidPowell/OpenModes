# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 15:50:37 2015

@author: dap124
"""

from __future__ import print_function

import os.path as osp

import numpy as np
from numpy.testing import assert_allclose

import matplotlib.pyplot as plt

import openmodes
from openmodes.basis import DivRwgBasis, LoopStarBasis
from openmodes.operator import EfieOperator

import logging
logging.getLogger().setLevel(logging.INFO)


def test_srr_pair_combined_poles(plot_figures=False):
    "Cauchy integral for poles of an SRR pair considered as a single part"

    sim = openmodes.Simulation(basis_class=LoopStarBasis,
                               operator_class=EfieOperator)

    parameters = {'inner_radius': 2.5e-3,
                  'outer_radius': 4e-3}
    mesh = sim.load_mesh(osp.join(openmodes.geometry_dir, 'SRR.geo'),
                         parameters=parameters, mesh_tol=0.5e-3)
    srr1 = sim.place_part(mesh)
    srr2 = sim.place_part(mesh, location=[0, 0, 1e-3])

    s_min = -0.5e11-1e9j
    s_max = -2e6+1.2e11j
    integration_line_i = np.array((s_min.imag, s_max.imag, s_max.imag,
                                   s_min.imag, s_min.imag))
    integration_line_r = np.array((s_min.real, s_min.real, s_max.real,
                                   s_max.real, s_min.real))

    # calculate modes of the SRR pair
    result = sim.estimate_poles(s_min, s_max)
    refined = sim.refine_poles(result)

    s_estimate = result['s']
    s_refined = refined['s']

    s_reference = [-1.18100087e+09 + 4.30266982e+10j,
                   -1.34656915e+07 + 4.74170628e+10j,
                   -3.05500910e+10 + 8.60872257e+10j,
                   -7.57094449e+08 + 9.52970974e+10j]

    assert_allclose(s_refined, s_reference, rtol=1e-5)

    if plot_figures:
        plt.figure(figsize=(6, 4))
        plt.plot(s_estimate.imag, s_estimate.real, 'x')
        plt.plot(s_refined.imag, s_refined.real, '+')
        plt.plot(integration_line_i, integration_line_r, 'r--')
        plt.show()


def test_srr_pair_separate_poles(plot_figures=False):
    "Cauchy integral for poles of an SRR pair considered as a single part"

    sim = openmodes.Simulation(basis_class=LoopStarBasis,
                               operator_class=EfieOperator)

    parameters = {'inner_radius': 2.5e-3,
                  'outer_radius': 4e-3}
    mesh = sim.load_mesh(osp.join(openmodes.geometry_dir, 'SRR.geo'),
                         parameters=parameters, mesh_tol=0.5e-3)
    srr1 = sim.place_part(mesh)
    srr2 = sim.place_part(mesh, location=[0, 0, 1e-3])

    s_min = -0.5e11-1e9j
    s_max = -2e6+1.2e11j
    integration_line_i = np.array((s_min.imag, s_max.imag, s_max.imag,
                                   s_min.imag, s_min.imag))
    integration_line_r = np.array((s_min.real, s_min.real, s_max.real,
                                   s_max.real, s_min.real))

    # calculate modes of the SRR pair
    result = sim.estimate_poles(s_min, s_max, parts=[srr1, srr2])
    refined = sim.refine_poles(result)

    s_estimate = result[srr1]['s']
    s_refined = refined[srr1]['s']

    s_reference = [-1.07078080e+09 + 4.44309178e+10j,
                   -2.46067038e+10 + 9.46785130e+10j]
    assert_allclose(s_refined, s_reference, rtol=1e-5)

    if plot_figures:
        plt.figure(figsize=(6, 4))
        plt.plot(s_estimate.imag, s_estimate.real, 'x')
        plt.plot(s_refined.imag, s_refined.real, '+')
        plt.plot(integration_line_i, integration_line_r, 'r--')
        plt.show()

if __name__ == "__main__":
    test_srr_pair_combined_poles(plot_figures=True)
    test_srr_pair_separate_poles(plot_figures=True)
