# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 15:50:37 2015

@author: dap124
"""

from __future__ import print_function

import os.path as osp

import numpy as np

import matplotlib.pyplot as plt

import openmodes
from openmodes.basis import LoopStarBasis
from openmodes.operator import EfieOperator
from openmodes.integration import ExternalModeContour
from openmodes.mesh import gmsh

import helpers

import logging
logging.getLogger().setLevel(logging.INFO)

tests_filename = __file__
input_dir = helpers.get_input_dir(tests_filename)
meshfile = osp.join(input_dir, 'srr.msh')

parameters = {'inner_radius': 2.5e-3,
              'outer_radius': 4e-3}


def generate_mesh():
    "Generate a fixed mesh file to ensure consistent results of tests"
    meshed_name = gmsh.mesh_geometry(osp.join(openmodes.geometry_dir, 'srr.geo'),
                                     input_dir, parameters=parameters)
    assert(meshed_name == meshfile)



def srr_pair_combined_poles(plot=False):
    "Cauchy integral for poles of an SRR pair considered as a single part"

    sim = openmodes.Simulation(basis_class=LoopStarBasis,
                               operator_class=EfieOperator)

    mesh = sim.load_mesh(meshfile)
    srr1 = sim.place_part(mesh)
    srr2 = sim.place_part(mesh, location=[0, 0, 1e-3])

    # calculate modes of the SRR pair
    contour = ExternalModeContour(-0.5e11+1.2e11j, overlap_axes=0.2e6)
    result = sim.estimate_poles(contour)
    refined = sim.refine_poles(result)

    s_estimate = result.s
    s_refined = refined.s

    if plot:
        plt.figure(figsize=(6, 4))
        plt.plot(s_estimate.imag, s_estimate.real, 'x')
        plt.plot(s_refined.imag, s_refined.real, '+')
        contour_points = np.array([s for s, w in contour])
        plt.plot(contour_points.imag, contour_points.real)
        plt.show()

    return {'name': 'srr_pair_combined_poles',
            'results' : {'s': s_refined.simple_view()},
            'rtol': {'s': 1e-5}}


def test_srr_pair_combined_poles():
    helpers.run_test(srr_pair_combined_poles, tests_filename)
test_srr_pair_combined_poles.__doc__ = srr_pair_combined_poles.__doc__


def srr_pair_separate_poles(plot=False):
    "Cauchy integral for poles of an SRR pair considered as a single part"

    sim = openmodes.Simulation(basis_class=LoopStarBasis,
                               operator_class=EfieOperator)

    mesh = sim.load_mesh(meshfile)
    srr1 = sim.place_part(mesh)
    srr2 = sim.place_part(mesh, location=[0, 0, 1e-3])

    # calculate modes of each SRR
    contour = ExternalModeContour(-0.5e11+1.2e11j, overlap_axes=0.2e9)
    estimate = sim.estimate_poles(contour, parts=[srr1, srr2])
    refined = sim.refine_poles(estimate)

    s_estimate = estimate[srr1].s
    s_refined = refined[srr1].s

    if plot:
        plt.figure(figsize=(6, 4))
        plt.plot(s_estimate.imag, s_estimate.real, 'x')
        plt.plot()
        plt.plot(s_refined.imag, s_refined.real, '+')
        contour_points = np.array([s for s, w in contour])
        plt.plot(contour_points.imag, contour_points.real)
        plt.show()
        
    return {'name': 'srr_pair_separate_poles',
            'results' : {'s_srr1': s_refined[0].simple_view()},
            'rtol': {'s_srr1': 1e-5}}

def test_srr_pair_separate_poles():
    helpers.run_test(srr_pair_separate_poles, tests_filename)
test_srr_pair_separate_poles.__doc__ = srr_pair_separate_poles.__doc__


if __name__ == "__main__":
    # Uncomment the following lines to update reference solutions
    # generate_mesh()
    # helpers.create_reference(srr_pair_combined_poles, tests_filename)
    # helpers.create_reference(srr_pair_separate_poles, tests_filename)

    srr_pair_combined_poles(plot=True)
    srr_pair_separate_poles(plot=True)
