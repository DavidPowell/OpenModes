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


import numpy as np
import matplotlib.pyplot as plt
import os.path as osp

import openmodes
from openmodes.basis import DivRwgBasis, LoopStarBasis
from openmodes.integration import DunavantRule
from openmodes import Simulation
from openmodes.visualise import write_vtk

from numpy.testing import assert_allclose

tests_location = osp.split(__file__)[0]


def test_interpolate_rwg(plot=False):
    "Interpolate an RWG basis function over a triangle"

    sim = Simulation()
    mesh_tol = 0.5e-3
    srr = sim.load_mesh(osp.join(tests_location, 'input', 'test_basis',
                                 'SRR.msh'))

    basis = DivRwgBasis(srr)

    rwg_function = np.zeros(len(basis), np.float64)
    rwg_function[20] = 1

    rule = DunavantRule(10)

    r, basis_func = basis.interpolate_function(rwg_function, rule)

    # save reference data
    # np.savetxt(osp.join(tests_location, 'reference', 'test_basis',
    #                     'rwg_r.txt'), r, fmt="%.8e")
    # np.savetxt(osp.join(tests_location, 'reference', 'test_basis',
    #                    'rwg_basis_func.txt'), basis_func, fmt="%.8e")

    r_ref = np.loadtxt(osp.join(tests_location, 'reference',
                                'test_basis', 'rwg_r.txt'))
    basis_func_ref = np.loadtxt(osp.join(tests_location, 'reference',
                                         'test_basis', 'rwg_basis_func.txt'))
    assert_allclose(r, r_ref)
    assert_allclose(basis_func, basis_func_ref)

    if plot:
        plt.figure(figsize=(6, 6))
        plt.quiver(r[:, 0], r[:, 1], basis_func[:, 0], basis_func[:, 1],
                   scale=5e4)
        plt.show()


def test_interpolate_loop_star(plot=False):
    "Interpolate loop and star basis functions"

    sim = Simulation()
    mesh_tol = 4e-3
    mesh = sim.load_mesh(osp.join(tests_location, 'input', 'test_basis',
                                  'rectangle.msh'))

    basis = LoopStarBasis(mesh)

    ls_function = np.zeros(len(basis), np.float64)
    # chose one loop and one star
    star_basis = 22
    loop_basis = 4

    ls_function[star_basis] = 1
    ls_function[loop_basis] = 1

    rule = DunavantRule(10)
    r, basis_func = basis.interpolate_function(ls_function, rule)

    the_basis = basis[star_basis]

    plus_nodes = mesh.nodes[basis.mesh.polygons[the_basis.tri_p,
                                                the_basis.node_p]]
    minus_nodes = mesh.nodes[basis.mesh.polygons[the_basis.tri_m,
                                                 the_basis.node_m]]

    # save reference data
    # np.savetxt(osp.join(tests_location, 'reference', 'test_basis',
    #                     'loop_star_r.txt'), r, fmt="%.8e")
    # np.savetxt(osp.join(tests_location, 'reference', 'test_basis',
    #                     'loop_star_basis_func.txt'), basis_func, fmt="%.8e")

    r_ref = np.loadtxt(osp.join(tests_location, 'reference',
                                'test_basis', 'loop_star_r.txt'))
    basis_func_ref = np.loadtxt(osp.join(tests_location, 'reference',
                                         'test_basis',
                                         'loop_star_basis_func.txt'))
    assert_allclose(r, r_ref)
    assert_allclose(basis_func, basis_func_ref)

    if plot:
        plt.figure(figsize=(6, 6))
        plt.quiver(r[:, 0], r[:, 1], basis_func[:, 0], basis_func[:, 1],
                   pivot='middle')
        plt.plot(plus_nodes[:, 0], plus_nodes[:, 1], 'x')
        plt.plot(minus_nodes[:, 0], minus_nodes[:, 1], '+')
        plt.show()


if __name__ == "__main__":
    test_interpolate_rwg(plot=True)
    test_interpolate_loop_star(plot=True)
