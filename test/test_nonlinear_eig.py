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

from __future__ import print_function

import numpy as np
from numpy.testing import assert_allclose
import scipy.linalg as la

from openmodes.eig import eig_newton, poles_cauchy
from openmodes.integration import RectangularContour

def test_nonlinear_eig():
    """Test routines for nonlinear eigenvalues problems
    by giving them linear eigenvalue problems"""

    # Construct matrix from eigenform
    exact_s = np.array([-1, -0.5+3.3j, 0.5+7j, 4.6j])
    exact_vr = np.array([[1, 0.7-.4j, 4, 2],
                         [6, -1, -7j, 0.5j],
                         [-2+1j, 3j, 0.5, 1],
                         [0.87, 4.2j, -5+1j, -6]]).T
    exact_vl = la.inv(exact_vr)
    full_matrix = np.dot(exact_vr, np.dot(np.diag(exact_s), exact_vl))
    exact_order = np.argsort(exact_s)

    # Compare with direct eigenvalue decomposition
    eig_s, eig_vl, eig_vr = la.eig(full_matrix, left=True)
    eig_order = np.argsort(eig_s)

    for exact_n, eig_n in zip(exact_order, eig_order):
        assert(np.abs((exact_s[exact_n]-eig_s[eig_n])/exact_s[exact_n]) < 1e-10)

        exact_vr_n = exact_vr[:, exact_n]
        eig_vr_n = eig_vr[:, eig_n]
        vr_ratio = np.abs(exact_vr_n/eig_vr_n)
        assert_allclose(vr_ratio, np.average(vr_ratio))

        exact_vl_n = exact_vl[exact_n, :]
        eig_vl_n = eig_vl[:, eig_n]
        vl_ratio = np.abs(exact_vl_n/eig_vl_n)
        assert_allclose(vl_ratio, np.average(vl_ratio))

    # Perform Cauchy line integral to estimate eigendecomposition
    def Z_func(s):
        return np.dot(exact_vr, np.dot(np.diag(s - exact_s), exact_vl))

    contour = RectangularContour(-3-1j, 7+9j)
    estimates = poles_cauchy(Z_func, contour)

    estimates_order = np.argsort(estimates['s'])

    for exact_n, estimate_n in zip(exact_order, estimates_order):
        assert(np.abs((exact_s[exact_n]-estimates['s'][estimate_n])/exact_s[exact_n]) < 1e-10)

        exact_vr_n = exact_vr[:, exact_n]
        estimate_vr_n = estimates['vr'][:, estimate_n]
        vr_ratio = np.abs(exact_vr_n/estimate_vr_n)
        assert_allclose(vr_ratio, np.average(vr_ratio))

        exact_vl_n = exact_vl[exact_n, :]
        estimate_vl_n = estimates['vl'][estimate_n, :]
        vl_ratio = np.abs(exact_vl_n/estimate_vl_n)

        assert_allclose(vl_ratio, np.average(vl_ratio))

    #     print(np.abs(exact_vr[:, exact_n]/estimates['vr'][:, estimate_n]))

    for estimate_n in estimates_order:
        magnitude = np.dot(estimates['vr'][:, estimate_n], estimates['vl'][estimate_n, :])
        estimates['vr'][:, estimate_n] /= magnitude

    estimate_matrix = estimates['vr'].dot(np.diag(estimates['s']).dot(estimates['vl']))
    assert(np.all(np.abs((estimate_matrix-full_matrix)/full_matrix) < 1e-10))

if __name__ == "__main__":
    test_nonlinear_eig()
