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
import scipy.linalg as la
from numpy.testing import assert_allclose

from openmodes.eig import eig_newton_bordered, eig_newton


def test_bordered(print_output=False):
    "Test the bordered iteration for the linear eigenvalue problem"

    np.random.seed(9322)
   
    size = 10    
    M = np.random.rand(size, size)

    # first test a symmetric matrix    
    M_sym = M + M.T

    w, vr = la.eig(M_sym)

    # Create an intial estimate by adding noise to the exact results
    w_0 = w[0]*1.2
    vr_0 = vr[:, 0].copy()
    vr_0 = vr_0*(1.3+0.4j) + np.random.rand(size)*0.2

    result = eig_newton_bordered(M_sym, w_0, vr_0)
    
    assert_allclose(w[0], result['w'])
    ratio = np.max(vr[:, 0])/np.max(result['vr'])
    assert_allclose(vr[:, 0], ratio*result['vr'])    
    
    print("Symmetric real matrix")
    print("Eigenvalue error:", abs((w[0]-result['w'])/w[0]))
    print("Right eigenvector relative value:", abs(vr[:, 0]/result['vr']))
    print()
   
    # Now solve asymmetric problem
    w, vl, vr = la.eig(M, left=True)
    
    # Create an intial estimate by adding noise to the exact results    
    w_0 = w[0]*1.2
    vr_0 = vr[:, 0].copy()
    vr_0 = vr_0*(1.3+0.4j) + np.random.rand(size)*0.2
    vl_0 = vl[:, 0].copy()

    result = eig_newton_bordered(M, w_0, vr_0, vl_0=vl_0)
    assert_allclose(w[0], result['w'])
    ratio = np.max(vr[:, 0])/np.max(result['vr'])
    assert_allclose(vr[:, 0], ratio*result['vr'])    
    ratio_l = np.max(vl[:, 0])/np.max(result['vl'])
    assert_allclose(vl[:, 0], ratio_l*result['vl'])    
    
    print("Asymmetric real matrix")
    print("Eigenvalue error:", abs((w[0]-result['w'])/w[0]))
    print("Right eigenvector relative value:", abs(vr[:, 0]/result['vr']))
    print("Left eigenvector relative value:", abs(vl[:, 0]/result['vl']))
    print()
    
    
    # Now solve for a complex matrix
    np.random.seed(412985)
    M_complex = np.random.rand(size, size)+1j*np.random.rand(size, size)
    
    w, vl, vr = la.eig(M_complex, left=True)
    vl = vl.conjugate()
    vl /= (np.diag(vl.T.dot(vr)))
    
    # verify the particular notation for left and right eigs
    M_recons = vr.dot(np.diag(w).dot(vl.T))
    # verify the accuracy of the reconstruction
    print("Asymmetric complex matrix")
    print("Error in matrix reconstruction",
          np.max(np.abs((M_complex-M_recons)/M_complex)))
    
    # Create an intial estimate by adding noise to the exact results
    w_0 = w[0]*1.2
    vr_0 = vr[:, 0].copy()
    vr_0 = vr_0*(1.3+0.4j) + np.random.rand(size)*0.2
    vl_0 = vl[:, 0].copy()
       
    result = eig_newton_bordered(M_complex, w_0, vr_0, vl_0=vl_0)
    assert_allclose(w[0], result['w'])
    ratio = np.max(vr[:, 0])/np.max(result['vr'])
    assert_allclose(vr[:, 0], ratio*result['vr'])    
    ratio_l = np.max(vl[:, 0])/np.max(result['vl'])
    assert_allclose(vl[:, 0], ratio_l*result['vl'])    
    
    print("Eigenvalue error:", abs((w[0]-result['w'])/w[0]))
    print("Right eigenvector relative value:", (vr[:, 0]/result['vr']))
    print("Left eigenvector relative value:", (vl[:, 0]/result['vl']))
    print("Normalisation of eigenvector:", np.dot(result['vr'], result['vl']))


def test_newton(print_output=False):
    "Test Newton iteration for the linear eigenvalue problem"

    np.random.seed(9322)

    size = 10
    M = np.random.rand(size, size)

    # first test a symmetric matrix
    M_sym = M + M.T

    w, vr = la.eig(M_sym)

    # Create an intial estimate by adding noise to the exact results
    w_0 = w[0]*1.2
    vr_0 = vr[:, 0].copy()
    vr_0 = vr_0*(1.3+0.4j) + np.random.rand(size)*0.2

    func = lambda x: M_sym - x*np.eye(size)
    result = eig_newton(func, w_0, vr_0, lambda_tol=1e-10)

    assert_allclose(w[0], result['eigval'])
    amax = np.argmax(vr[:, 0])
    ratio = vr[amax, 0]/result['eigvec'][amax]
    assert_allclose(vr[:, 0], ratio*result['eigvec'])
    assert_allclose(np.dot(result['eigvec'], result['eigvec_left']), -np.ones(1))

    print("Symmetric real matrix")
    print("Eigenvalue error:", abs((w[0]-result['eigval'])/w[0]))
    print("Right eigenvector relative value:", abs(vr[:, 0]/result['eigvec']))
    print("Normalisation of eigenvector:", np.dot(result['eigvec'], result['eigvec_left']))
    print()

    # Now solve asymmetric problem
    w, vl, vr = la.eig(M, left=True)

    # Create an intial estimate by adding noise to the exact results
    w_0 = w[0]*1.2
    vr_0 = vr[:, 0].copy()
    vr_0 = vr_0*(1.3+0.4j) + np.random.rand(size)*0.2
    vl_0 = vl[:, 0].copy()*(0.6+0.1j) + np.random.rand(size)*0.2

    func = lambda x: M - x*np.eye(size)
    result = eig_newton(func, w_0, vr_0, y_0=vl_0, lambda_tol=1e-10, weight='rayleigh asymmetric')
    assert_allclose(w[0], result['eigval'])
    amax = np.argmax(vr[:, 0])
    ratio = vr[amax, 0]/result['eigvec'][amax]
    assert_allclose(vr[:, 0], ratio*result['eigvec'])
    amax = np.argmax(vl[:, 0])
    ratio_l = vl[amax, 0]/result['eigvec_left'][amax]
    assert_allclose(vl[:, 0], ratio_l*result['eigvec_left'])
    assert_allclose(np.dot(result['eigvec'], result['eigvec_left']), -np.ones(1))

    print("Asymmetric real matrix")
    print("Eigenvalue error:", abs((w[0]-result['eigval'])/w[0]))
    print("Right eigenvector relative value:", abs(vr[:, 0]/result['eigvec']))
    print("Left eigenvector relative value:", abs(vl[:, 0]/result['eigvec_left']))
    print("Normalisation of eigenvector:", np.dot(result['eigvec'], result['eigvec_left']))
    print()

    # Now solve for a complex matrix
    np.random.seed(412985)
    M_complex = np.random.rand(size, size)+1j*np.random.rand(size, size)

    w, vl, vr = la.eig(M_complex, left=True)
    vl = vl.conjugate()
    vl /= (np.diag(vl.T.dot(vr)))

    # verify the particular notation for left and right eigs
    M_recons = vr.dot(np.diag(w).dot(vl.T))
    # verify the accuracy of the reconstruction
    print("Asymmetric complex matrix")
    print("Error in matrix reconstruction",
          np.max(np.abs((M_complex-M_recons)/M_complex)))

    # Create an intial estimate by adding noise to the exact results
    w_0 = w[0]*1.2
    vr_0 = vr[:, 0].copy()
    vr_0 = vr_0*(1.3+0.4j) + np.random.rand(size)*0.2
    vl_0 = vl[:, 0].copy()

    func = lambda x: M_complex - x*np.eye(size)
    result = eig_newton(func, w_0, vr_0, y_0=vl_0, weight='rayleigh asymmetric', lambda_tol=1e-10)
    assert_allclose(w[0], result['eigval'])
    amax = np.argmax(vr[:, 0])
    ratio = vr[amax, 0]/result['eigvec'][amax]
    assert_allclose(vr[:, 0], ratio*result['eigvec'])
    amax = np.argmax(vl[:, 0])
    ratio_l = vl[amax, 0]/result['eigvec_left'][amax]
    assert_allclose(vl[:, 0], ratio_l*result['eigvec_left'])
    assert_allclose(np.dot(result['eigvec'], result['eigvec_left']), -np.ones(1))

    print("Eigenvalue error:", abs((w[0]-result['eigval'])/w[0]))
    print("Right eigenvector relative value:", (vr[:, 0]/result['eigvec']))
    print("Left eigenvector relative value:", (vl[:, 0]/result['eigvec_left']))
    print("Normalisation of eigenvector:", np.dot(result['eigvec'], result['eigvec_left']))


if __name__ == "__main__":
    test_bordered()
    test_newton()
