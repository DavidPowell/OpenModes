# -*- coding: utf-8 -*-
"""
OpenModes - An eigenmode solver for open electromagnetic resonantors
Copyright (C) 2013 David Powell

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

from __future__ import division#, print_function

# numpy and scipy
import numpy as np
import scipy.linalg as la
#import itertools
from scipy.optimize import nnls

#from openmodes.constants import epsilon_0, mu_0    
#from openmodes.utils import SingularSparse
from openmodes import integration
from openmodes.parts import Part#, Triangles, RwgBasis

from openmodes.impedance import ImpedanceMatrix, ImpedanceParts
from openmodes.basis import DivRwgBasis, get_basis_functions
from openmodes.operator import EfieOperator, FreeSpaceGreensFunction
from openmodes.eig import eig_linearised, eig_newton


def delta_eig(s, j, part, Z_func, eps = None):
    """Find the derivative of the eigenimpedance at the resonant frequency
    
    See section 5.7 of numerical recipes for calculating the step size h

    Impedance derivative is based on
    C. E. Baum, Proceedings of the IEEE 64, 1598 (1976).
    """

    if eps is None:
        # find the machine precision (this should actually be the accuracy with which Z is calculated)
        eps = np.finfo(s.dtype).eps
    
    # first determine the optimal value of h
    h = abs(s)*eps**(1.0/3.0)*(1.0 + 1.0j)
    
    # make h exactly representable in floating point
    temp = s + h
    h = (temp - s)

    delta_Z = (Z_func(s+h) - Z_func(s-h))/(2*h)
    
    return np.dot(j.T, np.dot(delta_Z, j))

def fit_circuit(s_0, z_der):
    """
    Fit a circuit model to a resonant frequency and impedance derivative
    To get reasonable condition number, omega_0 should be scaled to be near unity,
    and z_der should be scaled by the inverse of this factor
    """
    M = np.zeros((4, 4), np.float64)
    rhs = np.zeros(4, np.float64)
    
    # order of coefficients is C, R, L, R2
    
    # fit impedance being zero at resonance
    eq1 = np.array([1/s_0, 1, s_0, -s_0**2]) # XXX: minus or not????
    M[0, :] = eq1.real
    M[1, :] = eq1.imag
    
    # fit impedance derivative at resonance
    eq2 = np.array([1/s_0**2, 0, 1, -2*s_0])
    M[2, :] = eq2.real
    M[3, :] = eq2.imag
    
    rhs[2] = z_der.real
    rhs[3] = z_der.imag
    
    return nnls(M, rhs)[0]
    

class Simulation(object):
    """This object controls everything within the simluation. It contains all
    the parts which have been placed, and the operator equation which is
    used to solve the scattering problem.
    """

    def __init__(self, integration_rule = 5, basis_class = DivRwgBasis,
                 operator_class = EfieOperator, 
                 greens_function=FreeSpaceGreensFunction()):
        """       
        Parameters
        ----------
        integration_rule : integer
            the order of the integration rule on triangles
        """
        
        self.quadrature_rule = integration.get_dunavant_rule(integration_rule)
        
        self.triangle_quadrature = {}
        self.singular_integrals = {}
               
        self.parts = []
        
        self.basis_class = basis_class
        self.operator = operator_class(quadrature_rule=self.quadrature_rule,
                                       basis_class=basis_class, 
                                       greens_function=greens_function)

    def place_part(self, mesh, location=None):
        """Add a part to the simulation domain
        
        Parameters
        ----------
        mesh : an appropriate mesh object
            The part to place
        location : array, optional
            If specified, place the part at a given location, otherwise it will
            be placed at the origin
            
        Returns
        -------
        part : Part
            The part placed in the simulation
            
        The part will be placed at the origin. It can be translated, rotated
        etc using the relevant methods of `Part`            
        """
        
        sim_part = Part(mesh, location=location) 
        self.parts.append(sim_part)

        return sim_part

    def calculate_impedance(self, s):
        """Evaluate the self and mutual impedances of all parts  in the
        simulation. Return an impedance object which can calculate several
        derived impedance quantities
        
        Parameters
        ----------        
        s : number
            complex frequency at which to calculate impedance (in rad/s)
        
        Returns
        -------
        impedance_matrix : ImpedanceMatrix
            The impedance matrix object which can represent the impedance of
            the object in several ways.
        """

        #return ImpedanceMatrix(s, self.operator, self.parts)
        
        #S_parts = []
        #L_parts = []
        matrices = []
        
        # TODO: cache individual part impedances to avoid repetition
        #parts_calculated = {}

        for index_a, part_a in enumerate(self.parts):
            #S_parts.append([])
            #L_parts.append([])
            matrices.append([])
            for index_b, part_b in enumerate(self.parts):
                if (index_b < index_a) and self.operator.reciprocal:
                    # use reciprocity to avoid repeated calculation
                    S = matrices[index_b][index_a].S.T
                    L = matrices[index_b][index_a].L.T
                else:
                    L, S = self.operator.impedance_matrix(s, part_a, part_b)
                matrices[-1].append(ImpedanceMatrix(L, S))
                #S_parts[-1].append(S)
                #L_parts[-1].append(L)
        
        return ImpedanceParts(s, len(self.parts), matrices)
    
            
    def source_plane_wave(self, e_inc, jk_inc):
        """Evaluate the source vectors due to an incident plane wave, returning
        separate vectors for each part.
        
        Parameters
        ----------        
        e_inc: ndarray
            incident field polarisation in free space
        jk_inc: ndarray
            incident wave vector in free space
            
        Returns
        -------
        V : list of ndarray
            the source vector for each part
        """
        return [self.operator.source_plane_wave(part, e_inc, jk_inc) for part 
                in self.parts]
                
    def part_singularities(self, s_start, num_modes):
        """Find the singularities of the system in the complex frequency plane

        Parameters
        ----------        
        s_start : number
            The complex frequency at which to perform the estimate. Should be
            within the band of interest

        """
        
        all_s = []
        all_j = []        
        
        for part in self.parts:
            basis = get_basis_functions(part.mesh, self.basis_class)
            L, S = self.operator.impedance_matrix(s_start, part)
            lin_s, lin_currents = eig_linearised(L, S, num_modes, basis)
            #print lin_s/2/np.pi

            mode_s = np.empty(num_modes, np.complex128)
            mode_j = np.empty((len(basis), num_modes), np.complex128)
        
            Z_func = lambda s: self.operator.impedance_matrix(s, part, combine=True)
        
            for mode in xrange(num_modes):
                res = eig_newton(Z_func, lin_s[mode], lin_currents[:, mode], 
                                 weight='max element', lambda_tol=1e-8, max_iter=200)
                                 
                print "Iterations", res['iter_count']
                #print res['eigval']/2/np.pi
                mode_s[mode] = res['eigval']
                j_calc = res['eigvec']
                mode_j[:, mode] = j_calc/np.sqrt(np.sum(j_calc**2))
                
            all_s.append(mode_s)
            all_j.append(mode_j)

#            all_s.append(lin_s)
#            all_j.append(lin_currents)

        return all_s, all_j
        
#    def circuit_models(self):
#        """
#        """
#        
#
#eig_derivs = []
#
#for part in xrange(n_parts):
#    eig_derivs.append(np.empty(n_modes, np.complex128))
#    for mode in xrange(n_modes):
#        eig_derivs[part][mode] = delta_eig(mode_omega[part][mode], mode_j[part][:, mode], part, loop_star=loop_star)

            