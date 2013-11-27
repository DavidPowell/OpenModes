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
import itertools
from scipy.optimize import nnls

#from openmodes.constants import epsilon_0, mu_0    
#from openmodes.utils import SingularSparse
from openmodes import integration
from openmodes.parts import Part#, Triangles, RwgBasis

from openmodes.basis import DivRwgBasis, get_basis_functions
from openmodes.operator import EfieOperator, FreeSpaceGreensFunction
from openmodes.eig import eig_linearised, eig_newton


# TODO: ImpedanceMatrix may need to know about number of loops and stars?
class ImpedanceMatrix(object):
    """Holds an impedance matrices calculated at a specific frequency
    
    This is a single impedance matrix for the whole system
    """
    
    def __init__(self, s, L, S):
        self.s = s
        assert(L.shape == S.shape)
        self.L = L
        self.S = S

    def eigenmodes(self, num_modes):
        """Calculate the eigenimpedance and eigencurrents of each part's modes
        
        The modes with the smallest imaginary part of their impedance will be
        returned.
        
        Note that the impedance matrix is typically *ill-conditioned*.
        Therefore this routine can return junk results, particularly if the
        mesh is dense.
        """
        z_all, v_all = la.eig(self.s*self.L + self.S/self.s)
        which_z = np.argsort(abs(z_all.imag))[:num_modes]
        eigenimpedance = z_all[which_z]
        
        v = v_all[:, which_z]
        eigencurrent = v/np.sqrt(np.sum(v**2, axis=0))
        
        return eigenimpedance, eigencurrent

    def impedance_modes(self, num_modes, mode_currents_o, 
                        mode_currents_s = None, return_arrays = False):
        """Calculate a reduced impedance matrix based on the scalar impedance
        of the modes of each part, and the scalar coupling coefficients.

        Parameters
        ----------        
        s : number
            complex frequency at which to calculate impedance (in rad/s)
        num_modes : integer
            The number of modes to take into account for each part
        mode_currents_o : array
            The modal currents of the observer part
        mode_currents_s : array, optional
            The modal currents of the source part (only for off-diagonal terms
            where the source differs from the observer)
        return_arrays : boolean, optional
            Return the impedance arrays directly, instead of constructing an
            `ImpedanceMatrix` object
            
        Returns
        -------
        L_red, S_red : ndarray
            the reduced inductance and susceptance matrices
        """
 
        # Parts are already combined, so we are talking about modes of
        # the complete coupled system
        L_red = np.zeros((num_modes, num_modes), np.complex128)
        S_red = np.zeros_like(L_red)

        if mode_currents_s is None:
            for i in xrange(num_modes):
                # only diagonal terms are non-zero
                L_red[i, i] = mode_currents_o[:, i].dot(self.L.dot(mode_currents_o[:, i]))
                S_red[i, i] = mode_currents_o[:, i].dot(self.S.dot(mode_currents_o[:, i]))
        else:
            for i, j in itertools.product(xrange(num_modes), xrange(num_modes)):
                L_red[i, j] = mode_currents_o[:, i].dot(self.L.dot(mode_currents_s[:, j]))
                S_red[i, j] = mode_currents_o[:, i].dot(self.S.dot(mode_currents_s[:, j]))
                            
        if return_arrays:
            return L_red, S_red
        else:
            return ImpedanceMatrix(self.s, L_red, S_red)
            
    def source_modes(self, V, num_modes, mode_currents):
        "Take a source field, and project it onto the modes of the system"
        
        # calculate separately
        V_red = np.zeros(num_modes, np.complex128)
        for i in xrange(num_modes):
            V_red[i] = mode_currents[:, i].dot(V)

        return V_red

    @property
    def shape(self):
        return self.L.shape

class ImpedanceParts(object):
    """Holds a impedance matrices calculated at a specific frequency
    
    This consists of separate matrices for each part, and their mutual
    coupling terms.
    """
    
    def __init__(self, s, num_parts, matrices):
        """
        Parameters
        ----------        
        s : complex
            complex frequency at which to calculate impedance (in rad/s)
        matrices : list of list of ImpedanceMatrix
            The impedance matrix for each part, or mutual terms between them
        num_parts : int
            The number of parts in the system
        """
        self.s = s
        self.num_parts = num_parts
        self.matrices = matrices

    def combine_parts(self):
        """Evaluate the self and mutual impedances of all parts combined into
        a pair of matrices for the whole system.
        
        Returns
        -------
        impedance : CombinedImpedance
            An object containing the combined impedance matrices
        """
        
        total_size = sum(M[0].shape[0] for M in self.matrices)
        L_tot = np.empty((total_size, total_size), np.complex128)
        S_tot = np.empty_like(L_tot)

        row_offset = 0
        #for L_row, S_row in zip(self.L, self.S):
        for row in self.matrices:
            row_size = row[0].shape[0]
            col_offset = 0
            #for L, S in zip(L_row, S_row):
            for matrix in row:
                #L = matrix.L
                #S = matrix.S
                col_size = matrix.shape[1]
                L_tot[row_offset:row_offset+row_size, col_offset:col_offset+col_size] = matrix.L
                S_tot[row_offset:row_offset+row_size, col_offset:col_offset+col_size] = matrix.S
                col_offset += col_size
            row_offset += row_size
            
        return ImpedanceMatrix(self.s, L_tot, S_tot)

    def eigenmodes(self, num_modes):
        """Calculate the eigenimpedance and eigencurrents of each part's modes
        
        The modes with the smallest imaginary part of their impedance will be
        returned.
        
        Note that the impedance matrix is typically *ill-conditioned*.
        Therefore this routine can return junk results, particularly if the
        mesh is dense.
        """

        # TODO: cache this if parts are identical (should be upstream caching
        # of L and S for this to work)
        mode_impedances = []
        mode_currents = []
        for count in xrange(self.num_parts):
            eig_z, eig_current = self.matrices[count][count].eigenmodes(num_modes)

            mode_impedances.append(eig_z)
            mode_currents.append(eig_current)

        return mode_impedances, mode_currents

    def impedance_modes(self, num_modes, mode_currents, combine = True):
        """Calculate a reduced impedance matrix based on the scalar impedance
        of the modes of each part, and the scalar coupling coefficients.

        Parameters
        ----------        
        s : number
            complex frequency at which to calculate impedance (in rad/s)
        num_modes : integer
            The number of modes to take into account for each part
        mode_currents : list
            The modal currents of each part
            
        Returns
        -------
        L_red, S_red : ndarray
            the reduced inductance and susceptance matrices
        """
 
        # calculate modal impedances for each part separately, and include
        # coupling between all modes of different parts
        num_parts = self.num_parts        
        L_red = np.zeros((num_parts, num_modes, num_parts, num_modes), 
                            np.complex128)
        S_red = np.zeros_like(L_red)
        
#        for i, j, k, l in itertools.product(xrange(num_parts), xrange(num_modes),
#                                  xrange(num_parts), xrange(num_modes)):
#            if k != i or j == l:
#                # The mutual impedance terms of modes within the
#                # same resonator have L and S exactly cancelling,
#                # so currently they are not calculated
#                L_red[i, j, k, l] = mode_currents[i][:, j].dot(
#                                self.L[i][k].dot(mode_currents[k][:, l]))
#                S_red[i, j, k, l] = mode_currents[i][:, j].dot(
#                                self.S[i][k].dot(mode_currents[k][:, l]))

        for i, j in itertools.product(xrange(num_parts), xrange(num_parts)):
            # The mutual impedance terms of modes within the
            # same resonator have L and S exactly cancelling,
            # so currently they are not calculated
            M = self.matrices[i][j]

            L, S = M.impedance_modes(num_modes, mode_currents[i],
                                        mode_currents[j], return_arrays=True)
            L_red[i, :, j, :] = L
            L_red[i, :, j, :] = S
            
        if combine:           
            L_red = L_red.reshape((num_parts*num_modes, num_parts*num_modes))
            S_red = S_red.reshape((num_parts*num_modes, num_parts*num_modes))
            return ImpedanceMatrix(self.s, L_red, S_red)
        else:
            raise NotImplementedError
        
    def source_modes(self, V, num_modes, mode_currents, combine = True):
        "Take a source field, and project it onto the modes of each part"
        
        V_red = np.zeros((self.num_parts, num_modes), np.complex128)
        
        for i in xrange(self.num_parts):
            V_red[i] = self.matrices[i][i].source_modes(V[i], num_modes,
                mode_currents[i])

        if combine:
            V_red = V_red.reshape(self.num_parts*num_modes)

        return V_red

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

            