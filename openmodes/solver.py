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

#from openmodes.constants import epsilon_0, mu_0    
#from openmodes.utils import SingularSparse
from openmodes import integration
from openmodes.parts import Part#, Triangles, RwgBasis

from openmodes.basis import DivRwgBasis, get_basis_functions
from openmodes.operator import EfieOperator, FreeSpaceGreensFunction
from openmodes.eig import eig_linearised, eig_newton

# SimulationResult?? (also contains field?)
class ImpedanceMatrix(object):
    """Holds all the different forms of the impedance matrices for a set of
    parts, calculated at a specific frequency"""
    
    def __init__(self, s, operator, parts):
        self.s = s
        self.operator = operator
        
        S_parts = []
        L_parts = []
        
        # TODO: cache individual part impedances to avoid repetition
        #parts_calculated = {}

        for index_a, part_a in enumerate(parts):
            S_parts.append([])
            L_parts.append([])
            for index_b, part_b in enumerate(parts):
                if index_b < index_a:
                    # use reciprocity to avoid repeated calculation
                    S = S_parts[index_b][index_a].T
                    L = L_parts[index_b][index_a].T
                else:
                    L, S = self.operator.impedance_matrix(self.s, part_a, part_b)
                S_parts[-1].append(S)
                L_parts[-1].append(L)

        self.L_parts = L_parts
        self.S_parts = S_parts

    def impedance_combined(self):
        """Evaluate the self and mutual impedances of all parts combined into
        a pair of matrices for the whole system.
        
        Parameters
        ----------        
        s : number
            complex frequency at which to calculate impedance (in rad/s)
        
        Returns
        -------
        L, S : ndarray
            the combined inductance and susceptance matrices
        """
        
        total_size = sum(L[0].shape[0] for L in self.L_parts)
        L_tot = np.empty((total_size, total_size), np.complex128)
        S_tot = np.empty_like(L_tot)
        L_tot[:] = np.nan
        S_tot[:] = np.nan

        row_offset = 0
        for L_row, S_row in zip(self.L_parts, self.S_parts):
            row_size = L_row[0].shape[0]
            col_offset = 0
            for L, S in zip(L_row, S_row):
                col_size = L.shape[1]
                L_tot[row_offset:row_offset+row_size, col_offset:col_offset+col_size] = L
                S_tot[row_offset:row_offset+row_size, col_offset:col_offset+col_size] = S
                col_offset += col_size
            row_offset += row_size
            
        return L_tot, S_tot

    def eig_linearised(self, num_modes = 1):
        """Solves a linearised approximation to the eigenvalue problem from
        the impedance calculated at some fixed frequency.
        
        Parameters
        ----------
        L, S : ndarray
            The two components of the impedance matrix. They *must* be
            calculated in the loop-star basis.
        n_modes : int
            The number of modes required.
        which_obj : int, optional
            Which object in the system to find modes for. If not specified, 
            then modes of the entire system will be found
            
        Returns
        -------
        omega_mode : ndarray, complex
            The resonant frequencies of the modes
        j_mode : ndarray, complex
            Columns of this matrix contain the corresponding modal currents
        """
        #basis = get_basis_functions(part.mesh, self.basis_class)
        #return eig_linearised(L, S, num_modes, basis)

#        if L_parts is None or S_parts is None:
#            L_parts, S_parts = self.impedance_parts(s)

        part_impedances = []
        part_currents = []        
        
        for i, part in enumerate(self.parts):
            basis = get_basis_functions(part.mesh, self.basis_class)
            mode_impedances, mode_currents = eig_linearised(self.L_parts[i][i], 
                                        self.S_parts[i][i], num_modes, basis)
            part_impedances.append(mode_impedances)
            part_currents.append(mode_currents)

        self.part_impedances = part_impedances
        self.part_currents = part_currents

        return part_impedances, part_currents

    def calculate_eigenmodes(self, num_modes = 1):
        """Calculate the eigenimpedance and eigencurrents of each part's modes
        
        The modes with the smallest imaginary part of their impedance will be
        returned.
        
        Note that the impedance matrix is typically *ill-conditioned*.
        Therefore this routine can return junk results, particularly if the
        mesh is dense.
        """
        
        part_impedances = []
        part_currents = []
        for count in xrange(len(self.L_parts)):
            L = self.L_parts[count][count]
            S = self.S_parts[count][count]
            Z = self.s*L + S/self.s

            z_all, v_all = la.eig(Z)
            which_z = np.argsort(abs(z_all.imag))[:num_modes]
            part_impedances.append(z_all[which_z])
            
            v = v_all[:, which_z]
            part_currents.append(v/np.sqrt(np.sum(v**2, axis=0)))

        self.part_impedances = part_impedances
        self.part_currents = part_currents
        return part_impedances, part_currents

    def impedance_reduced(self, num_modes = 0, flattened = True):
        """Calculate a reduced impedance matrix based on the scalar impedance
        of the modes of each part, and the scalar coupling coefficients.

        Parameters
        ----------        
        s : number
            complex frequency at which to calculate impedance (in rad/s)
        num_modes : integer, optional
            The number of modes to take into account for each part
            
        Returns
        -------
        L_red, S_red : ndarray
            the reduced inductance and susceptance matrices
        """
 
        if not (hasattr(self, "part_currents")
            and len(self.part_impedances[0]) >= num_modes):
            # impedances have not been calculated or not enough modes
            raise ValueError("Modes must be calculated in order to find the reduced impedance")

        num_modes = num_modes or len(self.part_impedances[0])
            
        #part_impedances = self.part_impedances
        part_currents = self.part_currents
        
        num_parts = len(self.L_parts)                                                             
        L_red = np.zeros((num_parts, num_modes, num_parts, num_modes), 
                            np.complex128)
        S_red = np.zeros_like(L_red)
        
        for i, j, k, l in itertools.product(xrange(num_parts), xrange(num_modes),
                                  xrange(num_parts), xrange(num_modes)):
            if k != i or j == l:
                # The mutual impedance terms of modes within the
                # same resonator have L and S exactly cancelling,
                # but they can be calculated anyway
                L_red[i, j, k, l] = part_currents[i][:, j].dot(
                                self.L_parts[i][k].dot(part_currents[k][:, l]))
                S_red[i, j, k, l] = part_currents[i][:, j].dot(
                                self.S_parts[i][k].dot(part_currents[k][:, l]))
                                     
        if flattened:           
            L_red = L_red.reshape((num_parts*num_modes, num_parts*num_modes))
            S_red = S_red.reshape((num_parts*num_modes, num_parts*num_modes))

        self.L_red = L_red
        self.S_red = S_red
        
        return L_red, S_red
        
    def source_reduced(self, V, num_modes = 0, flattened = True):
        if not (hasattr(self, "part_currents")
            and len(self.part_impedances[0]) >= num_modes):
            # impedances have not been calculated or not enough modes
            raise ValueError("Modes must be calculated in order to find the reduced source term")

        num_modes = num_modes or len(self.part_impedances[0])

        part_currents = self.part_currents
        
        num_parts = len(self.L_parts)                                                             
        V_red = np.zeros((num_parts, num_modes), np.complex128)
        
        for i, j in itertools.product(xrange(num_parts), xrange(num_modes)):
            V_red[i, j] = part_currents[i][:, j].dot(V[i])

        if flattened:
            V_red = V_red.reshape(num_parts*num_modes)

        return V_red
        
    def __getattr__(self, name):
        """Implements a caching mechanism to store impedance data which will
        need to be reused"""
        
        #if name == "L_parts":
        #    return self.impedance_parts[0]
        #elif name == "S_parts":
        #    return self.impedance_parts[1]
        #if name == "L_tot":
        #    return self.impedance_combined[0]
        #elif name == "S_tot":
        #    return self.impedance_combined[1]
        if name == "L_red":
            return self.impedance_reduced[0]
        elif name == "S_red":
            return self.impedance_reduced[1]
        else:
            raise AttributeError

    

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
        return ImpedanceMatrix(s, self.operator, self.parts)
    
            
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