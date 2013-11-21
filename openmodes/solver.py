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

from __future__ import division, print_function

# numpy and scipy
import numpy as np
import numpy.linalg as la

#from openmodes.constants import epsilon_0, mu_0    
#from openmodes.utils import SingularSparse
from openmodes import integration
from openmodes.parts import Part#, Triangles, RwgBasis

from openmodes.basis import DivRwgBasis, get_basis_functions
from openmodes.operator import EfieOperator, FreeSpaceGreensFunction
from openmodes.eig import linearised_eig

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
        mesh : LibraryPart
            The part to place
        location : array, optional
            If specified, place the part at a given location, otherwise it will
            be placed at the origin
            
        Returns
        -------
        sim_part : SimulationPart
            The part placed in the simulation
            
        The part will be placed at the origin. It can be translated, rotated
        etc using the relevant methods of `SimulationPart`            
            
        Currently the part can only be modelled as a perfect electric
        conductor
        """
        
        sim_part = Part(mesh, location=location) 
        self.parts.append(sim_part)

        return sim_part
    
    def impedance_matrix(self, s, serial_interpolation = False, 
                         loop_star = True):
        """Evaluate the impedances matrices
        
        Parameters
        ----------        
        s : number
            complex frequency at which to calculate impedance (in rad/s)
        serial_interpolation : boolean, optional
            do interpolation of Green's function serially
            which is slower, but easier to debug
        loop_star : boolean, optional
            transform impedance from RWG to loop-star basis
        
        This version assumes that the mesh is associated with each object,
        which is then used to build a single global mesh        
        
        Returns
        -------
        L, S : ndarray
            the inductance and susceptance matrices
         
        """
        return self.operator.impedance_matrix(s, self.parts[0])

          
    def source_term(self, e_inc, jk_inc):
        """Evaluate the source vector due to the incident wave
        
        Parameters
        ----------        
        e_inc: ndarray
            incident field polarisation in free space
        jk_inc: ndarray
            incident wave vector in free space
            
        Returns
        -------
        V : ndarray
            the source "voltage" vector
        """

        return self.operator.plane_wave_source(self.parts[0], e_inc, jk_inc)

    def linearised_eig(self, part, L, S, num_modes):
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
        basis = get_basis_functions(part.mesh, self.basis_class)
        return linearised_eig(L, S, num_modes, basis)
        
    #def write_vtk(self, file_name
