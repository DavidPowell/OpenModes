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

from openmodes.constants import epsilon_0, mu_0    
from openmodes.utils import SingularSparse
from openmodes import integration
from openmodes.parts import SimulationPart#, Triangles, RwgBasis

from openmodes.basis import DivRwgBasis
from openmodes.operator import EfieOperator, FreeSpaceGreensFunction

# compiled fortran libraries
#import core_for
 
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
               
        self._parts = []
        
        self.basis_class = basis_class
        self.operator = operator_class(quadrature_rule=self.quadrature_rule,
                                       basis_class=basis_class, 
                                       greens_function=greens_function)

    def place_part(self, library_part, location=None):
        """Add a part to the simulation domain
        
        Parameters
        ----------
        library_part : LibraryPart
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
        
        sim_part = SimulationPart(library_part, self.parts_modified, 
                                  location=location)
        self._parts.append(sim_part)

        #sim_part.add_observer(self.objects_changed)
        #self.objects_changed()
        #self.parts_modified()
        return sim_part
    
    @property
    def parts(self):
        return self._parts
    
    def parts_modified(self):
        """Called when any parts have been modified, invalidating the combined 
        mesh and precalculated impedance terms"""
        
        if hasattr(self, "combined_mesh"):
            del self._combined_mesh
            
        if hasattr(self, "combined_precalc"):
            del self._combined_precalc
        
  
    def object_basis(self, loop_star = True):
        """Find the ranges of the impedance matrix and source vectors 
        corresponding to each MOM_object which has been added
        
        Parameters
        ----------
        loop_star : bool
            Whether or not to calculate in loop-star basis
        
        Returns
        -------
        slices : list of `slice`
            each slice object gives the relevant part of the matrix/vector
        """
        
        if loop_star:
            return ([slice(a, b) for a, b in self.combined_mesh['loop_ranges']],
                    [slice(a, b) for a, b in self.combined_mesh['star_ranges']])
                    
        else:
            return [slice(a, b) for a, b in self.combined_mesh['objs'].basis]


    def object_triangles(self):
        """Find the ranges of triangle faces
        corresponding to each MOM_object which has been added
        
        Returns
        -------
        slices : list of `slice`
            each slice object gives the relevant part of the matrix/vector
            for each object
        """
        
        sizes = [0]+[len(obj.tri) for obj in self._parts]
        sizes = np.cumsum(sizes)
        return [slice(sizes[i], sizes[i+1]) for i in xrange(len(sizes)-1)]
        
    @property
    def combined_mesh(self):
        """Combine the meshes of all objects into a global mesh

        If the combined mesh is invalid then recombine it
        
        Returns
        -------
        nodes : ndarray
        triangles : Triangles
        basis : RwgBasis
        objs
        max_distance
        shortest_edge : number
            the shortest of all inter-node distances
        
        """
        try:
            return self._combined_mesh
        except AttributeError:
            N_objs = len(self._parts)
            objs = np.rec.recarray(N_objs, dtype=[("nodes", np.int32, 2), 
                                                  ("triangles", np.int32, 2), 
                                                  ("basis", np.int32, 2)])
        
            # calculate the total number of nodes and triangles    
            N_nodes = sum(len(obj.nodes) for obj in self._parts)
            N_tri = sum(len(obj.tri) for obj in self._parts)
            N_basis = sum(len(obj.basis) for obj in self._parts)
        
            nodes = np.empty((N_nodes, 3), np.float64, order='F')
            tri = Triangles(N_tri)                                  
            basis = RwgBasis(N_basis)
        
            # merge all the nodes, triangles and basis functions into a
            # master array
            offset_nodes = 0
            offset_tri = 0
            offset_basis = 0
            
            shortest_edge = np.inf
        
            loop_basis = []
            star_basis = []
        
            for obj_count, obj in enumerate(self._parts):
        
                nodes[offset_nodes:offset_nodes+len(obj.nodes)] = obj.nodes
        
                # renumber triangle nodes to global indices
                tri_range = slice(offset_tri, offset_tri+len(obj.tri))
                tri.nodes[tri_range] = obj.tri.nodes+offset_nodes
                tri.area[tri_range] = obj.tri.area
                tri.lens[tri_range] = obj.tri.lens
                #tri.mid[tri_range] = obj.tri.mid
        
                # renumber the basis functions
                basis_range = slice(offset_basis, offset_basis+len(obj.basis))
                basis.tri_p[basis_range] = obj.basis.tri_p+offset_tri
                basis.tri_m[basis_range] = obj.basis.tri_m+offset_tri
                basis.node_p[basis_range] = obj.basis.node_p
                basis.node_m[basis_range] = obj.basis.node_m
        
                basis.rho_cp[basis_range] = obj.basis.rho_cp
                basis.rho_cm[basis_range] = obj.basis.rho_cm
                basis.len[basis_range] = obj.basis.len
        
                # record the ranges of each array corresponding to each object            
                objs[obj_count].nodes[0] = offset_nodes
                objs[obj_count].nodes[1] = offset_nodes+len(obj.nodes)
                objs[obj_count].triangles[0] = offset_tri
                objs[obj_count].triangles[1] = offset_tri+len(obj.tri)
                objs[obj_count].basis[0] = offset_basis
                objs[obj_count].basis[1] = offset_basis+len(obj.basis)
        
                offset_nodes += len(obj.nodes)
                offset_tri += len(obj.tri)
                offset_basis += len(obj.basis)
        
                shortest_edge = min(shortest_edge, obj.shortest_edge)
                
                # get loop and star basis information from children
                try:
                    loop_basis.append(obj.loop_basis)
                    # arbitrarily chose the last star of each element to drop
                    star_basis.append(obj.star_basis[:-1, :])
                except AttributeError:
                    pass

            # find the largest distance between nodes within the structure
            max_distance = np.sqrt(np.sum((nodes[:, None, :]-nodes[None, :, :])**2, axis=2)).max()

            # Store the parts of the combined mesh which don't depend on the loop
            # star basis functions being defined
            self._combined_mesh = {'nodes' : nodes, 'tri' : tri, 
                                   'basis' : basis, 'objs' : objs, 
                                   'max_distance' : max_distance,
                                   'shortest_edge' : shortest_edge}

            # only keep loop and star basis if all children have it
            if len(loop_basis) != N_objs or len(star_basis) != N_objs:
                #loop_basis = None
                #star_basis = None
                pass
            else:
                n_loop = np.hstack(([0], 
                                np.cumsum([l.shape[0] for l in loop_basis])))
                n_J = np.hstack(([0], 
                                 np.cumsum([l.shape[1] for l in loop_basis])))
                
                # scipy sparse matrices cannot have zero dimensions        
                if n_loop[-1] == 0:
                    loop_basis_comb = np.zeros((0, N_basis))
                else:
                    loop_basis_comb = np.zeros((n_loop[-1], n_J[-1]), np.float32)
                    #loop_basis_comb = sp.lil_matrix((n_loop[-1], N_basis), dtype=np.float64)
                    for index, sub_matrix in enumerate(loop_basis):
                        loop_basis_comb[n_loop[index]:n_loop[index+1], 
                                        n_J[index]:n_J[index+1]] = sub_matrix
    
                    # loop_basis_comb = loop_basis_comb.tocsr()

                n_star = np.hstack(([0], np.cumsum([s.shape[0] for s in star_basis])))
                star_basis_comb = np.zeros((n_star[-1], n_J[-1]), np.float32)
                #star_basis_comb = sp.lil_matrix((n_star[-1], N_basis), dtype=np.float64)
                for index, sub_matrix in enumerate(star_basis):
                    star_basis_comb[n_star[index]:n_star[index+1], 
                                    n_J[index]:n_J[index+1]] = sub_matrix
                
                #star_basis_comb = star_basis_comb.tocsr()

                # scipy sparse matrices cannot have zero dimensions        
                if n_loop[-1] == 0:
                    loop_star_transform = star_basis_comb
                else:
                    #loop_star_transform = sp.vstack((loop_basis, star_basis)).tocsr()
                    loop_star_transform = np.vstack((loop_basis_comb, star_basis_comb))
                
                self._combined_mesh['loop_star_transform'] = loop_star_transform 
                self._combined_mesh['n_loop'] = n_loop[-1]

                # the ranges into the impedance matrix for the loops and stars of
                # each object
                self._combined_mesh['loop_ranges'] = np.array([[n_loop[i], n_loop[i+1]] for i in xrange(N_objs)])
                self._combined_mesh['star_ranges'] = np.array([[n_star[i], n_star[i+1]] for i in xrange(N_objs)])+n_loop[-1]
            
            return self._combined_mesh

    @property
    def combined_precalc(self):
        """Combine the precalculated data of several parts
        
        If the combined precalc data is invalid then recalculate it"""
        
        try:
            return self._combined_precalc    
        except AttributeError:
            
            offset_tri = 0
            precalc_sparse = SingularSparse()
        
            for part in self._parts:
        
                singular_terms = part.precalc_singular(self.quadrature_rule)
        
                # the pre-calculated self-impedance data
                for key, value in singular_terms.iteritems():
                    precalc_sparse[(key[0]+offset_tri, key[1]+offset_tri)] = value
                
                offset_tri += len(part.tri)
        
            self._combined_precalc =  precalc_sparse.tocsr()
            return self._combined_precalc


    # simplify access to the nodes and triangles using properties    
    @property
    def nodes(self):
        """The combined nodes of all objects in this simulation"""
        return self.combined_mesh['nodes']

    @property
    def tri(self):
        """The combined triangles of all objects in this simulation"""
        return self.combined_mesh['tri']

    @property
    def basis(self):
        """The combined basis functions of all objects in this simulation"""
        return self.combined_mesh['basis']
        
    @property
    def loop_star_transform(self):
        """The transformation from RWG to loop-star basis functions"""
        return self.combined_mesh['loop_star_transform']

    @property
    def n_loop(self):
        """The number of loop basis functions"""
        return self.combined_mesh['n_loop']

    @property
    def triangle_to_rwg(self):
        """The matrix which converts from triangle to RWG basis"""
        return self.combined_mesh['triangle_to_rwg']

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
        return self.operator.self_impedance_matrix(self._parts[0], s)

  
    def impedance_matrix2(self, s, serial_interpolation = False, 
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
        
        xi_eta_eval, weights = self.quadrature_rule

        nodes = self.nodes
        basis = self.basis
        tri = self.tri

        (I_A_sing, I_phi_sing, index_sing, indptr_sing) = self.combined_precalc
   
        A_faces, phi_faces = core_for.z_efie_faces(nodes, tri.nodes, s, 
           xi_eta_eval, weights, I_phi_sing, I_A_sing, index_sing, indptr_sing)

#        A_faces, phi_faces = Z_EFIE_faces(nodes, tri.nodes, s, 
#           xi_eta_eval, weights, I_phi_sing, I_A_sing, index_sing, indptr_sing)

        
        #import core_cython
        L, S = core_for.triangle_face_to_rwg(basis.tri_p, basis.tri_m, 
                                basis.node_p, basis.node_m, A_faces, phi_faces)
        
        L *= mu_0/(4*np.pi)
        S *= 1/(np.pi*epsilon_0)

                                             
        if loop_star:
            # perform transformation to loop-star basis
            
            L = self.loop_star_transform.dot(L.dot(self.loop_star_transform.T))
            S = self.loop_star_transform.dot(S.dot(self.loop_star_transform.T))
            
            n_loop = self.n_loop
            S[:n_loop, :] = 0.0
            S[:, :n_loop] = 0.0            

        return L, S

    def sub_matrix(self, matrix, part1, part2 = None, loop_star = True):
        """Return a sub-matrix, corresponding to the self term for a particular
        object or mutual term between a pair of objects
        
        Parameters
        ----------
        matrix: ndarray
            the impedance matrix, or one of its components
        part1: number
            which part to calculate matrix for
        part2: number, optional
            Second part. If not specified, self matrix given by part1 will
            be returned
        loop_star: boolean, optional
            Whether loop-star basis functions are used
            
        Returns
        -------
        sub_matrix : ndarray
            The requested subsection of the matrix
        """

        if part2 is None:
            part2 = part1
        
        if loop_star:
            loop_ranges, star_ranges = self.object_basis()
            
            loop1 = loop_ranges[part1]
            loop2 = loop_ranges[part2]

            star1 = star_ranges[part1]
            star2 = star_ranges[part2]

            return np.vstack((np.hstack((matrix[loop1, loop2], 
                                         matrix[loop1, star2])), 
                              np.hstack((matrix[star1, loop2], 
                                         matrix[star1, star2])) ))
            
        else:
            raise NotImplementedError
        
    def source_term(self, e_inc, jk_inc, loop_star = True):
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

        return self.operator.plane_wave_source(self._parts[0], e_inc, jk_inc)
        

        xi_eta_eval, weights = self.quadrature_rule
        
        nodes = self.nodes
        tri = self.tri
        basis = self.basis

        incident = core_for.voltage_plane_wave(nodes, tri.nodes, basis.tri_p, 
                                       basis.tri_m, basis.node_p, basis.node_m, 
                                       xi_eta_eval, weights, e_inc, jk_inc)

        if loop_star:
            incident = self.loop_star_transform.dot(incident)

        return incident

    def linearised_eig(self, L, S, n_modes, which_obj = None):
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
        
        if which_obj is None:
            # just solve for the whole system, which is easy
            
            loop_range = slice(0, self.n_loop)
            star_range = slice(self.n_loop, len(self.basis))
            
        else:
            loop_ranges, star_ranges = self.object_basis()
            loop_range = loop_ranges[which_obj]
            star_range = star_ranges[which_obj]

        if loop_range.start == loop_range.stop:
            # object has no loops
            no_loops = True
        else:
            no_loops = False
            
        if no_loops:
            L_red = L[star_range, star_range]
        else:
            L_conv = la.solve(L[loop_range, loop_range], 
                              L[loop_range, star_range])
            L_red = L[star_range, star_range] - np.dot(L[star_range, loop_range], L_conv)

        # find eigenvalues, and star part of eigenvectors, for LS combined modes
        w, v_s = la.eig(S[star_range, star_range], L_red)
        
        if no_loops:
            vr = v_s
        else:
            v_l = -np.dot(L_conv, v_s)
            vr = np.vstack((v_l, v_s))
        
        w_freq = np.sqrt(w)/2/np.pi
        w_selected = np.ma.masked_array(w_freq, w_freq.real < w_freq.imag)
        which_modes = np.argsort(w_selected.real)[:n_modes]
        
        return np.sqrt(w[which_modes]), vr[:, which_modes]

