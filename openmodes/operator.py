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


import numpy as np

import openmodes_core

from openmodes.constants import epsilon_0, mu_0, pi
from openmodes.basis import DivRwgBasis, generate_basis_functions, triangle_face_to_rwg

class FreeSpaceGreensFunction(object):
    "Green's function in Free Space"

class SingularSparse(object):
    """A sparse matrix class for holding A and phi arrays with the same 
    sparsity pattern to store singular triangle impedances"""
    def __init__(self):
        self.rows = {}
        
    def __setitem__(self, index, item):
        """Add an item, which will be stored in a dictionary of dictionaries. 
        Item is assumed to be (A, phi)"""
        
        row, col = index
        try:
            self.rows[row][col] = item
        except KeyError:
            self.rows[row] = {col: item}
    
    def iteritems(self):
        for row, row_dict in self.rows.iteritems():
            for col, item in row_dict.iteritems():
                yield ((row, col), item)
    
    def to_csr(self):
        """Convert the matrix to compressed sparse row format, with 
        common index array and two data arrays for A and phi"""
        A_data = []
        phi_data = []
        indices = []
        indptr = [0]
        
        data_index = 0

        num_rows = max(self.rows.keys())+1

        for row in xrange(num_rows):
            if row in self.rows:
                # the row exists, so process it
                for col, item in self.rows[row].iteritems():
                    A_data.append(item[0])
                    phi_data.append(item[1])
                    indices.append(col)
                    
                    data_index = data_index + 1
            # regardless of whether the row exists, update the index pointer
            indptr.append(data_index)
            
        return (np.array(phi_data, dtype=np.float64, order="F"),
                np.array(A_data, dtype=np.float64, order="F"), 
                np.array(indices, dtype=np.int32, order="F"), 
                np.array(indptr, dtype=np.int32, order="F"))

cached_singular_terms = {}

def singular_impedance_rwg_efie_homogeneous(basis, quadrature_rule):
    """Precalculate the singular impedance terms for an object

    Parameters
    ----------
    quadrature_rule : tuple of 2 ndarrays
        the barycentric coordinates and weights of the quadrature to
        use for the non-analytical neighbour terms
        
    Returns
    -------
    singular_terms : SingularSparse object
        the sparse array of singular impedance terms
    
    """
    unique_id = ("EFIE", "RWG", basis.__hash__(), quadrature_rule.__repr__())
    if unique_id in cached_singular_terms:
        #print "singular terms retrieved from cache"
        return cached_singular_terms[unique_id]
    else:
        xi_eta_eval, weights = quadrature_rule        
        
        sharing_nodes = basis.mesh.triangles_sharing_nodes()        
        
        # Precalculate the singular integration rules for faces, which depend
        # on the observation point    
        triangle_nodes = basis.mesh.triangle_nodes
        nodes = basis.mesh.nodes
        N_face = len(triangle_nodes)
    
        singular_terms = SingularSparse()
        # find the neighbouring triangles (including self terms) to integrate
        # singular part
        for p in xrange(0, N_face): # observer:
            
            nodes_p = nodes[triangle_nodes[p]]

            sharing_triangles = set()
            for node in triangle_nodes[p]:
                sharing_triangles = sharing_triangles.union(sharing_nodes[node])
            
            # find any neighbouring elements which are touching
            for q in sharing_triangles:
                if q == p:
                    # calculate the self term using the exact formula
                    res = openmodes_core.arcioni_singular(nodes_p,)
                    assert(np.all(np.isfinite(res[0])) and np.all(np.isfinite(res[1])))
                    singular_terms[p, p] = res
                else:
                    # at least one node is shared
                    # calculate neighbour integrals semi-numerically
                    #nodes_q = 
                    res = openmodes_core.face_integrals_hanninen(
                                        nodes[triangle_nodes[q]], xi_eta_eval, weights, nodes_p)
                    assert(np.all(np.isfinite(res[0])) and np.all(np.isfinite(res[1])))
                    singular_terms[p, q] = res
        
        cached_singular_terms[unique_id] = singular_terms.to_csr()
        return cached_singular_terms[unique_id]

#import core_cython

def impedance_rwg_efie_free_space(s, quadrature_rule, basis_o, nodes_o, basis_s = None, nodes_s = None):

    xi_eta_eval, weights = quadrature_rule

    if (basis_s is None):
        # calculate self impedance

        singular_terms = singular_impedance_rwg_efie_homogeneous(basis_o, 
                                                             quadrature_rule)

        (I_phi_sing, I_A_sing, index_sing, indptr_sing) = singular_terms
        #assert(sum(np.isnan(I_phi_sing)) == 0)
        #assert(sum(np.isnan(I_A_sing)) == 0)
   
        #n_basis = len(basis_o)
        n_faces = len(basis_o.mesh.triangle_nodes)
        A_faces, phi_faces = openmodes_core.z_efie_faces_self(nodes_o, 
                                          basis_o.mesh.triangle_nodes, s, 
                                          xi_eta_eval, weights, I_phi_sing, I_A_sing, index_sing, indptr_sing)
                                          #*singular_terms) #

        #L = triangle_face_to_rwg(A_faces, basis.rwg, basis.rwg)
        #S = triangle_face_to_rwg(phi_faces, basis.rwg, basis.rwg)
    
        # needs to be cached
        transform_L, transform_S = basis_o.transformation_matrices
    
        # A faces does not need to be transposed as it is symmetric
        L = transform_L.dot(transform_L.dot(A_faces.reshape(n_faces*3, n_faces*3)).T)
        S = transform_S.dot(transform_S.dot(phi_faces).T)
    
        #L, S = openmodes_core.triangle_face_to_rwg(basis_o.rwg.tri_p, basis_o.rwg.tri_m,
        #                                           basis_o.rwg.node_p, basis_o.rwg.node_m, A_faces, phi_faces)
    
        #L, S = core_cython.triangle_face_to_rwg(basis.rwg.tri_p, basis.rwg.tri_m,
        #                                           basis.rwg.node_p, basis.rwg.node_m, A_faces, phi_faces)


    else:
        # calculate mutual impedance

        A_faces, phi_faces = openmodes_core.z_efie_faces_mutual(nodes_o, 
                                basis_o.mesh.triangle_nodes, nodes_s, 
                                basis_s.mesh.triangle_nodes, s, xi_eta_eval, weights)

        L = triangle_face_to_rwg(A_faces, basis_o.rwg, basis_s.rwg)
        S = triangle_face_to_rwg(phi_faces, basis_o.rwg, basis_s.rwg)
#    L, S = openmodes_core.triangle_face_to_rwg(basis.tri_p, basis.tri_m, 
#                            basis.node_p, basis.node_m, A_faces, phi_faces)
    
    L *= mu_0/(4*pi)
    S *= 1/(pi*epsilon_0)
    return L, S

class EfieOperator(object):
    """An operator for the electric field integral equation, discretised with
    respect to some set of basis functions. Assumes that Galerkin's method is
    used, such that the testing functions are the same as the basis functions.
    """
    def __init__(self, quadrature_rule, 
                 greens_function=FreeSpaceGreensFunction(),
                 basis_class=DivRwgBasis):
        self.basis_class = basis_class
        self.quadrature_rule = quadrature_rule
        self.greens_function = greens_function
        
    def impedance_matrix(self, s, part_o, part_s=None):
        """Calculate the impedance matrix for a single part, at a given
        complex frequency s"""
        
        basis_o = generate_basis_functions(part_o.mesh, self.basis_class)
        
        if part_s is None:
            basis_s = None
            nodes_s = None
        else:
            basis_s = generate_basis_functions(part_s.mesh, self.basis_class)
            nodes_s = part_s.mesh
            
        
        if isinstance(self.greens_function, FreeSpaceGreensFunction):
            if True: #isinstance(basis_o, DivRwgBasis):
                return impedance_rwg_efie_free_space(s, self.quadrature_rule, 
                                                     basis_o, part_o.nodes, 
                                                     basis_s, nodes_s)
            else:
                raise NotImplementedError
    
        else:
            raise NotImplementedError
            
    def plane_wave_source(self, part, e_inc, jk_inc):
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
        basis = generate_basis_functions(part.mesh, self.basis_class)
        #basis = self.basis[part_number]

        if (True and #isinstance(basis, DivRwgBasis) and 
            isinstance(self.greens_function, FreeSpaceGreensFunction)):

            xi_eta_eval, weights = self.quadrature_rule
  
            incident_faces = openmodes_core.v_efie_faces_plane_wave(part.nodes, 
                                        basis.mesh.triangle_nodes, xi_eta_eval,
                                        weights, e_inc, jk_inc)


            transform_L, _ = basis.transformation_matrices

            return transform_L.dot(incident_faces.flatten())
          
#            from core_cython import voltage_plane_wave_serial as voltage_plane_wave          
#            incident = voltage_plane_wave(np.ascontiguousarray(part.nodes), 
#                            np.ascontiguousarray(basis.mesh.triangle_nodes, dtype=np.int32), 
#                            np.ascontiguousarray(basis.rwg.tri_p, dtype=np.int32), 
#                            np.ascontiguousarray(basis.rwg.tri_m, dtype=np.int32), 
#                            np.ascontiguousarray(basis.rwg.node_p, dtype=np.int32), 
#                            np.ascontiguousarray(basis.rwg.node_m, dtype=np.int32), 
#                            np.ascontiguousarray(xi_eta_eval), 
#                            np.ascontiguousarray(weights[0]), e_inc, jk_inc)


#            from openmodes_core import voltage_plane_wave
#            
#            incident = voltage_plane_wave(part.nodes, 
#                            basis.mesh.triangle_nodes, basis.rwg.tri_p, 
#                            basis.rwg.tri_m, basis.rwg.node_p, basis.rwg.node_m, 
#                            xi_eta_eval, weights, e_inc, jk_inc)
                                           
            #return incident
        else:
            raise NotImplementedError
            
        

        