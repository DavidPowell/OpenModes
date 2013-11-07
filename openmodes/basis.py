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


"""

The basis functions are defined in a coordinate independent manner, so that
they can be re-used for the same mesh placed in a different location

Created on Wed Nov 06 09:56:58 2013

@author: dap124
"""

import numpy as np

class DivRwgBasis(object):
    """Divergence-conforming RWG basis functions
    
    The simplest basis functions which can be defined on a triangular mesh.
    """
    
    def __init__(self, mesh):
        """Generate basis functions for a meshicular mesh. Note that the mesh
        will be referenced, so it should not be modified after generating the
        basis functions.
        """
        self.mesh = mesh

        edges, triangles_shared_by_edges = mesh.get_edges(True)

        sharing_count = np.array([len(n) for n in triangles_shared_by_edges])

        if min(sharing_count) < 1 or max(sharing_count) > 2:
            raise ValueError("Mesh edges must be part of exactly 1 or 2" +
                             "triangles for RWG basis functions")

        shared_edge_indices = np.where(sharing_count == 2)[0]

        N_basis = len(shared_edge_indices)
        self.tri_p = np.empty(N_basis, np.int32) # index of T+
        self.tri_m = np.empty(N_basis, np.int32) # index of T-
        self.node_p = np.empty(N_basis, np.int32) # internal index of free node of T+
        self.node_m = np.empty(N_basis, np.int32) # internal index of free node of T-

        for basis_count, edge_count in enumerate(shared_edge_indices):
            # set the RWG basis function triangles
            tri_p = triangles_shared_by_edges[edge_count][0]
            tri_m = triangles_shared_by_edges[edge_count][1]
            
            self.tri_p[basis_count] = tri_p
            self.tri_m[basis_count] = tri_m

            # determine the indices of the unshared nodes, indexed within the
            # sharing triangles (i.e. 0, 1 or 2)
            self.node_p[basis_count] = np.where([n not in edges[edge_count] for n in mesh.triangle_nodes[tri_p]])[0][0]
            self.node_m[basis_count] = np.where([n not in edges[edge_count] for n in mesh.triangle_nodes[tri_m]])[0][0]

    # TODO: add members for len, rho_cp, rho_cm          

    def __len__(self):
        return len(self.tri_p)

cached_basis_functions = {}      
        
def generate_basis_functions(mesh, basis_class=DivRwgBasis):
    """Generate basis functions for a mesh. Performs caching, so that if an
    identical mesh has already been generated, the basis functions will
    not be unnecessarily duplicated
    
    Parameters
    ----------
    mesh : object
        The mesh to generate the basis functions for
    
    basis_class : class, optional
        Which class of basis function should be created
    """
    
    # The following parameters are needed to determine if basis functions are
    # unique. Potentially this could be expanded to include non-affine mesh
    # transformations, or other parameters passed to the basis function
    # constructor
    unique_key = (mesh, basis_class)    
    
    if unique_key in cached_basis_functions:
        #print "Basis functions retrieved from cache"
        return cached_basis_functions[unique_key]
    else:
        result = basis_class(mesh)
        cached_basis_functions[unique_key] = result
        return result
        