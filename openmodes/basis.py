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

from collections import namedtuple

# A named tuple for holding the positive and negative triangles and nodes
# which are used by both RWG and loop-star basis functions
RWG = namedtuple('RWG', ('tri_p', 'tri_m', 'node_p', 'node_m'))

# TODO: interpolation needs to account for factor 1/(2A)

def nodes_not_in_edge(nodes, edge):
    """Given a list of nodes numbers, return the indices of those which do 
    not belong to the specified edge
    
    Parameters
    ----------
    nodes : array or list
        The list of node numbers
    edge: array or list
        The edge, defined by two (or more) node number
    
    Return
    ------
    non_member_nodes : list
        The nodes which are not a member of the given edge
    """
    return [node_index for node_index, node_num in enumerate(nodes)
                if node_num not in edge]
                    
def shared_nodes(nodes1, nodes2):
    """Return all the nodes shared by two polyhedra
    
    Parameters
    ----------
    nodes1 : array/list
        A list of node numbers
    nodes2 : array/list
        A list of node numbers
    
    Returns
    -------
    common_nodes : list
        A list of nodes found in both lists
    """
    return [node for node in nodes1 if node in nodes2]

def interpolate_triangle(nodes, edge_vals, xi_eta):
    """Interpolate a function over a triangle using first order interpolation
    
    Parameters
    ---------
    nodes : ndarray
        The coordinates of the 3 triangle nodes
    edge_vals : ndarray
        The values of the function normal to each of the edges
    xi_eta : ndarray
        The barycentric coordinates over which to obtain values
        
    Returns
    -------
    r : ndarray
        The points within the triangles
    result : ndarray
        The vector solution at points within the triangle
    """

    num_points = len(xi_eta)
    result = np.zeros((num_points, 3), edge_vals.dtype)
    r = np.empty((num_points, 3), np.float64)

    for count, (xi, eta) in enumerate(xi_eta):
        # Barycentric coordinates of the observer
        zeta = 1.0 - eta - xi

        # Cartesian coordinates of the point
        r[count] = xi*nodes[0] + eta*nodes[1] + zeta*nodes[2]

        for node_count in xrange(3):
            # Vector rho within the observer triangle
            rho = r[count] - nodes[node_count]

            result[count] += rho*edge_vals[node_count]

    return r, result

def interpolate_triangle_mesh(mesh, tri_func, num_tri, xi_eta, flatten=True):
    """Interpolate a function on a triangular mesh with linear basis functions
    on each face"""

    points_per_tri = len(xi_eta)

    r_all = np.empty((num_tri, points_per_tri, 3), mesh.nodes.dtype)
    func_all = np.empty((num_tri, points_per_tri, 3), tri_func.dtype)


    for tri_count, node_nums in enumerate(mesh.triangle_nodes):
        r, interp_func = interpolate_triangle(mesh.nodes[node_nums], 
                                              tri_func[tri_count], xi_eta)
        r_all[tri_count] = r
        func_all[tri_count] = interp_func

    if flatten:
        r_all = r_all.reshape((-1, 3))
        func_all = func_all.reshape((-1, 3))
    
    return r_all, func_all


def triangle_face_to_rwg(face_val, rwg_o, rwg_s = None):
    """Take quantities which are defined as interaction between faces and 
    convert them to RWG basis
    
    Parameters
    ----------
    face_val : ndarray
        The function, which can be either a vector or scalar defined over faces
    rwg_o : RwgDivBasis
        The RWG basis functions of the observer
    rwg_s : RwgDivBasis
        The RWG basis functions of the source. This is not required if the
        desired function is a vector rather than a matrix

    Returns
    -------
    rwg_val : ndarray
        Either a 1D or 2D array, collected over RWG basis functions
    """
 
#    if basis_s is None:
#        basis_s = basis
 
#    rwg_val = np.empty((len(basis_o), len(basis_s)), face_val.dtype)
    if len(face_val.shape) == 4:

#        p_p = basis_o.tri_p
#        p_m = basis_o.tri_m
#        ip_p = basis_o.node_p
#        ip_m = basis_o.node_m
#
#        q_p = basis_s.tri_p
#        q_m = basis_s.tri_m
#        iq_p = basis_s.node_p
#        iq_m = basis_s.node_m

        p_p, p_m, ip_p, ip_m = rwg_o
        q_p, q_m, iq_p, iq_m = rwg_s

        rwg_val = ( 
            face_val[p_p[:, None], q_p[None, :], ip_p[:, None], iq_p[None, :]]
          - face_val[p_p[:, None], q_m[None, :], ip_p[:, None], iq_m[None, :]]
          - face_val[p_m[:, None], q_p[None, :], ip_m[:, None], iq_p[None, :]] 
          + face_val[p_m[:, None], q_m[None, :], ip_m[:, None], iq_m[None, :]])


    elif len(face_val.shape) == 2:
        p_p = rwg_o.tri_p
        p_m = rwg_o.tri_m
        q_p = rwg_s.tri_p
        q_m = rwg_s.tri_m

        rwg_val = ( 
                face_val[p_p[:, None], q_p[None, :]] 
              - face_val[p_p[:, None], q_m[None, :]]
              - face_val[p_m[:, None], q_p[None, :]] 
              + face_val[p_m[:, None], q_m[None, :]])
    else:
        raise ValueError("Don't know how to convert his function to RWG basis")

    return rwg_val

def rwg_to_triangle_face(rwg_func, num_tri, rwg):
    """Convert from RWG basis, to triangle face basis
    
    Parameters
    ----------
    rwg_func : ndarray
        The function defined in RWG basis
    num_tri : integer
        The number of triangles
    rwg : RWG
        The RWG data
    Returns
    -------
    tri_func : ndarray
        The function defined over triangle faces
    """
    tri_func = np.zeros((num_tri, 3), rwg_func.dtype)
    
    for count, func in enumerate(rwg_func):
        tri_func[rwg.tri_p[count], rwg.node_p[count]] += func
        tri_func[rwg.tri_m[count], rwg.node_m[count]] -= func
        
    return tri_func

class DivRwgBasis(object):
    """Divergence-conforming RWG basis functions
    
    The simplest basis functions which can be defined on a triangular mesh.
    
    The basis functions are defined in a coordinate independent manner, so that
    they can be re-used for the same mesh placed in a different location    
    """
    
    def __init__(self, mesh, edge_cache = None):
        """Generate basis functions for a meshicular mesh. Note that the mesh
        will be referenced, so it should not be modified after generating the
        basis functions.
        """
        self.mesh = mesh

        if edge_cache is None:
            edges, triangles_shared_by_edges = mesh.get_edges(True)
        else:
            edges, triangles_shared_by_edges = edge_cache

        sharing_count = np.array([len(n) for n in triangles_shared_by_edges])

        if min(sharing_count) < 1 or max(sharing_count) > 2:
            raise ValueError("Mesh edges must be part of exactly 1 or 2" +
                             "triangles for RWG basis functions")

        shared_edge_indices = np.where(sharing_count == 2)[0]

        num_basis = len(shared_edge_indices)
        # index of T+
        tri_p = np.empty(num_basis, np.int32)
        # index of T-
        tri_m = np.empty(num_basis, np.int32)
        # internal index of free node of T+
        node_p = np.empty(num_basis, np.int32)
        # internal index of free node of T-
        node_m = np.empty(num_basis, np.int32)

        for basis_count, edge_count in enumerate(shared_edge_indices):
            # set the RWG basis function triangles
            tri_p[basis_count] = triangles_shared_by_edges[edge_count][0]
            tri_m[basis_count] = triangles_shared_by_edges[edge_count][1]
            
            # determine the indices of the unshared nodes, indexed within the
            # sharing triangles (i.e. 0, 1 or 2)
            node_p[basis_count] = nodes_not_in_edge(
                            mesh.triangle_nodes[tri_p[basis_count]], edges[edge_count])[0]
            node_m[basis_count] = nodes_not_in_edge(
                            mesh.triangle_nodes[tri_m[basis_count]], edges[edge_count])[0]

        self.rwg = RWG(tri_p, tri_m, node_p, node_m)

    def interpolate_function(self, rwg_func, xi_eta, flatten = True):
        """Interpolate a function defined in RWG basis, over the complete mesh
        
        Parameters
        ----------
        rwg_func : ndarray
            The function defined over RWG basis functions
        xi_eta : ndarray
            The barycentric coordinates to interpolate on each triangle
        flatten : bool, optional
            If false, data for each triangle will be identified by a specific
            index of the array, otherwise all points are identical
            
        Returns
        -------
        r_all : ndarray
            The interpolation points
        func_all : ndarray
            The vector function at each interpolation point
        """
        num_tri = len(self.mesh.triangle_nodes)
        tri_func = rwg_to_triangle_face(rwg_func, num_tri, self.rwg)

        return interpolate_triangle_mesh(self.mesh, tri_func, num_tri, xi_eta, flatten)
     
    def transformation_matrix(self):
        """Returns the (sparse???) transformation matrix to turn quantities
        defined on faces to RWG basis
        
        For vector quantities, assumes that the face-based quantity has been 
        packed to a 2D array of size (n_basis*3, n_basis*3)

        For scalar quantities, assumes that the face-based quantity has been 
        packed to a 2D array of size (n_basis, n_basis)

        """
        
        num_basis = len(self)
        num_tri = len(self.mesh.triangle_nodes)
        scalar_transform = np.zeros((num_basis, num_tri), np.float64)
        vector_transform = np.zeros((num_basis, 3*num_tri), np.float64)
        
        for basis_count, (tri_p, tri_m, node_p, node_m) in enumerate(self):
            scalar_transform[basis_count, tri_p] = 1.0
            scalar_transform[basis_count, tri_m] = -1.0

            vector_transform[basis_count, tri_p*3+node_p] = 1.0
            vector_transform[basis_count, tri_m*3+node_m] = -1.0

        return vector_transform, scalar_transform
            
     
    def __len__(self):
        return len(self.rwg.tri_p)
        
    def __getitem__(self, index):
        return RWG(self.rwg.tri_p[index], self.rwg.tri_m[index], 
                   self.rwg.node_p[index], self.rwg.node_m[index])

def construct_stars(mesh, edges, triangles_shared_by_edges, sharing_count):
    """Construct star basis functions on a triangular mesh"""
    
    num_tri = len(mesh.triangle_nodes)        

    shared_edge_indices = np.where(sharing_count == 2)[0]

    tri_p = [list() for _ in xrange(num_tri)]
    tri_m = [list() for _ in xrange(num_tri)]
    node_p = [list() for _ in xrange(num_tri)]
    node_m = [list() for _ in xrange(num_tri)]
    
    # Go through shared edges, and update both star-basis functions
    # to add the influence of this shared edge
    for edge_count in shared_edge_indices:
        tri1, tri2 = triangles_shared_by_edges[edge_count]

        tri_p[tri1].append(tri1)
        tri_p[tri2].append(tri2)

        tri_m[tri1].append(tri2)
        tri_m[tri2].append(tri1)

        node1 = nodes_not_in_edge(mesh.triangle_nodes[tri1], 
                                  edges[edge_count])[0]
        node2 = nodes_not_in_edge(mesh.triangle_nodes[tri2], 
                                  edges[edge_count])[0]
        node_p[tri1].append(node1)
        node_p[tri2].append(node2)

        node_m[tri1].append(node2)
        node_m[tri2].append(node1)
 
    return RWG(tri_p, tri_m, node_p, node_m)

def construct_loop(loop_triangles, triangle_nodes):
    """Construct a single loop basis function corresponding to a single inner
    node of a triangular mesh"""

    tri_p = []
    tri_m = []
    node_p = []
    node_m = []
    
    # The first triangle to be picked doesn't get removed from
    # the list, as we will need to refer to it again
    current_triangle = loop_triangles.pop() #[0]
    #ordered_triangles = [current_triangle]
    first_triangle = current_triangle
   
    #done_triangles = [current_triangle]
   
    while len(loop_triangles) > 0:
        for triangle_count, next_triangle in enumerate(loop_triangles):
            shared = shared_nodes(triangle_nodes[next_triangle], triangle_nodes[current_triangle])
            if len(shared) == 2:
                break
        #raise ValueError("Cannot find adjoining triangle") # wrong!
        loop_triangles.pop(triangle_count)
        #ordered_triangles.append(current_triangle)
        
        # find the unshared nodes
        free_current = nodes_not_in_edge(triangle_nodes[current_triangle], shared)[0]
        free_next = nodes_not_in_edge(triangle_nodes[next_triangle], shared)[0]
        
        tri_p.append(current_triangle)
        tri_m.append(next_triangle)

        node_p.append(free_current)
        node_m.append(free_next)
        
        #done_triangles.append(current_triangle)
        current_triangle = next_triangle

    # now connect the loop with the first and last triangle
    shared = shared_nodes(triangle_nodes[first_triangle], triangle_nodes[current_triangle])

    free_current = nodes_not_in_edge(triangle_nodes[current_triangle], shared)[0]
    free_next = nodes_not_in_edge(triangle_nodes[first_triangle], shared)[0]

    tri_p.append(current_triangle)
    tri_m.append(first_triangle)

    node_p.append(free_current)
    node_m.append(free_next)



    return RWG(tri_p, tri_m, node_p, node_m)

class LoopStarBasis(object):
    """Loop-Star basis functions.
    
    Similar to div conforming RWG, but explicitly divides the basis functions
    into divergence-free loops and approximately curl-free stars.

    See:

    [1] G. Vecchi, “Loop-star decomposition of basis functions in the 
    discretization of the EFIE,” IEEE Transactions on Antennas and 
    Propagation, vol. 47, no. 2, pp. 339–346, 1999.

    [2] J.-F. Lee, R. Lee, and R. J. Burkholder, “Loop star basis 
    functions and a robust preconditioner for EFIE scattering
    problems,” IEEE Transactions on Antennas and Propagation, 
    vol. 51, no. 8, pp. 1855–1863, Aug. 2003.
    """
    
    def __init__(self, mesh):
        self.mesh = mesh

        edges, triangles_shared_by_edges = mesh.get_edges(True)

        sharing_count = np.array([len(n) for n in triangles_shared_by_edges])

        if min(sharing_count) < 1 or max(sharing_count) > 2:
            raise ValueError("Mesh edges must be part of exactly 1 or 2" +
                             "triangles for loop-star basis functions")


        self.rwg_star = construct_stars(mesh, edges, triangles_shared_by_edges, sharing_count)


        # Now start searching for loops

        num_nodes = len(mesh.nodes)        

        # find the set of unshared edges
        unshared_edges = edges[np.where(sharing_count == 1)[0]]

        # then find all the boundary nodes
        outer_nodes = set()
        for edge in unshared_edges:
            # TODO: check correct datatype of edge
            outer_nodes.add(edge[0])
            outer_nodes.add(edge[1])

        # find the nodes which don't belong to any shared edge
        inner_nodes = set(xrange(num_nodes)) - outer_nodes

        #n_vertices = len(self.inner_nodes)+len(outer_nodes)
        #print "vertices", n_vertices
        #print "faces", len(triangles)
        #print "edges", len(all_edges)
        boundary_contours = 2-num_nodes+len(edges)-len(mesh.triangle_nodes)
        #print "separated contours", boundary_contours
        
        if boundary_contours > len(inner_nodes):
            # TODO: need to deal with holes in 2D (and 3D?) structures
            raise NotImplementedError
            
        
        # Note that this would create one basis function for each inner 
        # node which may exceed the number of RWG degrees of freedom. In
        # this case arbitrary loops at the end of the list are dropped in
        # the conversion
        
        num_loops = len(inner_nodes) + boundary_contours - 1

        triangles_sharing_nodes = mesh.triangles_sharing_nodes()

        loop_tri_p = []
        loop_tri_m = []
        loop_node_p = []
        loop_node_m = []

        if num_loops != 0:
      
            for loop_number, node_number in enumerate(inner_nodes):

                # there may be fewer loops than inner nodes
                if loop_number == num_loops:
                    break

                # find all the triangles sharing this node
                loop_triangles = list(triangles_sharing_nodes[node_number])
                
                this_loop = construct_loop(loop_triangles, mesh.triangle_nodes)
                loop_tri_p.append(this_loop[0])
                loop_tri_m.append(this_loop[1])
                loop_node_p.append(this_loop[2])
                loop_node_m.append(this_loop[3])
                
        self.rwg_loop = RWG(loop_tri_p, loop_tri_m, loop_node_p, loop_node_m)

    def interpolate_function(self, ls_func, xi_eta, flatten = True):
        """Interpolate a function defined in loop-star basis over the complete
        mesh
        
        Parameters
        ----------
        ls_func : ndarray
            The function defined over loop-star basis functions
        xi_eta : ndarray
            The barycentric coordinates to interpolate on each triangle
        flatten : bool, optional
            If false, data for each triangle will be identified by a specific
            index of the array, otherwise all points are identical
            
        Returns
        -------
        r_all : ndarray
            The interpolation points
        func_all : ndarray
            The vector function at each interpolation point
        """
        num_tri = len(self.mesh.triangle_nodes)
        combined_rwg = RWG._make(a + b for a, b in zip(self.rwg_loop, self.rwg_star))
        tri_func = rwg_to_triangle_face(ls_func, num_tri, combined_rwg)

        return interpolate_triangle_mesh(self.mesh, tri_func, num_tri, xi_eta, flatten)

    def __len__(self):
        return len(self.rwg_loop.tri_p)+len(self.rwg_star.tri_p)

    @property
    def num_loops(self):
        "The number of loops in the loop-star mesh"
        return len(self.rwg_loop.tri_p)

    def __getitem__(self, index):
        if index < 0:
            index += len(self)
            
        if index >= self.num_loops:
            index -= self.num_loops
            return RWG(self.rwg_star.tri_p[index], self.rwg_star.tri_m[index], 
                    self.rwg_star.node_p[index], self.rwg_star.node_m[index])
        else:
            return RWG(self.rwg_loop.tri_p[index], self.rwg_loop.tri_m[index],
                    self.rwg_loop.node_p[index], self.rwg_loop.node_m[index])
            

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
        