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
"""
Routines to construct the basis functions on a mesh
"""

from collections import namedtuple
from scipy.sparse import dok_matrix
import scipy.linalg as la
import numpy as np
import uuid

from openmodes.mesh import nodes_not_in_edge, shared_nodes

# A named tuple for holding the positive and negative triangles and nodes
# which are used by both RWG and loop-star basis functions
RWG = namedtuple('RWG', ('tri_p', 'tri_m', 'node_p', 'node_m'))

# TODO: routines to integrate over a triangle, with no scaling by area

def interpolate_triangle_mesh(mesh, tri_func, num_tri, xi_eta, flatten=True,
                              nodes = None):
    """Interpolate a function on a triangular mesh with linear basis functions
    on each face
    
    Parameters
    ----------
    flatten : boolean, optional
        Return a 2D array, instead of a 3D array
    """

    if nodes is None:
        nodes = mesh.nodes
    points_per_tri = len(xi_eta)

    r = np.empty((num_tri, points_per_tri, 3), mesh.nodes.dtype)
    vector_func = np.zeros((num_tri, points_per_tri, 3), tri_func.dtype)
    scalar_func = np.zeros((num_tri, points_per_tri), tri_func.dtype)

    for tri_count, node_nums in enumerate(mesh.polygons):
        for point_count, (xi, eta) in enumerate(xi_eta):
            # Barycentric coordinates of the observer
            zeta = 1.0 - eta - xi
    
            # Cartesian coordinates of the point
            r[tri_count, point_count] = xi*nodes[node_nums][0] + \
                                        eta*nodes[node_nums][1] + \
                                        zeta*nodes[node_nums][2]

            scalar_func[tri_count, point_count] = sum(tri_func[tri_count])

            for node_count in xrange(3):
                # Vector rho within the observer triangle
                rho = r[tri_count, point_count] - nodes[node_nums][node_count]

                vector_func[tri_count, point_count] += rho*tri_func[tri_count,
                             node_count]

    if flatten:
        r = r.reshape((num_tri*points_per_tri, 3))
        vector_func = vector_func.reshape((num_tri*points_per_tri, 3))
        scalar_func = scalar_func.reshape((num_tri*points_per_tri,))

    return r, vector_func, scalar_func

def inner_product_triangle_face(nodes):
    """Inner product of linear basis functions sharing the same triangle,
    integrated by sympy"""

    n0, n1, n2 = nodes
    res = np.empty((3, 3), np.float64)

    res[0, 0] = np.sum(n0**2/4 - n0*n1/4 - n0*n2/4 + n1**2/12 + n1*n2/12 + n2**2/12)
    res[0, 1] = np.sum(-n0**2/12 + n0*n1/4 - n0*n2/12 - n1**2/12 - n1*n2/12 + n2**2/12)
    res[0, 2] = np.sum(-n0**2/12 - n0*n1/12 + n0*n2/4 + n1**2/12 - n1*n2/12 - n2**2/12)
    res[1, 0] = res[0, 1]
    res[1, 1] = np.sum(n0**2/12 - n0*n1/4 + n0*n2/12 + n1**2/4 - n1*n2/4 + n2**2/12)
    res[1, 2] = np.sum(n0**2/12 - n0*n1/12 - n0*n2/12 - n1**2/12 + n1*n2/4 - n2**2/12)
    res[2, 0] = res[0, 2]
    res[2, 1] = res[1, 2]
    res[2, 2] = np.sum(n0**2/12 + n0*n1/12 - n0*n2/4 + n1**2/12 - n1*n2/4 + n2**2/4)

    return res


class LinearTriangleBasis(object):
    "An abstract base class for first order basis functions on triangles"

    def __init__(self):
        self.id = uuid.uuid4()

    def interpolate_function(self, linear_func, xi_eta = [[1.0/3, 1.0/3]], 
                             flatten=True, return_scalar=False, nodes=None,
                             scale_area=True):
        """Interpolate a function defined in RWG or loop-star basis over the 
        complete mesh

        Parameters
        ----------
        linear_func : ndarray
            The function defined over linear basis functions
        xi_eta : ndarray, optional
            The barycentric coordinates to interpolate on each triangle. If not
            specified, one point at the centre of each triangle will be used.
        flatten : bool, optional
            If false, data for each triangle will be identified by a specific
            index of the array, otherwise all points are identical
        nodes : array, optional
            Nodes of the Part, if they are not equal to the original nodes
            of the mesh.
        scale_area : boolean, optional
            Whether to include scaling by area. Normally this should be `True`,
            however when integrating it should be `False` as the area
            is already normalised out in the weights.

        Returns
        -------
        r_all : ndarray
            The interpolation points
        func_all : ndarray
            The vector function at each interpolation point
        """
        num_tri = len(self.mesh.polygons)

        vector_transform, _ = self.transformation_matrices
        tri_func = vector_transform.T.dot(linear_func) 
        tri_func = tri_func.reshape((num_tri, 3))
        
        if scale_area:
            tri_func /= 2*self.mesh.polygon_areas[:, None]

        r, vector_func, scalar_func = interpolate_triangle_mesh(self.mesh, 
                                            tri_func, num_tri, xi_eta, flatten,
                                            nodes)

        if return_scalar:
            return r, vector_func, scalar_func
        else:
            return r, vector_func

    def gram_matrix_faces(self):
        "Return the gram matrix defined between faces"
        num_tri = len(self.mesh.polygons)
        G = np.zeros((num_tri, 3, num_tri, 3), dtype=np.float64)
        for tri_count, (tri, area) in enumerate(zip(self.mesh.polygons, self.mesh.polygon_areas)):
            nodes = self.mesh.nodes[tri]
            G[tri_count, :, tri_count, :] = inner_product_triangle_face(nodes)/(2*area)
            # factor of 1/(2*area) is for second integration
            
        return G

    @property
    def gram_matrix(self):
        """Calculate the Gram matrix which is the inner product between each
        basis function

        Returns
        -------
        G : ndarray
            The Gram matrix, giving the inner product between each basis
            function            
        """
        try:
            return self.stored_gram
        except AttributeError:
            G = self.gram_matrix_faces()
            num_tri = len(self.mesh.polygons)
    
            # convert from faces to the appropriate basis functions
            vector_transform, _ = self.transformation_matrices
            self.stored_gram = vector_transform.dot(vector_transform.dot(G.reshape(3*num_tri, 3*num_tri)).T).T
            
            return self.stored_gram

    @property
    def gram_factored(self):
        """A an eigenvalue decomposed version of the Gram matrix
        
        Returns
        -------
        sqrt_val : ndarray
            The square root of each eigenvalue
        vec : ndarray
            Each column is a correctly normalised eigenvector such that the
            transpose of this matrix is equal to its inverse.
        """
        try:
            return self.stored_factored_gram
        except AttributeError:
            G = self.gram_matrix
            Gw, Gv = la.eigh(G)
            Gv /= np.sqrt(np.sum(Gv**2, axis=0))
            
            self.stored_factored_gram = (np.sqrt(Gw), Gv)
            return self.stored_factored_gram


class DivRwgBasis(LinearTriangleBasis):
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
        super(DivRwgBasis, self).__init__()
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
                       mesh.polygons[tri_p[basis_count]], edges[edge_count])[0]
            node_m[basis_count] = nodes_not_in_edge(
                       mesh.polygons[tri_m[basis_count]], edges[edge_count])[0]

        self.rwg = RWG(tri_p, tri_m, node_p, node_m)

    @property
    def transformation_matrices(self):
        """Returns the sparse transformation matrix to turn quantities
        defined on faces to loop-star basis
        
        For vector quantities, assumes that the face-based quantity has been 
        packed to a 2D array of size (n_basis*3, n_basis*3)

        For scalar quantities, assumes that the face-based quantity has been 
        packed to a 2D array of size (n_basis, n_basis)

        """
        
        try:
            return self.__vector_transform, self.__scalar_transform
        except AttributeError:
        
            num_basis = len(self)
            num_tri = len(self.mesh.polygons)
            # scalar_transform = np.zeros((num_basis, num_tri), np.float64)
            # vector_transform=np.zeros((num_basis, 3*num_tri), np.float64)
            scalar_transform = dok_matrix((num_basis, num_tri), np.float64)
            vector_transform = dok_matrix((num_basis, 3*num_tri), np.float64)
            
            for basis_count, (tri_p, tri_m, node_p, node_m) in enumerate(self):
                scalar_transform[basis_count, tri_p] = 1.0
                scalar_transform[basis_count, tri_m] = -1.0
    
                vector_transform[basis_count, tri_p*3+node_p] = 1.0
                vector_transform[basis_count, tri_m*3+node_m] = -1.0
    
            self.__vector_transform = vector_transform.tocsr()
            self.__scalar_transform = scalar_transform.tocsr()
            return self.__vector_transform, self.__scalar_transform
     
    def __len__(self):
        return len(self.rwg.tri_p)
        
    def __getitem__(self, index):
        return RWG(self.rwg.tri_p[index], self.rwg.tri_m[index], 
                   self.rwg.node_p[index], self.rwg.node_m[index])

def construct_stars(mesh, edges, triangles_shared_by_edges, sharing_count):
    """Construct star basis functions on a triangular mesh. The star
    corresponding to one triangle faces will be arbitrarily dropped"""
    
    num_tri = len(mesh.polygons)        

    shared_edge_indices = np.where(sharing_count == 2)[0]

    tri_p = [list() for _ in xrange(num_tri)]
    tri_m = [list() for _ in xrange(num_tri)]
    node_p = [list() for _ in xrange(num_tri)]
    node_m = [list() for _ in xrange(num_tri)]
    
    # Go through shared edges, and update both star-basis functions
    # to add the influence of this shared edge.
    for edge_count in shared_edge_indices:
        tri1, tri2 = triangles_shared_by_edges[edge_count]

        tri_p[tri1].append(tri1)
        tri_p[tri2].append(tri2)

        tri_m[tri1].append(tri2)
        tri_m[tri2].append(tri1)

        node1 = nodes_not_in_edge(mesh.polygons[tri1], 
                                  edges[edge_count])[0]
        node2 = nodes_not_in_edge(mesh.polygons[tri2], 
                                  edges[edge_count])[0]
        node_p[tri1].append(node1)
        node_p[tri2].append(node2)

        node_m[tri1].append(node2)
        node_m[tri2].append(node1)
 
    return RWG(tri_p[:-1], tri_m[:-1], node_p[:-1], node_m[:-1])

def construct_loop(loop_triangles, polygons):
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
            shared = shared_nodes(polygons[next_triangle], 
                                  polygons[current_triangle])
            if len(shared) == 2:
                break
        #raise ValueError("Cannot find adjoining triangle") # wrong!
        loop_triangles.pop(triangle_count)
        #ordered_triangles.append(current_triangle)
        
        # find the unshared nodes
        free_current = nodes_not_in_edge(polygons[current_triangle], shared)[0]
        free_next = nodes_not_in_edge(polygons[next_triangle], shared)[0]
        
        tri_p.append(current_triangle)
        tri_m.append(next_triangle)

        node_p.append(free_current)
        node_m.append(free_next)
        
        #done_triangles.append(current_triangle)
        current_triangle = next_triangle

    # now connect the loop with the first and last triangle
    shared = shared_nodes(polygons[first_triangle], polygons[current_triangle])

    free_current = nodes_not_in_edge(polygons[current_triangle], shared)[0]
    free_next = nodes_not_in_edge(polygons[first_triangle], shared)[0]

    tri_p.append(current_triangle)
    tri_m.append(first_triangle)

    node_p.append(free_current)
    node_m.append(free_next)

    return RWG(tri_p, tri_m, node_p, node_m)

class LoopStarBasis(LinearTriangleBasis):
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
        super(LoopStarBasis, self).__init__()
        self.mesh = mesh

        edges, triangles_shared_by_edges = mesh.get_edges(True)

        sharing_count = np.array([len(n) for n in triangles_shared_by_edges])

        if min(sharing_count) < 1 or max(sharing_count) > 2:
            raise ValueError("Mesh edges must be part of exactly 1 or 2" +
                             "triangles for loop-star basis functions")

        self.rwg_star = construct_stars(mesh, edges, triangles_shared_by_edges,
                                        sharing_count)

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

        triangles_sharing_nodes = mesh.triangles_sharing_nodes()

#        # eliminate nodes which don't belong to any edge at all 
#        # (e.g. the point at the centre when constructing an arc)
#        for node_count in xrange(num_nodes):
#            if len(triangles_sharing_nodes[node_count]) == 0:
#                print "eliminated node", node_count
#                inner_nodes.remove(node_count)


        #n_vertices = len(self.inner_nodes)+len(outer_nodes)
        #print "vertices", n_vertices
        #print "faces", len(triangles)
        #print "edges", len(all_edges)
        boundary_contours = 2-num_nodes+len(edges)-len(mesh.polygons)
        #print "separated contours", boundary_contours
        
        
        # Note that this would create one basis function for each inner 
        # node which may exceed the number of RWG degrees of freedom. In
        # this case arbitrary loops at the end of the list are dropped in
        # the conversion
        
        num_loops = len(inner_nodes) + boundary_contours - 1

        if num_loops > len(inner_nodes):
            # TODO: need to deal with holes in 2D (and 3D?) structures
            raise NotImplementedError
        
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
                
                this_loop = construct_loop(loop_triangles, mesh.polygons)
                loop_tri_p.append(this_loop[0])
                loop_tri_m.append(this_loop[1])
                loop_node_p.append(this_loop[2])
                loop_node_m.append(this_loop[3])
                
        self.rwg_loop = RWG(loop_tri_p, loop_tri_m, loop_node_p, loop_node_m)

    def __len__(self):
        return len(self.rwg_loop.tri_p)+len(self.rwg_star.tri_p)

    @property
    def num_loops(self):
        "The number of loops in the loop-star mesh"
        return len(self.rwg_loop.tri_p)

    @property
    def num_stars(self):
        "The number of stars in the loop-star mesh"
        return len(self.rwg_star.tri_p)

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

    @property
    def rwg(self):
        "Combine the loop and star lists"
        return RWG._make(a + b for a, b in zip(self.rwg_loop, self.rwg_star))

    @property
    def transformation_matrices(self):
        """Returns the sparse transformation matrices to turn quantities
        defined on faces to loop-star basis
        
        For vector quantities, assumes that the face-based quantity has been 
        packed to a 2D array of size (n_basis*3, n_basis*3)

        For scalar quantities, assumes that the face-based quantity has been 
        packed to a 2D array of size (n_basis, n_basis)

        """
        
        try:
            return self.__vector_transform, self.__scalar_transform
        except AttributeError:
        
            num_basis = len(self)
            num_tri = len(self.mesh.polygons)
            # scalar_transform = np.zeros((num_basis, num_tri), np.float64)
            # vector_transform=np.zeros((num_basis, 3*num_tri), np.float64)

            scalar_transform = dok_matrix((num_basis, num_tri), np.float64)
            vector_transform = dok_matrix((num_basis, 3*num_tri), np.float64)
            
            for basis_count, (tri_p, tri_m, node_p, node_m) in enumerate(self):
                # Assume that tri_p, tri_m, node_p, node_m are all of same 
                # length, which they must be for loop-star basis
                if basis_count >= self.num_loops:
                    for tri_n in tri_p:
                        scalar_transform[basis_count, tri_n] += 1.0
                    for tri_n in tri_m:
                        scalar_transform[basis_count, tri_n] += -1.0
                      
#                # TODO: slicing dok_matrix may be quite slow
#                norm = np.sqrt(scalar_transform[basis_count, :].multiply(
#                                scalar_transform[basis_count, :]).sum())
#                scalar_transform[basis_count, :] /= norm
#    
                for tri_n, node_n in zip(tri_p, node_p):
                    vector_transform[basis_count, tri_n*3+node_n] += 1.0
                    
                for tri_n, node_n in zip(tri_m, node_m):
                    vector_transform[basis_count, tri_n*3+node_n] += -1.0

#                norm = np.sqrt(vector_transform[basis_count, :].multiply(
#                                vector_transform[basis_count, :]).sum())
#                vector_transform[basis_count, :] /= norm
                    
            self.__vector_transform = vector_transform.tocsr()
            self.__scalar_transform = scalar_transform.tocsr()
    
            return self.__vector_transform, self.__scalar_transform            

cached_basis_functions = {}      
        
def get_basis_functions(mesh, basis_class):
    """Generate basis functions for a mesh. Performs caching, so that if an
    identical mesh has already been generated, the basis functions will
    not be unnecessarily duplicated

    Parameters
    ----------
    mesh : object
        The mesh to generate the basis functions for

    basis_class : class
        Which class of basis function should be created
    """

    # The following parameters are needed to determine if basis functions are
    # unique. Potentially this could be expanded to include non-affine mesh
    # transformations, or other parameters passed to the basis function
    # constructor
    unique_key = (mesh.id, basis_class) 

    if unique_key in cached_basis_functions:
        #print "Basis functions retrieved from cache"
        return cached_basis_functions[unique_key]
    else:
        #print "Calculating new basis functions"
        result = basis_class(mesh)
        cached_basis_functions[unique_key] = result
        return result
