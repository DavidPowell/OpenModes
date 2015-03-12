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

import logging
from collections import namedtuple
from scipy.sparse import lil_matrix
import numpy as np

from openmodes.mesh import nodes_not_in_edge, shared_nodes
from openmodes.helpers import (cached_property, inc_slice, Identified, memoize,
                               equivalence, MeshError)
from openmodes.integration import triangle_centres

# A named tuple for holding the positive and negative triangles and nodes
# which are used by both RWG and loop-star basis functions
RWG = namedtuple('RWG', ('tri_p', 'tri_m', 'node_p', 'node_m'))


def interpolate_triangle_mesh(mesh, tri_func, num_tri, integration_rule,
                              flatten=True, nodes=None):
    """Interpolate a function on a triangular mesh with linear basis functions
    on each face

    Parameters
    ----------
    mesh : LinearTriangleMesh
        The mesh which the triangle belong to
    tri_func : array
        The function to interpolate, one coefficient per triangle
    num_tri : integer
        The number of triangles
    integration_rule : object
        The integration over the triangle, in barycentric coordinates
    flatten : boolean, optional
        Return a 2D array, instead of a 3D array
    nodes : array(num_nodes, 3), optional
        Override the nodes positions defined in the mesh
    """

    if nodes is None:
        nodes = mesh.nodes
    points_per_tri = len(integration_rule)

    r = np.empty((num_tri, points_per_tri, 3), mesh.nodes.dtype)
    vector_func = np.zeros((num_tri, points_per_tri, 3), tri_func.dtype)
    scalar_func = np.zeros((num_tri, points_per_tri), tri_func.dtype)

    for tri_count, node_nums in enumerate(mesh.polygons):
        for point_count, (xi, eta) in enumerate(integration_rule.xi_eta):
            # Barycentric coordinates of the observer
            zeta = 1.0 - eta - xi

            # Cartesian coordinates of the point
            r[tri_count, point_count] = (xi*nodes[node_nums][0] +
                                         eta*nodes[node_nums][1] +
                                         zeta*nodes[node_nums][2])

            scalar_func[tri_count, point_count] = sum(tri_func[tri_count])

            for node_count in range(3):
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


class AbstractBasis(Identified):
    "An abstract class for arbitrary basis functions"

    def __init__(self):
        super(AbstractBasis, self).__init__()


class LinearTriangleBasis(AbstractBasis):
    "An abstract base class for first order basis functions on triangles"

    def __init__(self, mesh):
        super(LinearTriangleBasis, self).__init__()
        self.mesh = mesh

    def interpolate_function(self, linear_func,
                             integration_rule=triangle_centres,
                             flatten=True, return_scalar=False, nodes=None,
                             scale_area=True):
        """Interpolate a function defined in RWG or loop-star basis over the
        complete mesh

        Parameters
        ----------
        linear_func : ndarray
            The function defined over linear basis functions
        integration_rule : DunavantRule, optional
            The integration rule giving the barycentric coordinates to
            interpolate on each triangle. If not specified, one point at the
            centre of each triangle will be used.
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
                                            tri_func, num_tri,
                                            integration_rule, flatten, nodes)

        if return_scalar:
            return r, vector_func, scalar_func
        else:
            return r, vector_func

    @memoize
    def integration_points(self, nodes, integration_rule):
        """Find all the integration points for the basis functions in cartesian
        coordinates

        Parameters
        ---------
        mesh : TriangularSurfaceMesh
            The mesh on which to find all the points
        integration_rule: DunavantRule
            The barycentric coordinates within each triangle

        Returns
        -------
        r : ndarray[num_tri, num_points, 3]
            The cartesian coordinates within every triangle
        rho : ndarray[num_tri, 3, num_points]
            The vector value of the rooftop function for each of the 3 basis
            functions defined on each triangle
        """

        r = np.empty((len(self.mesh.polygons), len(integration_rule), 3),
                     self.mesh.nodes.dtype)
        rho = np.empty((len(self.mesh.polygons), 3, len(integration_rule), 3),
                       self.mesh.nodes.dtype)

        for tri_count, node_nums in enumerate(self.mesh.polygons):
            for point_count, (xi, eta) in enumerate(integration_rule.xi_eta):
                zeta = 1.0 - eta - xi

                r[tri_count, point_count] = (xi*nodes[node_nums[0]] +
                                             eta*nodes[node_nums[1]] +
                                             zeta*nodes[node_nums[2]])

                for node_count in range(3):
                    # Vector rho within the observer triangle
                    rho[tri_count, node_count, point_count] = r[tri_count, point_count] - nodes[node_nums][node_count]

        return r, rho

    def weight_function(self, func, integration_rule, nodes, n_cross=False):
        """Weight a function (e.g. a source field) by integrating it over this
        set of basis functions

        Parameters
        ----------
        func : function(r)
            A function of the coordinates which returns the field value at
            each coordinate point. Must be able to accept r as a 3d array
        integration_rule : DunavantRule
            An object with the barycentric coordinates and weights for
            integration over a triangle
        nodes : array[num_modes, 3]
            The location of the triangle nodes for the part of interest
        n_cross : boolean, optional
            If True, take the cross product of the surface normal with the
            vector function

        Returns
        -------
        tested_func : ndarray[num_basis]
            The function tested over each basis function
        """

        # This implementation uses vector operations, making it relatively
        # fast, but somewhat memory inefficient
        r, rho = self.integration_points(nodes, integration_rule)
        func_points = func(r)  # dim[num_tri, num_points, 3]
        if n_cross:
            func_points = np.cross(self.mesh.surface_normals[:, None, :],
                                   func_points)
        func_rho = np.sum(func_points[:, None, :, :]*rho, axis=3)
        # func_rho has dim[num_tri, 3, num_points]
        func_tri = np.dot(func_rho, integration_rule.weights)  # dim[num_tri, 3]
        vector_transform, _ = self.transformation_matrices
        return vector_transform.dot(func_tri.flatten())



    @cached_property
    def gram_matrix(self):
        """Calculate the Gram matrix which is the inner product between each
        basis function

        Returns
        -------
        G : ndarray
            The Gram matrix, giving the inner product between each basis
            function
        """
        num_tri = len(self.mesh.polygons)
        G = np.zeros((num_tri, 3, num_tri, 3), dtype=np.float64)
        for tri_count, (tri, area) in enumerate(zip(self.mesh.polygons,
                                                    self.mesh.polygon_areas)):
            nodes = self.mesh.nodes[tri]
            G[tri_count, :, tri_count, :] = inner_product_triangle_face(nodes)/(2*area)
            # factor of 1/(2*area) is for second integration

        # convert from faces to the appropriate basis functions
        vector_transform, _ = self.transformation_matrices
        return vector_transform.dot(vector_transform.dot(G.reshape(3*num_tri, 3*num_tri)).T).T


class DivRwgBasis(LinearTriangleBasis):
    """Divergence-conforming RWG basis functions

    The simplest basis functions which can be defined on a triangular mesh.

    The basis functions are defined in a coordinate independent manner, so that
    they can be re-used for the same mesh placed in a different location.
    """

    def __init__(self, mesh):
        """Generate basis functions for a particular mesh. Note that the mesh
        will be referenced, so it should not be modified after generating the
        basis functions.
        """
        super(DivRwgBasis, self).__init__(mesh)
        self.canonical_basis = DivRwgBasis

        edges, triangles_shared_by_edges = mesh.get_edges(True)

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
        self.sections = (num_basis,)

        logging.info("Constructing %d RWG basis functions over %d faces"
                     % (num_basis, len(mesh.polygons)))

    @cached_property
    def transformation_matrices(self):
        """Returns the sparse transformation matrix to turn quantities
        defined on faces to loop-star basis

        For vector quantities, assumes that the face-based quantity has been
        packed to a 2D array of size (n_basis*3, n_basis*3)

        For scalar quantities, assumes that the face-based quantity has been
        packed to a 2D array of size (n_basis, n_basis)
        """

        num_basis = len(self)
        num_tri = len(self.mesh.polygons)
        # scalar_transform = np.zeros((num_basis, num_tri), np.float64)
        # vector_transform=np.zeros((num_basis, 3*num_tri), np.float64)
        scalar_transform = lil_matrix((num_basis, num_tri))
        vector_transform = lil_matrix((num_basis, 3*num_tri))

        for basis_count, (tri_p, tri_m, node_p, node_m) in enumerate(self):
            scalar_transform[basis_count, tri_p] = 1.0
            scalar_transform[basis_count, tri_m] = -1.0

            vector_transform[basis_count, tri_p*3+node_p] = 1.0
            vector_transform[basis_count, tri_m*3+node_m] = -1.0

        return vector_transform.tocsr(), scalar_transform.tocsr()

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

    tri_p = [list() for _ in range(num_tri)]
    tri_m = [list() for _ in range(num_tri)]
    node_p = [list() for _ in range(num_tri)]
    node_m = [list() for _ in range(num_tri)]

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
    current_triangle = loop_triangles.pop()
    first_triangle = current_triangle

    while len(loop_triangles) > 0:
        for triangle_count, next_triangle in enumerate(loop_triangles):
            shared = shared_nodes(polygons[next_triangle],
                                  polygons[current_triangle])
            if len(shared) == 2:
                break
        loop_triangles.pop(triangle_count)

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
        super(LoopStarBasis, self).__init__(mesh)
        self.canonical_basis = LoopStarBasis

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
        inner_nodes = set(range(num_nodes)) - outer_nodes

        triangles_sharing_nodes = mesh.triangles_sharing_nodes()

        # Note that this would create one basis function for each inner
        # node which may exceed the number of RWG degrees of freedom. In
        # this case arbitrary loops at the end of the list are dropped in
        # the conversion

        num_loops = len(edges) - len(unshared_edges) - self.num_stars

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

        if num_loops > len(inner_nodes):
            # The structure has internal holes, so additional loops are needed
            node_sets = equivalence(unshared_edges)

            if len(node_sets) < num_loops-len(inner_nodes):
                raise MeshError("Unable to find a full set of loops")

            # For each additional loop needed, find the set of triangles which
            # loop around one of the edges
            for loop_number in range(num_loops-len(inner_nodes)):
                needed_nodes = node_sets[loop_number]
                loop_triangles = set()
                for node_number in needed_nodes:
                    loop_triangles.update(triangles_sharing_nodes[node_number])

                this_loop = construct_loop(list(loop_triangles), mesh.polygons)
                loop_tri_p.append(this_loop[0])
                loop_tri_m.append(this_loop[1])
                loop_node_p.append(this_loop[2])
                loop_node_m.append(this_loop[3])

        self.rwg_loop = RWG(loop_tri_p, loop_tri_m, loop_node_p, loop_node_m)

        self.sections = (num_loops, self.num_stars)

        logging.info("Constructing %d loop-star basis functions\n"
                     "%d loops\n%d stars\n%d faces\n%d edges\n"
                     "%d unshared_edges\n"
                     "%d nodes on boundary"
                     % (len(self), num_loops, self.num_stars,
                        len(mesh.polygons), len(edges), len(unshared_edges),
                        len(outer_nodes)))

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

    @property
    def loop_range(self):
        """The range of indicies into any vector corresponding to the loops of
        the basis function"""
        return slice(0, self.num_loops)

    @property
    def star_range(self):
        """The range of indicies into any vector corresponding to the stars of
        the basis function"""
        return slice(self.num_loops, len(self))

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

    @cached_property
    def transformation_matrices(self):
        """Returns the sparse transformation matrices to turn quantities
        defined on faces to loop-star basis

        For vector quantities, assumes that the face-based quantity has been
        packed to a 2D array of size (n_basis*3, n_basis*3)

        For scalar quantities, assumes that the face-based quantity has been
        packed to a 2D array of size (n_basis, n_basis)

        """

        num_basis = len(self)
        num_tri = len(self.mesh.polygons)
        # scalar_transform = np.zeros((num_basis, num_tri), np.float64)
        # vector_transform = np.zeros((num_basis, 3*num_tri), np.float64)

        scalar_transform = lil_matrix((num_basis, num_tri))
        vector_transform = lil_matrix((num_basis, 3*num_tri))

        for basis_count, (tri_p, tri_m, node_p, node_m) in enumerate(self):
            # Assume that tri_p, tri_m, node_p, node_m are all of same
            # length, which they must be for loop-star basis
            if basis_count >= self.num_loops:
                for tri_n in tri_p:
                    scalar_transform[basis_count, tri_n] += 1.0
                for tri_n in tri_m:
                    scalar_transform[basis_count, tri_n] += -1.0

            for tri_n, node_n in zip(tri_p, node_p):
                vector_transform[basis_count, tri_n*3+node_n] += 1.0

            for tri_n, node_n in zip(tri_m, node_m):
                vector_transform[basis_count, tri_n*3+node_n] += -1.0

        return vector_transform.tocsr(), scalar_transform.tocsr()


class BasisContainer(object):
    """A container to hold the basis functions for a simulation, constructing
    them on the fly as they are required"""

    def __init__(self, basis_class, default_args=dict()):
        """Set the class of the basis functions, and the default arguments to
        pass when constructing them"""
        self.basis_class = basis_class
        self.args = dict()
        self.default_args = default_args
        self.cached_basis = {}

    def set_args(self, part, args):
        "Override the default basis function arguments for a particular part"
        self.args[part] = dict(args)
        pass

    def __getitem__(self, part):
        """Return the basis functions for a particular part, constructing them
        if they do not already exist"""
        args = self.args.get(part, self.default_args)
        unique_key = (part.mesh.id, self.basis_class, frozenset(args.items()))

        try:
            return self.cached_basis[unique_key]
        except KeyError:
            logging.debug("Constructing basis functions of class %s for "
                          "mesh %s with args %s"
                          % (self.basis_class, part.mesh, args))
            result = self.basis_class(part.mesh, **args)
            self.cached_basis[unique_key] = result
            return result


class CombinedBasis(AbstractBasis):
    "A set of basis functions which have been combined together"

    def __init__(self, basis_list):
        super(CombinedBasis, self).__init__()
        self.basis_list = basis_list
        self.sections = (len(self),)
        self.canonical_basis = DivRwgBasis

    @cached_property
    def gram_matrix(self):
        "Calculate the total Gram matrix of the complete system"
        total_size = len(self)

        G_tot = np.zeros((total_size, total_size))

        row_offset = 0
        for basis in self.basis_list:
            row_size = len(basis)
            G_tot[row_offset:row_offset+row_size,
                  row_offset:row_offset+row_size] = basis.gram_matrix

            row_offset += row_size

        return G_tot

    def __len__(self):
        return sum(len(basis) for basis in self.basis_list)


class CombinedLoopStarBasis(CombinedBasis):
    "A set of loop-star basis functions which have been combined together"

    def __init__(self, basis_list):
        super(CombinedLoopStarBasis, self).__init__(basis_list)
        self.sections = (self.num_loops, len(self)-self.num_loops)
        self.canonical_basis = LoopStarBasis

    @property
    def num_loops(self):
        "The number of loops in the loop-star basis"
        return sum(basis.num_loops for basis in self.basis_list)

    @cached_property
    def gram_matrix(self):
        "Calculate the total Gram matrix of the complete system"
        total_size = len(self)
        num_loops = self.num_loops

        G_tot = np.zeros((total_size, total_size))

        loop_range = slice(0, 0)
        star_range = slice(num_loops, num_loops)

        for basis in self.basis_list:
            loop_range = inc_slice(loop_range, basis.num_loops)
            star_range = inc_slice(star_range, basis.num_stars)

            G = basis.gram_matrix
            # assumes symmetric weighting and testing
            if loop_range.stop > loop_range.start:
                G_tot[loop_range, loop_range] = G[basis.loop_range, basis.loop_range]
                G_tot[loop_range, star_range] = G[basis.loop_range, basis.star_range]
                G_tot[star_range, loop_range] = G[basis.star_range, basis.loop_range]
            G_tot[star_range, star_range] = G[basis.star_range, basis.star_range]

        return G_tot

CACHED_COMBINED_BASIS = {}


def get_combined_basis(basis_list):
    """Generate combined basis functions from several existing. Performs
    caching, so that if an identical combination of basis functions already
    exists it will not be unnecessarily duplicated

    Parameters
    ----------
    basis_list : list
        A list of the basis functions in all parts to be combined
    """

    # The following parameters are needed to determine if basis functions are
    # unique. Potentially this could be expanded to include non-affine mesh
    # transformations, or other parameters passed to the basis function
    # constructor
    unique_key = tuple(basis.id for basis in basis_list)

    if unique_key in CACHED_COMBINED_BASIS:
        #print "Combined basis functions retrieved from cache"
        return CACHED_COMBINED_BASIS[unique_key]
    else:
        if all(isinstance(basis, LoopStarBasis) for basis in basis_list):
            result = CombinedLoopStarBasis(basis_list)
        else:
            result = CombinedBasis(basis_list)

        CACHED_COMBINED_BASIS[unique_key] = result
        return result
