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
Routines and classes for storing a mesh, and querying it to calculate
derived quantities
"""

import logging
import numpy as np
from openmodes.helpers import Identified, cached_property


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


class TriangularSurfaceMesh(Identified):
    """A physical part represented by a surface mesh

    A part should correspond to the smallest unit of interest in the
    simulation and must not be connected with or overlap any other object

    Once created a mesh cannot be modified, as it is designed to be an
    unchanging reference object

    The internal member `nodes` contains the original node locations. However,
    the mesh object can also refer to the connectivity of a set of nodes which
    have been transformed, and is held externally.

    For external nodes, the areas and lengths of the mesh elements are
    correct if the nodes have been translated or rotated, but not if they
    have been scaled or sheared.

    """

    polygon_name = 'triangles'

    def __init__(self, raw_mesh, scale=None):
        """
        Parameters
        ----------
        raw_mesh : dict
            A dictionary, which must contain at least the following members
            nodes : ndarray
                The nodes making up the object
            triangles : ndarray
                The node indices of triangles making up the object
        scale : real, optional
            A scaling factor to apply to all nodes, in case conversion between
            units is required.

        The internal storage of `nodes` and `polygons` will be put into
        fortran order as these arrays will be passed to fortran routines
        """

        super(TriangularSurfaceMesh, self).__init__()

        self.nodes = np.asfortranarray(raw_mesh['nodes'])
        if scale is not None:
            self.nodes *= scale

        self.polygons = np.asfortranarray(raw_mesh[self.polygon_name])
        self.nodes.setflags(write=False)
        self.polygons.setflags(write=False)

        try:
            self.physical_name = raw_mesh['physical_name']
        except KeyError:
            self.physical_name = None

        logging.info('Creating triangular mesh\n%d nodes\n%d triangles' %
                     (len(self.nodes), len(self.polygons)))

#    def __repr__(self):
#        return "Nodes

    # TODO: memoize these lookup methods?

    def get_edges(self, get_shared_triangles=False):
        """Calculate the edges in the mesh, and optionally return the triangles
        which share each edge

        Parameters
        ----------
        get_shared_triangles : boolean, optional
            Whether to return information about which triangles share each edge

        Returns
        -------
        edges : array[num_edges, 2] of int
            each edge contains the incides of the nodes
        triangles_shared_by_edges : list of lists
            the triangles that share each edge, indexed by the edges
        """

        all_edges = set()
        shared_edges = []
        triangles_shared_by_edges = dict()

        for count, t_nodes in enumerate(self.polygons):

            # edges are represented as sets to avoid ambiguity of order
            triangle_edges = [frozenset((t_nodes[0], t_nodes[1])),
                              frozenset((t_nodes[0], t_nodes[2])),
                              frozenset((t_nodes[1], t_nodes[2]))]

            for edge in triangle_edges:
                if edge in all_edges:
                    shared_edges.append(edge)
                    triangles_shared_by_edges[edge].append(count)
                else:
                    all_edges.add(edge)
                    triangles_shared_by_edges[edge] = [count]

        edges_array = np.empty((len(all_edges), 2), np.int)
        sharing_list = []
        for edge_count, edge in enumerate(all_edges):
            edges_array[edge_count] = tuple(edge)
            sharing_list.append(triangles_shared_by_edges[edge])

        if get_shared_triangles:
            return edges_array, sharing_list
        else:
            return edges_array

    def triangles_sharing_nodes(self):
        """Return a set of all the triangles which share each node"""

        polygons = [set() for _ in range(len(self.nodes))]

        for count, t_nodes in enumerate(self.polygons):
            # tell each node that it is a part of this triangle
            for node in t_nodes:
                polygons[node].add(count)

        return polygons

    @property
    def shortest_edge(self):
        """The shortest edge in the mesh"""
        return self.edge_lens.min()

    @property
    def max_distance(self):
        """The furthest distance between any two nodes in this mesh"""
        return np.sqrt(np.sum((self.nodes[:, None, :] -
                       self.nodes[None, :, :])**2, axis=2)).max()

    def fast_size(self):
        """A fast estimate of the size of the mesh, based on the maximum
        distance along any of the three cartesian axes"""
        return max(np.max(self.nodes, axis=0)-np.min(self.nodes, axis=0))

    @property
    def polygon_areas(self):
        """The area of each triangle in the mesh"""
        areas = np.empty(len(self.polygons), np.float64)

        # calculate all the edges in the mesh
        for count, t_nodes in enumerate(self.polygons):
            # calculate the area of each triangle
            vec1 = self.nodes[t_nodes[1]]-self.nodes[t_nodes[0]]
            vec2 = self.nodes[t_nodes[2]]-self.nodes[t_nodes[0]]
            areas[count] = 0.5*np.sqrt(sum(np.cross(vec1, vec2)**2))

        return areas

    @cached_property
    def surface_normals(self):
        """The surface normal of each triangle in the mesh"""
        normals = np.empty((len(self.polygons), 3), np.float64)

        # calculate all the edges in the mesh
        for count, t_nodes in enumerate(self.polygons):
            vec1 = self.nodes[t_nodes[1]]-self.nodes[t_nodes[0]]
            vec2 = self.nodes[t_nodes[2]]-self.nodes[t_nodes[0]]
            normal = np.cross(vec1, vec2)
            normals[count] = normal/np.sqrt(np.dot(normal, normal))

        return normals

    @property
    def edge_lens(self):
        """The length of each triangle's edges"""
        # indexing: triangle, vertex_num, x/y/z
        vertices = self.nodes[self.polygons]

        # each edge is numbered according to its opposite node
        return np.sqrt(np.sum((np.roll(vertices, 1, axis=1) -
                       np.roll(vertices, 2, axis=1))**2, axis=2))

    @cached_property
    def closed_surface(self):
        """Whether the mesh represents a closed surface corresponding to a
        solid object"""
        edges, triangles_shared_by_edges = self.get_edges(True)

        sharing_count = np.array([len(n) for n in triangles_shared_by_edges])

        return np.all(sharing_count == 2)


def combine_mesh(meshes, nodes=None):
    """Combine several meshes into one, optionally overriding their node
    locations"""

    mesh_class = type(meshes[0])
    all_nodes = []
    all_polygons = []
    node_offset = 0

    # using existing mesh nodes if not overridden
    if nodes is None:
        nodes = [mesh.nodes for mesh in meshes]

    for mesh, current_nodes in zip(meshes, nodes):
        if type(mesh) != mesh_class:
            raise TypeError("Cannot combine meshes of different types")
        all_nodes.append(current_nodes)
        all_polygons.append(mesh.polygons+node_offset)
        node_offset += len(current_nodes)

    raw_mesh = {'nodes': np.vstack(all_nodes),
                mesh_class.polygon_name: np.vstack(all_polygons)}

    return mesh_class(raw_mesh)
