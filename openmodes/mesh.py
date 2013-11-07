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

class TriangularSurfaceMesh(object):
    """A physical part represented by a surface mesh

    A part should correspond to the smallest unit of interest in the 
    simulation and must not be connected with or overlap any other object

    Once created a mesh cannot be modified, as it is designed to be an
    unchanging reference object
    """

    def __init__(self, raw_mesh):
        """
        Parameters
        ----------
        raw_mesh : dict
            A dictionary, which must contain at least the following members
            nodes : ndarray
                The nodes making up the object
            triangles : ndarray
                The node indices of triangles making up the object
            
        The internal storage of `nodes` and `triangle_nodes` will be put into
        fortran order as these arrays will be passed to fortran routines
        """
        
        self.nodes = np.asfortranarray(raw_mesh['nodes'])
        #N_nodes = len(self.nodes)

        self.triangle_nodes = np.asfortranarray(raw_mesh['triangles'])
        #N_tri = len(self.triangle_nodes)

        #self.edges = raw_mesh['edges']

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
        edges : list of frozenset
            each edge is a frozenset containing the incides of the nodes
        triangles_shared_by_edges : list of lists
            the triangles that share each edge, indexed by the edges
        """
        
        all_edges = set()
        shared_edges = [] 
        triangles_shared_by_edges = dict()
        
        for count, t_nodes in enumerate(self.triangle_nodes):    
    
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
        #edges_list = []
        sharing_list = []
        for edge_count, edge in enumerate(all_edges):
            edges_array[edge_count] = tuple(edge)
            #edges_list.append(edge)
            sharing_list.append(triangles_shared_by_edges[edge])
          
        if get_shared_triangles:
            #return all_edges, triangles_shared_by_edges
            return edges_array, sharing_list
        else:
            return edges_array

    def triangles_sharing_nodes(self):
        """Return a set of all the triangles which share each node"""
        
        triangle_nodes = [set() for _ in xrange(len(self.nodes))]        
        
        for count, t_nodes in enumerate(self.triangle_nodes):    
            # tell each node that it is a part of this triangle
            for node in t_nodes:
                triangle_nodes[node].add(count)       
                
        return triangle_nodes


    @property
    def shortest_edge(self):
        """The shortest edge in the mesh"""
        return self.triangle_lens.min()
     
    @property
    def triangle_areas(self):
        """The area of each triangle in the mesh"""
        areas = np.empty(self.N_tri, np.float64)#, order="F")

        # calculate all the edges in the mesh
        for count, t_nodes in enumerate(self.triangle_nodes):    
            # calculate the area of each triangle
            vec1 = self.nodes[t_nodes[1]]-self.nodes[t_nodes[0]]
            vec2 = self.nodes[t_nodes[2]]-self.nodes[t_nodes[0]]
            areas[count] = 0.5*np.sqrt(sum(np.cross(vec1, vec2)**2))
     
        return areas
    
    @property
    def triangle_lens(self):
        """The length of each triangle's edges"""
        # indexing: triangle, vertex_num, x/y/z
        vertices = self.nodes[self.triangle_nodes]

        # each edge is numbered according to its opposite node
        return np.sqrt(np.sum((np.roll(vertices, 1, axis=1) - 
                            np.roll(vertices, 2, axis=1))**2, axis=2))    

         