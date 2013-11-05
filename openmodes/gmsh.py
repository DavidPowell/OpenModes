# -*- coding: utf-8 -*-
"""
Mesh routines for MOM code, for loading and manipulating raw meshes
before they are processed to have basis functions etc

Created on Fri Apr 27 10:07:16 2012

@author: dap124
"""

import subprocess
import tempfile
import os.path as osp
import struct
import numpy as np
import collections

# the minimum version of gmsh required
min_version = (2, 5, 0)

def mesh_geometry(filename, mesh_tol=None, binary=True, dirname=None):
    """Call gmsh to surface mesh a geometry file with a specified maximum tolerance
    
    Parameters    
    ----------
    filename : string
        the name of the file to be meshed
    mesh_tol : number, optional
        override the maximum mesh tolerance distance of all edges
    binary : boolean, optional
        (default True) output a binary file
    dirname : string, optional
        The location in which to create the mesh file. If unspecified a
        temporary directory will be created
        
    Returns
    -------
    meshname : string
        the full path of the .msh file
        
    This routine instructs gmsh to use algorithm 1 for 2D meshing, which seems
    to yield the most consistent and uniform mesh.
    """
    
    if not osp.exists(filename):
        raise ValueError("Geometry file %s not found" % filename)
    
    if dirname is None:
        dirname = tempfile.mkdtemp()

    meshname = osp.join(dirname, osp.splitext(osp.basename(filename))[0]+".msh")

    call_options = ['gmsh', filename, '-2', '-o', meshname, 
                     '-string', 'Mesh.Algorithm=1;']
                     
    if mesh_tol is not None:
        call_options += ['-clmax', '%f' % mesh_tol]
        
    if binary:
        call_options += ['-bin']
    proc = subprocess.Popen(call_options, stdout=subprocess.PIPE)
    
    # run the process and read in stderr and stdout streams
    # currently these are just suppressed
    #stdouttxt, stderrtxt = proc.communicate()
    _, _ = proc.communicate()
    
    #print stdouttxt, stderrtxt
    return meshname

# the number of nodes in different gmsh element types which may be encountered
gmsh_element_nodes = {1: 2, 2: 3, 15: 1}

triangle_type = 2

def read_mesh(filename, split_geometry=True):
    """Read a gmsh mesh file
    
    Will only read in triangle elements
    
    Parameters
    ----------
    filename : string
        the full name of the gmesh meshed file
    split_geometry : boolean, optional
        whether to split the geometry
        
    Returns
    -------
    nodes : ndarray
        all nodes (may include some which are not referenced)
    triangles : ndarray
        the indices of nodes belonging to each triangle
    
    If the geometry is split, then nodes and triangles will be repeated for
    each geometric object within the mesh file, e.g.
    `((nodes1, triangles1), (nodes2, triangles2), ...)`

    
    All nodes are included, even if unused
    Assumes that the default geometric tags are used.
    Node references are set to be zero based indexing as per python standard 
    
    Currently assumes gmsh binary format 2.2, little endian
    
    """
    with open(filename, "rb") as f:
        
        # check the header version
        assert(f.readline().strip() == "$MeshFormat")
        assert(f.readline().split() == ['2.2', '1', '8'])
        
        # check the endianness of the file
        assert(struct.unpack('=i', f.read(4))[0] == 1)

        f.readline()
        assert(f.readline().strip() == "$EndMeshFormat")
        
        # read in the nodes
        assert(f.readline().strip() == "$Nodes")
        num_nodes = int(f.readline())
        
        nodes = np.empty((num_nodes, 3), np.float32)
        for node_count in xrange(num_nodes):
            this_node = struct.unpack('=iddd', f.read(28))
            assert(this_node[0] == node_count+1)
            nodes[node_count] = this_node[1:]
    
        f.readline()
        assert(f.readline().strip() == "$EndNodes")
        
        # read in the elements
        assert(f.readline().strip() == "$Elements")
        num_elements = int(f.readline())

        # allocate the maximum number of triangles which there could be,
        # later we will resize to reduce this
        triangles = np.empty((num_elements, 3), np.int32)

        element_type_count = collections.defaultdict(int)
        
        triangle_count = 0        
        
        object_triangles = collections.defaultdict(list)
        object_nodes = collections.defaultdict(set)
        
        # currently we are only interested in the triangle elements
        # so skip over all others
        for _ in xrange(num_elements):
            element_type, num_element_type, num_tags = struct.unpack('=iii', f.read(12))
            assert(num_tags >= 2) # need to have the elementary geometry tag
            element_type_count[element_type] += num_element_type
            
            element_bytes = 4*(1 + num_tags + gmsh_element_nodes[element_type])
            elem_format = "=i" + "i"*num_tags + "iii"

            element_data = f.read(num_element_type*element_bytes)
            if element_type == triangle_type:
                # iterate over all elements within the same header block
                for these_elements_count in xrange(num_element_type):
                    this_triangle = struct.unpack(elem_format, 
                        element_data[these_elements_count*element_bytes:(these_elements_count+1)*element_bytes])
                        
                    # NB: conversion to python 0-based indexing is done here
                    triangle_nodes = np.array(this_triangle[-3:])-1
                    triangles[triangle_count] = triangle_nodes

                    # assumes that the default tags are used, ie the elementary
                    # geometric object is the second tag
                    object_triangles[this_triangle[2]].append(triangle_count)
                    object_nodes[this_triangle[2]].update(triangle_nodes)
                    
                    triangle_count += 1
        
        triangles.resize((element_type_count[triangle_type], 3))

        f.readline()
        assert(f.readline().strip() == "$EndElements")
        
    if split_geometry:
        # for the split geometry, return ((nodes1, triangles1),
        # (nodes2, triangles2) ...)
        
        return_vals = []        
        
        # Go through each sub-object, and work out which nodes and triangles
        # belong to it. Nodes will be renumbered, so update triangles
        # accordingly
        for obj_nodes, obj_triangles in zip(object_nodes.itervalues(), 
                                            object_triangles.itervalues()):
            orig_nodes = np.sort(list(obj_nodes))
            new_nodes = np.zeros(len(nodes), np.int)
            for node_count, node in enumerate(orig_nodes):
                new_nodes[node] = node_count
            
            which_triangles = new_nodes[triangles[np.sort(obj_triangles)]]
            return_vals.append((nodes[orig_nodes], which_triangles))
        
        return tuple(return_vals)
    else:
        return nodes, triangles

def check_installed():
    "Check if a supported version of gmsh is installed"
    call_options = ['gmsh', '--version']

    try:
        proc = subprocess.Popen(call_options, stderr=subprocess.PIPE)
    except OSError:
        raise ValueError("gmsh not found")
        
    ver = tuple([int(x) for x in proc.stderr.readline().split(".")])
    
    if ver < min_version:
        raise ValueError("gmsh version %d.%d.%d found, "+
            "but version %d.%d.%d required" % (ver+min_version))
           
check_installed()

if __name__ == "__main__":
    file_name = mesh_geometry("geometry/asymmetric_ring.geo", 0.4e-3)
    print read_mesh(file_name, split_geometry=True)
    