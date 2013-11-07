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
from collections import defaultdict

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


edge_type = 1
triangle_type = 2
point_type = 15
# the number of nodes in different gmsh element types which may be encountered
gmsh_element_nodes = {edge_type: 2, triangle_type: 3, point_type: 1}

element_name_mapping = {"edges" : edge_type, "triangles" : triangle_type,
                        "points" : point_type}

def read_mesh(filename, returned_elements = ("edges", "triangles")):
    """Read a gmsh binary mesh file
    
    Parameters
    ----------
    filename : string
        the full name of the gmesh meshed file
    returned_elements : tuple, optional
        A tuple of string saying which types of elements are desired
        
    Returns
    -------
    raw_mesh : dict
        Containing the following
        nodes : ndarray
            all nodes referred to by this geometric entity
        triangles : ndarray
            the indices of nodes belonging to each triangle
        edges : ndarray
    
    Node references are set to be zero based indexing as per python standard 
    
    Currently assumes gmsh binary format 2.2, little endian
    as defined in http://www.geuz.org/gmsh/doc/texinfo/#MSH-binary-file-format    
    
    """
    
    wanted_element_types = set(element_name_mapping[n] for n in returned_elements)
    
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

        #element_count = defaultdict(int)
        
        object_nodes = defaultdict(set)
        object_elements = defaultdict(lambda : defaultdict(list))
        #elements = defaultdict(list)
        
        # currently we are only interested in the triangle elements
        # so skip over all others
        for _ in xrange(num_elements):
            element_type, num_elem_in_group, num_tags = struct.unpack('=iii', f.read(12))
            
            num_nodes_in_elem = gmsh_element_nodes[element_type]
            assert(num_tags >= 2) # need to have the elementary geometry tag
            
            element_bytes = 4*(1 + num_tags + num_nodes_in_elem)

            elem_format = "=i" + "i"*num_tags + "i"*num_nodes_in_elem

            element_data = f.read(num_elem_in_group*element_bytes)

            # Avoid reading in unwanted element types. This is important for
            # getting rid of nodes which are not a part of any triangle
            if element_type not in wanted_element_types:
                continue
            
            # iterate over all elements within the same header block
            for these_elements_count in xrange(num_elem_in_group):
                this_element = struct.unpack(elem_format, 
                    element_data[these_elements_count*element_bytes:(these_elements_count+1)*element_bytes])
                    
                # assumes that the required default tags are used
                physical_entity = this_element[1]
                #geometric_entity = this_element[2]
                #print physical_entity, geometric_entity
                    
                # NB: conversion to python 0-based indexing is done here
                element_nodes = np.array(this_element[-num_nodes_in_elem:])-1
                #elements[element_type].append(element_nodes)

                #object_elements[physical_entity][element_type].append(element_count[element_type])
                object_elements[physical_entity][element_type].append(element_nodes)
                object_nodes[physical_entity].update(element_nodes)
        
        f.readline()
        assert(f.readline().strip() == "$EndElements")
        
    return_vals = []        
    
    # Go through each entity, and work out which nodes belong to it. Nodes are
    # renumbered, so elements are updated to reflect new numbering
    for obj_nodes, obj_elements in zip(object_nodes.itervalues(), 
                                        object_elements.itervalues()):
        orig_nodes = np.sort(list(obj_nodes))
        new_nodes = np.zeros(len(nodes), np.int)
        for node_count, node in enumerate(orig_nodes):
            new_nodes[node] = node_count
        
        this_part = {'nodes': nodes[orig_nodes]}
        
        for elem_name in returned_elements:
            returned_type = element_name_mapping[elem_name]
            which_elems = new_nodes[np.array(obj_elements[returned_type])]
            this_part[elem_name] = which_elems
            
        return_vals.append(this_part)
        
    return tuple(return_vals)

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
    