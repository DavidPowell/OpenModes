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
Routines for using `gmsh` to mesh a geometry, and load the resulting mesh into
a format which OpenModes recognises.
"""

import subprocess
import tempfile
import os.path as osp
import struct
import numpy as np
from collections import defaultdict

from openmodes.helpers import MeshError

# the minimum version of gmsh required
MIN_VERSION = (2, 5, 0)

def mesh_geometry(filename, mesh_tol=None, binary=True, dirname=None):
    """Call gmsh to surface mesh a geometry file with a specified maximum 
    tolerance
    
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
        raise MeshError("Geometry file %s not found" % filename)
    
    if dirname is None:
        dirname = tempfile.mkdtemp()

    meshname = osp.join(dirname, osp.splitext(osp.basename(filename))[0] 
                        + ".msh")

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


EDGE_TYPE = 1
TRIANGLE_TYPE = 2
POINT_TYPE = 15
# the number of nodes in different gmsh element types which may be encountered
GMSH_ELEMENT_NODES = {EDGE_TYPE: 2, TRIANGLE_TYPE: 3, POINT_TYPE: 1}

ELEMENT_NAME_MAPPING = {"edges" : EDGE_TYPE, "triangles" : TRIANGLE_TYPE,
                        "points" : POINT_TYPE}

def read_nodes(file_handle):
    "Read in the nodes of a gmsh file"
    num_nodes = int(file_handle.readline())
    
    nodes = np.empty((num_nodes, 3), np.float32)
    for node_count in xrange(num_nodes):
        this_node = struct.unpack('=iddd', file_handle.read(28))
        if this_node[0] != node_count+1:
            raise MeshError("Inconsistent node numbering")

        nodes[node_count] = this_node[1:]

    file_handle.readline()
    
    return nodes

def check_format(file_handle):
    "Check that the format of a gmsh file"
    # check the header version
    if file_handle.readline().split() != ['2.2', '1', '8']:
        raise MeshError("gmsh file has incorrect version format")
        
    # check the endianness of the file
    if struct.unpack('=i', file_handle.read(4))[0] != 1:
        raise MeshError("gmsh file format invalid")

    file_handle.readline()

def read_elements(file_handle, wanted_element_types):
    "Read in all the elements from a gmsh file"
    num_elements = int(file_handle.readline())

    object_nodes = defaultdict(set)
    object_elements = defaultdict(lambda : defaultdict(list))
    
    # currently we are only interested in the triangle elements
    # so skip over all others
    for _ in xrange(num_elements):
        element_type, num_elem_in_group, num_tags = struct.unpack('=iii', 
                                                      file_handle.read(12))
        
        num_nodes_in_elem = GMSH_ELEMENT_NODES[element_type]
        if num_tags < 2:
            raise MeshError("Missing elementary geometry tag")
        
        element_bytes = 4*(1 + num_tags + num_nodes_in_elem)

        elem_format = "=i" + "i"*num_tags + "i"*num_nodes_in_elem

        element_data = file_handle.read(num_elem_in_group*element_bytes)

        # Avoid reading in unwanted element types. This is important for
        # getting rid of nodes which are not a part of any triangle
        if element_type not in wanted_element_types:
            continue
        
        # iterate over all elements within the same header block
        for these_elements_count in xrange(num_elem_in_group):
            this_element = struct.unpack(elem_format, 
                element_data[these_elements_count*element_bytes:
                                (these_elements_count+1)*element_bytes])
                
            # Assumes that the required default tags are used, and finds the
            # *physical entity* of the mesh element
            entity = this_element[1]
            #entity = this_element[2]
                
            # NB: conversion to python 0-based indexing is done here
            element_nodes = np.array(this_element[-num_nodes_in_elem:])-1

            object_elements[entity][element_type].append(element_nodes)
            object_nodes[entity].update(element_nodes)
    
    file_handle.readline()
    
    return object_nodes, object_elements
    
def read_physical_names(file_handle):
    "Read the physical names from a gmsh file"
    
    physical_names = {}
    num_physical_names = int(file_handle.readline())
    
    for _ in xrange(num_physical_names):
        dimension, num, name = file_handle.readline().split()
        physical_names[int(num)] = name.strip('"')
        
    return physical_names

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
    
    wanted_element_types = set(ELEMENT_NAME_MAPPING[n] for n in 
                                                            returned_elements)
    
    physical_names = None # may not exist in file    
    
    with open(filename, "rb") as file_handle:
        header = "Nothing"
        while True:
            header = file_handle.readline().strip()
            if header == "":
                # should be at the end of the file
                break
            elif header == "$MeshFormat":
                check_format(file_handle)
            elif header == "$Nodes":
                nodes = read_nodes(file_handle)
                if len(nodes) == 0:
                    raise MeshError("No nodes in mesh")
            elif header == "$Elements":
                object_nodes, object_elements = read_elements(file_handle, 
                                                          wanted_element_types)
            elif header == "$PhysicalNames":
                physical_names = read_physical_names(file_handle)
            else:
                raise MeshError("Unkown header type " + header)
                
            end_header = "$End"+header[1:]
            if file_handle.readline().strip() != end_header:
                raise MeshError("Header %s with no matching %s" % (header, end_header))
            
    return_vals = []        
    
    # Go through each entity, and work out which nodes belong to it. Nodes are
    # renumbered, so elements are updated to reflect new numbering
    for obj_nums, obj_nodes, obj_elements in zip(object_nodes.iterkeys(),
            object_nodes.itervalues(), object_elements.itervalues()):

        # renumber the nodes                
        orig_nodes = np.sort(list(obj_nodes))
        new_nodes = np.zeros(len(nodes), np.int)
        for node_count, node in enumerate(orig_nodes):
            new_nodes[node] = node_count
        
        this_part = {'nodes': nodes[orig_nodes]}
        
        # let the elements know about the renumbered nodes
        for elem_name in returned_elements:
            returned_type = ELEMENT_NAME_MAPPING[elem_name]
            if len(obj_elements[returned_type]) > 0:
                this_part[elem_name] = new_nodes[
                                        np.array(obj_elements[returned_type])]
            
        # add the physical names
        if physical_names is not None:
            this_part["physical_name"] = physical_names[obj_nums]
            
        return_vals.append(this_part)
        
    return tuple(return_vals)

def check_installed():
    "Check if a supported version of gmsh is installed"
    call_options = ['gmsh', '--version']

    try:
        proc = subprocess.Popen(call_options, stderr=subprocess.PIPE)
    except OSError:
        raise MeshError("gmsh not found")
        
    ver = tuple([int(x) for x in proc.stderr.readline().split(".")])
    
    if ver < MIN_VERSION:
        raise MeshError("gmsh version %d.%d.%d found, "+
            "but version %d.%d.%d required" % (ver+MIN_VERSION))
           
check_installed()

if __name__ == "__main__":
    file_name = mesh_geometry("geometry/asymmetric_ring.geo", 0.4e-3)
    print read_mesh(file_name)
    