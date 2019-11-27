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

Gmsh 2.8.4 is required, as it introduced the `setnumber` command line parameter
"""

from __future__ import print_function

import subprocess
import os.path as osp
import os
import struct
import numpy as np
from collections import defaultdict
import re
import logging
import tempfile

from openmodes.helpers import MeshError

# the minimum version of gmsh required
MIN_VERSION = (3, 0, 0)

try:
    gmsh_path = os.environ['GMSH_PATH']
except KeyError:
    gmsh_path = 'gmsh'


def mesh_geometry(filename, dirname, mesh_tol=None, binary=True,
                  parameters={}):
    """Call gmsh to surface mesh a geometry file with a specified maximum
    tolerance

    Parameters
    ----------
    filename : string
        the name of the file to be meshed
    dirname : string
        The location in which to create the mesh file
    mesh_tol : number, optional
        override the maximum mesh tolerance distance of all edges
    binary : boolean, optional
        (default True) output a binary file
    parameters : dictionary, optional
        A dictionary containing the values of geometric parameters to be
        modified within the gmsh geometry before meshing. Note that the
        geometry file must be written to check for these values, otherwise
        they will be overwritten if they are assigned within the geometry file

    Returns
    -------
    meshname : string
        the full path of the .msh file

    This routine instructs gmsh to use algorithm 1 for 2D meshing, which seems
    to yield the most consistent and uniform mesh.
    """

    check_installed()

    if not osp.exists(filename):
        raise MeshError("Geometry file %s not found" % filename)

    meshname = osp.join(dirname, osp.splitext(osp.basename(filename))[0]
                        + ".msh")

    call_options = [gmsh_path, filename, '-2', '-o', meshname,
                    '-string', 'Mesh.Algorithm=1;']

    # override geometric parameters on the command-line
    for param, value in parameters.items():
        call_options += ['-setnumber', param, str(value)]

    if mesh_tol is not None:
        call_options += ['-clmax', '%f' % mesh_tol]

    if binary:
        call_options += ['-bin']

    logging.info("Calling gmsh with options %s" % call_options)
    proc = subprocess.Popen(call_options, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, universal_newlines=True)

    # run the process and read in stderr and stdout streams
    stdouttxt, stderrtxt = proc.communicate()
    if proc.returncode != 0:
        # Non-zero return code incidates some problem
        print(stdouttxt)
        print(stderrtxt)
        raise MeshError("Gmsh did not run successfully")

    logging.info(stdouttxt)
    if len(stderrtxt) > 0:
        logging.warning("gmsh error/warning for file %s:\n%s"
                        % (filename, stderrtxt))

    return meshname


EDGE_TYPE = 1
TRIANGLE_TYPE = 2
POINT_TYPE = 15
# the number of nodes in different gmsh element types which may be encountered
GMSH_ELEMENT_NODES = {EDGE_TYPE: 2, TRIANGLE_TYPE: 3, POINT_TYPE: 1}

ELEMENT_NAME_MAPPING = {"edges": EDGE_TYPE, "triangles": TRIANGLE_TYPE,
                        "points": POINT_TYPE}


def read_nodes(file_handle):
    "Read in the nodes of a gmsh file"
    num_nodes = int(file_handle.readline())

    nodes = np.empty((num_nodes, 3), np.float32)
    for node_count in range(num_nodes):
        this_node = struct.unpack('=iddd', file_handle.read(28))
        if this_node[0] != node_count+1:
            raise MeshError("Inconsistent node numbering")

        nodes[node_count] = this_node[1:]

    file_handle.readline()

    return nodes


def check_format(file_handle):
    "Check that the format of a gmsh file"
    # check the header version
    if file_handle.readline().decode('ascii').split() != ['2.2', '1', '8']:
        raise MeshError("gmsh file has incorrect version format")

    # check the endianness of the file
    if struct.unpack('=i', file_handle.read(4))[0] != 1:
        raise MeshError("gmsh file format invalid")

    file_handle.readline()


def read_elements(file_handle, wanted_element_types):
    "Read in all the elements from a gmsh file"
    num_elements = int(file_handle.readline())

    object_nodes = defaultdict(set)
    object_elements = defaultdict(lambda: defaultdict(list))

    # currently we are only interested in the triangle elements
    # so skip over all others
    for _ in range(num_elements):
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
        for these_elements_count in range(num_elem_in_group):
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

    names_regex = re.compile(r'(\d*)\s(\d*)\s"([^"]*)"')

    for _ in range(num_physical_names):
        names_match = names_regex.match(file_handle.readline().decode('ascii'))
        dimension, num, name = names_match.groups()
        physical_names[int(num)] = name

    return physical_names


def read_mesh(filename, returned_elements=("edges", "triangles")):
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

    physical_names = None  # may not exist in file

    with open(filename, "rb") as file_handle:
        header = "Nothing"
        while True:
            header = file_handle.readline().decode('ascii').strip()
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
            if file_handle.readline().decode('ascii').strip() != end_header:
                raise MeshError("Header %s with no matching %s" % (header, end_header))

    return_vals = []

    # Go through each entity, and work out which nodes belong to it. Nodes are
    # renumbered, so elements are updated to reflect new numbering
    for obj_nums, obj_nodes, obj_elements in zip(object_nodes.keys(),
            object_nodes.values(), object_elements.values()):

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

        # add the physical name if it exists
        try:
            this_part["physical_name"] = physical_names[obj_nums]
        except (TypeError, KeyError):
            pass

        return_vals.append(this_part)

    return tuple(return_vals)

def read_mesh_meshio(filename):
    """Read a gmsh binary mesh file using the meshio library

    Parameters
    ----------
    filename : string
        the full name of the gmesh meshed file

    Returns
    -------
    list of:
        raw_mesh : dict
            Containing the following
            nodes : ndarray
                all nodes referred to by this geometric entity
            triangles : ndarray
                the indices of nodes belonging to each triangle

    Node references are set to be zero based indexing as per python standard

    Any geometric elements which are not part of the final mesh are pruned.
    """

    import meshio

    mesh = meshio.read(filename, file_format='gmsh')

    # check whether multiple physical objects are defined
    if 'gmsh:physical' in mesh.cell_data['triangle']:
        triangle_physical = mesh.cell_data['triangle']['gmsh:physical']
        if not np.all(triangle_physical == triangle_physical[0]):
        raise NotImplementedError('Multiple physical objects in one mesh not yet implemented')

    # eliminate points which are not part of the object
    mesh.prune()

    return [{'nodes': mesh.points, 'triangles': mesh.cells['triangle']}]


def check_installed():
    "Check if a supported version of gmsh is installed"
    call_options = [gmsh_path, '-info']

    try:
        # Workaround for different versions of gmsh, 3.x writes to stderr,
        # 4.x writes to stdout, but only if redirected to a file
        with tempfile.TemporaryFile() as out:
            proc = subprocess.run(call_options, stdout=out, stderr=out,
                                    universal_newlines=True, encoding='utf-8')
            out.seek(0)
            version_string = out.readline().decode('utf-8').split(":")[1].strip()

    except OSError:
        raise MeshError("gmsh not found")

    ver = tuple([int(x) for x in version_string.split(".")])

    if ver < MIN_VERSION:
        raise MeshError(("gmsh version %d.%d.%d found, " +
            "but version %d.%d.%d required") % (ver+MIN_VERSION))
