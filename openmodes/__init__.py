# -*- coding: utf-8 -*-
"""
OpenModes
=========

A Method of Moments (Boundary Element Method) code designed to find the modes
of open resonators such as meta-atoms, (nano) antennas, scattering particles
etc.

Using these modes, broadband models of these elements can be created, enabling
excitation, coupling between them and scattering to be solved easily, and
broadband models to be created

Copyright 2013 David Powell

TODO: License to go here
"""

import gmsh
import os.path as osp
from openmodes.parts import LibraryPart
from openmodes.solver import Simulation

def load_parts(filename, mesh_tol=None, force_tuple = False):
    """
    Open a gmsh geometry or mesh file into the relevant parts
    
    Parameters
    ----------
    filename : string
        The name of the file to open. Can be a gmsh .msh file, or a geometry
        file, which will be meshed first
    mesh_tol : float, optional
        If opening a geometry file, it will be meshed with this tolerance
    force_tuple : boolean, optional
        Ensure that a tuple is always returned, even if only a single part
        is found in the file

    Returns
    -------
    parts : tuple
        A tuple of `SimulationParts`, one for each separate geometric entity
        found in the gmsh file
    """
    
    if osp.splitext(osp.basename(filename))[1] == ".msh":
        # assume that this is a binary mesh already generate by gmsh
        meshed_name = filename
    else:
        # assume that this is a gmsh geometry file, so mesh it first
        meshed_name = gmsh.mesh_geometry(filename, mesh_tol)

    node_tri_pairs = gmsh.read_mesh(meshed_name)
    
    parts = tuple(LibraryPart(nodes, triangles) for (nodes, triangles) in node_tri_pairs)
    if len(parts) == 1 and not force_tuple:
        return parts[0]
    else:
        return parts

__all__ = [Simulation, load_parts]

