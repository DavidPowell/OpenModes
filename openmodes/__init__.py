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


"""
*OpenModes*

A Method of Moments (Boundary Element Method) code designed to find the modes
of open resonators such as meta-atoms, (nano) antennas, scattering particles
etc.

Using these modes, broadband models of these elements can be created, enabling
excitation, coupling between them and scattering to be solved easily, and
broadband models to be created

Copyright 2013 David Powell

TODO: License to go here
"""

from openmodes.solver import Simulation

import os.path as osp

#from openmodes.parts import LibraryPart
from openmodes.mesh import TriangularSurfaceMesh
from openmodes import gmsh

def load_parts(filename, mesh_tol=None, force_tuple = False):
    """
    Open a geometry file and mesh it (or directly open a mesh file), then
    convert it into a mesh object.
    
    Parameters
    ----------
    filename : string
        The name of the file to open. Can be a gmsh .msh file, or a gmsh 
        geometry file, which will be meshed first
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
        
    Currently only `TriangularSurfaceMesh` objects are created
    """
    
    if osp.splitext(osp.basename(filename))[1] == ".msh":
        # assume that this is a binary mesh already generate by gmsh
        meshed_name = filename
    else:
        # assume that this is a gmsh geometry file, so mesh it first
        meshed_name = gmsh.mesh_geometry(filename, mesh_tol)

    raw_mesh = gmsh.read_mesh(meshed_name)
    
    parts = tuple(TriangularSurfaceMesh(sub_mesh) for sub_mesh in raw_mesh)
    if len(parts) == 1 and not force_tuple:
        return parts[0]
    else:
        return parts
#__all__ = [Simulation, load_parts]

