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
Routines for displaying parts and solutions.
"""
import numpy as np

from openmodes.parts import Part


def compress(func, factor):
    "Compress a function to smooth out extreme values"
    return np.tanh(func*factor/max(abs(func)))


def plot_parts(parts, currents=None, figsize=(10, 4), view_angles = (40, 90)):
    """Create a simple 3D plot to show the location of loaded parts in space

    Parameters
    ----------
    parts : list of `Mesh` or `Part`
        The parts to plot
    figsize : tuple, optional
        The figsize (in inches) which will be passed to matplotlib
    viewangles : tuple, optional
        The viewing angle of the 3D plot in degrees, will be passed to
        matplotlib
        
    Requires that matplotlib is installed
    """

    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    for part in parts:
        if isinstance(part, Part):
            mesh = part.mesh
        else:
            mesh = part
            
        for edge in mesh.get_edges():
            nodes = part.nodes[edge]
            ax.plot(nodes[:, 0], nodes[:, 1], nodes[:, 2], 'k')
        #if currents is not None:
        #    ax.quiver(
        #ax.scatter(part.nodes[:, 0], part.nodes[:, 1], part.nodes[:, 2], marker='x')
     
    ax.view_init(*view_angles)
    ax.autoscale()
    plt.show()


def plot_mayavi(parts, vector_function=None, scalar_function=None, 
                vector_points=None, vector_name="vector", scalar_name="scalar",
                compress_scalars=None, num_vectors=None):
    try:
        from mayavi import mlab
    except ImportError:
        raise ImportError("Please ensure that Enthought Mayavi is correctly " +
                          "installed before calling this function")

    if scalar_function is not None:
        opacity = 0.0
    else:
        opacity = 1.0

    mlab.figure(bgcolor=(1.0, 1.0, 1.0))
    for part_num, part in enumerate(parts):

        triangle_nodes = part.mesh.polygons
        nodes = part.nodes
        tri_plot = mlab.triangular_mesh(nodes[:, 0], nodes[:, 1], nodes[:, 2], 
                                    triangle_nodes, representation='wireframe', 
                                    color=(0, 0, 0), line_width=0.5,
                                    opacity=opacity)
        if scalar_function is not None:
            if compress_scalars:
                compressed_scalars=compress(scalar_function[part_num],
                                            compress_scalars)
            else:
                compressed_scalars=scalar_function[part_num]

            cell_data = tri_plot.mlab_source.dataset.cell_data
            cell_data.scalars = compressed_scalars
            cell_data.scalars.name = scalar_name
            cell_data.update()

            mesh2 = mlab.pipeline.set_active_attribute(tri_plot,
                    cell_scalars=scalar_name)
            mlab.pipeline.surface(mesh2)

        if vector_function is not None:
            points = vector_points[part_num]
            vectors = vector_function[part_num]
            if num_vectors is None:
                mask_points = None
            else:
                mask_points = int(len(vectors)//num_vectors)
            #print mask_points
            mask_points=5
            mlab.quiver3d(points[:, 0], points[:, 1], points[:, 2], 
                          vectors[:, 0], vectors[:, 1], vectors[:, 2], 
                          color=(0, 0, 0), opacity=0.75, line_width=1.0,
                          mask_points=mask_points)

    mlab.show()


def write_vtk(mesh, nodes, filename, vector_function=None, 
              scalar_function=None, vector_name="vector", 
              scalar_name="scalar"):
    """Write the mesh and solution data to a VTK file
    
    If the current vector is given, then currents and charges will be
    written to the VTK file. Otherwise only the mesh is written
    
    Parameters
    ----------
    filename : string
        file to save to (.vtk extension will be added)
    I : ndarray, optional
        current vector of MOM solution in RWG basis
    scale_coords : number, optional
        a scaling factor to apply to the coordinates
        
    Requires that pyvtk is installed
    """
    
    from pyvtk import PolyData, VtkData, Scalars, CellData, Vectors
    
    polygons = mesh.polygons.tolist()
    struct = PolyData(points=nodes, polygons=polygons)

    data = []

    if vector_function is not None:
        data.append(Vectors(vector_function.real, name=vector_name+"_real"))
        data.append(Vectors(vector_function.imag, name=vector_name+"_imag"))
        
    if scalar_function is not None:
        data.append(Scalars(scalar_function.real, name=scalar_name+"_real", 
                            lookup_table="default"))
        data.append(Scalars(scalar_function.imag, name=scalar_name+"_imag", 
                            lookup_table="default"))
    
    cell_data = CellData(*data)

    vtk = VtkData(struct, cell_data, "OpenModes mesh and data")
    #else:
    #    vtk = VtkData(struct, "MOM mesh")
        
    vtk.tofile(filename, "ascii")
    
        