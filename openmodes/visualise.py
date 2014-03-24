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

from openmodes.mesh import combine_mesh


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
    
    for part in parts.iter_single():
        mesh = part.mesh
            
        for edge in mesh.get_edges():
            nodes = part.nodes[edge]
            ax.plot(nodes[:, 0], nodes[:, 1], nodes[:, 2], 'k')
     
    ax.view_init(*view_angles)
    ax.autoscale()
    plt.show()


def plot_mayavi(parts, scalar_function=None, vector_function=None,  
                vector_points=None, scalar_name="scalar", vector_name="vector",
                compress_scalars=None, filename=None):
    """Generate a mayavi plot of the mesh of the parts, and optionally also
    show a plot of vector and scalar functions defined on its surface.
    
    Parameters
    ----------
    parts : list of `Part`
        All the parts to plot
    scalar_function : list of real arrays, optional
        A scalar function which will be plotted on the surface of the parts
    vector_function : list of real arrays, optional
        An optional vector function to plot as arrows
    vector_points : list of real arrays, optional
        The points at which the vector function is calculated
    scalar_name : string, optional
        A name for the scalar function
    vector_name : string, optional
        A name for the vector function
    compress_scalars : real, optional
        The parameter of a tanh function which will be used to scale the data
        in a nonlinear fashion. This can smooth out the extreme values of the
        function at certain points, allowing the user to see more detail in the
        regions where the value is smaller.
    filename : string, optional
        If specified, the figure will be saved to this file, rather than
        being plotted on screen.
        
    Only the real part of the scalar and vector functions will be plotted
    """
    try:
        from mayavi import mlab
    except ImportError:
        raise ImportError("Please ensure that Enthought Mayavi is correctly " +
                          "installed before calling this function")

    # If plotting a scalar function, plotting the mesh lines tends to make
    # the plot too busy, so switch them off.
    if scalar_function is not None:
        opacity = 0.0
    else:
        opacity = 1.0

    if filename is not None:
        # render to a file, so we don't need to display anything
        offscreen = mlab.options.offscreen
        mlab.options.offscreen = True

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
                part_scalars=compress(scalar_function[part_num],
                                            compress_scalars)
            else:
                part_scalars=scalar_function[part_num]

            cell_data = tri_plot.mlab_source.dataset.cell_data
            cell_data.scalars = part_scalars.real
            cell_data.scalars.name = scalar_name
            cell_data.update()

            mesh2 = mlab.pipeline.set_active_attribute(tri_plot,
                    cell_scalars=scalar_name)
            mlab.pipeline.surface(mesh2)

        if vector_function is not None:
            points = vector_points[part_num]
            vectors = vector_function[part_num].real
            mlab.quiver3d(points[:, 0], points[:, 1], points[:, 2],
                          vectors[:, 0], vectors[:, 1], vectors[:, 2],
                          color=(0, 0, 0), opacity=0.75, line_width=1.0)

        mlab.view(distance='auto')
    if filename is None:
        mlab.show()
    else:
        mlab.savefig(filename)
        mlab.options.offscreen = offscreen


def write_vtk(parts, scalar_function=None, vector_function=None,
                vector_points=None, scalar_name="scalar", vector_name="vector",
                compress_scalars=None, filename=None, autoscale_vectors=False,
                compress_separately=False):
    """Write the mesh and solution data to a VTK file

    If the current vector is given, then currents and charges will be
    written to the VTK file. Otherwise only the mesh is written

    Parameters
    ----------
    filename : string
        File to save to. If it has the extension `.vtk`, then the file will be
        written in the simple legacy format. Otherwise the modern XML format
        will be used, which should be correctly indicated by using the tile
        extension `.vtp`.
    parts : list
        The list of all SingleParts
    scalar_function : list, optional
        The scalar function to plot for each part
    vector_function : list, optional
        The vector function to plot for each part
    scalar_name : string, optional
        The name of the scalar function
    vector_name : string, optional
        The name of the vector function
    compress_scalar : float, optional
        If specified, this compression factor will be applied to reduce the
        dynamic range of the scalar data
    autoscale_vectors : boolean, optional
        Automatically scale the vectors so that the maximum value is 1.0
    compress_separately : boolean, optional
        Apply the compression and scaling separately to each part, which will
        hide the differences between parts
        
    Requires that tvtk is installed
    """

    try:
        from tvtk.api import tvtk, write_data
    except ImportError:
        raise ImportError("Please ensure that tvtk is correctly installed")

    meshes = [part.mesh for part in parts]
    nodes = [part.nodes for part in parts]
    mesh = combine_mesh(meshes, nodes)

    polygons = mesh.polygons.tolist()
    struct = tvtk.PolyData(points=mesh.nodes, polys=polygons)

    if scalar_function is not None:
        if compress_scalars and compress_separately:
            scalar_function=[compress(s, compress_scalars) for s in scalar_function]
        scalar_function = np.hstack(scalar_function)
        if compress_scalars and not compress_separately:
            scalar_function=compress(scalar_function, compress_scalars)
        
        scalar_real = tvtk.FloatArray(name=scalar_name+"_real")
        scalar_real.from_array(scalar_function.real)
        struct.cell_data.add_array(scalar_real)

        scalar_imag = tvtk.FloatArray(name=scalar_name+"_imag")
        scalar_imag.from_array(scalar_function.imag)
        struct.cell_data.add_array(scalar_imag)

    if vector_function is not None:
        if autoscale_vectors and compress_separately:
            vector_function=[v/np.max(np.abs(v)) for v in vector_function]
        vector_function = np.vstack(vector_function)
        if autoscale_vectors and not compress_separately:
            vector_function=vector_function/np.max(np.abs(vector_function))
        
        vector_real = tvtk.FloatArray(name=vector_name+"_real")
        vector_real.from_array(vector_function.real)
        struct.cell_data.add_array(vector_real)

        vector_imag = tvtk.FloatArray(name=vector_name+"_imag")
        vector_imag.from_array(vector_function.imag)
        struct.cell_data.add_array(vector_imag)

    write_data(struct, filename)
    
        