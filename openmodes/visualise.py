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


def compress(func, factor, max_val=None):
    "Compress a function to smooth out extreme values"
    max_val = max_val or max(abs(func))
    return np.tanh(abs(func)*factor/max_val)*np.exp(1j*np.angle(func))


def preprocess(parts, solution=None, basis_container=None,
               compress_scalars=None, compress_separately=False):
    """Pre-process the parts and solution before plotting, including scaling

    Parameters
    ----------
    parts: Part
        The tree containing all parts in the simulation
    solution: PartsArray, optional
        The solution to plot on each element
    basis_container: optional
        The container for the basis functions
    compress_scalar : float, optional
        If specified, this compression factor will be applied to reduce the
        dynamic range of the scalar data
    autoscale_vectors : boolean, optional
        Automatically scale the vectors so that the maximum value is 1.0
    compress_separately : boolean, optional
        Apply the compression and scaling separately to each part, which will
        hide the differences between parts
    """

    parts_list = []
    charges = []
    currents = []
    centres = []
    max_charge = 0.0

    for part_num, part in enumerate(parts.iter_single()):
        parts_list.append(part)

        I = solution[part]
        basis = basis_container[part]

        centre, current, charge = basis.interpolate_function(I,
                                                        return_scalar=True,
                                                        nodes=part.nodes)

        if compress_scalars:
            if compress_separately:
                charge = compress(charge, compress_scalars)
            else:
                max_charge = max(max_charge, max(abs(charge)))

        charges.append(charge)
        currents.append(current)
        centres.append(centre)

    if compress_scalars and not compress_separately:
        for charge_count, charge in enumerate(charges):
            charges[charge_count] = compress(charge, compress_scalars, max_charge)

    return parts_list, charges, currents, centres


def plot_parts(parts, figsize=(10, 4), view_angles = (40, 90)):
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
                                        triangle_nodes,
                                        representation='wireframe',
                                        color=(0, 0, 0), line_width=0.5,
                                        opacity=opacity)
        if scalar_function is not None:
            if compress_scalars:
                part_scalars = compress(scalar_function[part_num],
                                        compress_scalars)
            else:
                part_scalars = scalar_function[part_num]

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

scalar_names = {'J': 'rho_e', 'M': 'rho_m'}

# Reduce precision for some types
vtk_type_map = {np.int32: 'Int32', np.int64: 'Int32',
                np.float32: 'Float32', np.float64: 'Float32'}
vtk_type_map = {np.dtype(key): val for key, val in vtk_type_map.items()}


def vtk_da(doc, ar, name=None, type_name=None):
    "Create a VTK DataArray in XML"

    da = doc.createElementNS("VTK", "DataArray")

    if len(ar.shape) > 1:
        da.setAttribute("NumberOfComponents", str(np.product(ar.shape[1:])))

    if type_name is None:
        type_name = vtk_type_map[ar.dtype]
    da.setAttribute("type", type_name)
    da.setAttribute("format", "ascii")
    
    if name is not None:
        da.setAttribute("Name", name)

    text = doc.createTextNode(" ".join(str(x) for x in ar.flat))
    da.appendChild(text)

    return da


def write_vtk(parts, filename, solution=None, basis_container=None):
    """Write the mesh and solution data to a VTK file, by directly generating
    the XML tree

    If the current vector is given, then currents and charges will be
    written to the VTK file. Otherwise only the mesh is written

    Parameters
    ----------
    filename : string
        The modern XML PolyData format will be used, which should be correctly
        indicated by using the file extension `.vtp`.
    part : Part
        The parent part
    solution: LookupArray, optional
        The solution to plot over the surface
    basis_container: BasisContainer, optional
        The basis container, required if a solution is given
    """

    import xml.dom.minidom
    import sys

    doc = xml.dom.minidom.Document()
    root = doc.createElementNS("VTK", "VTKFile")

    root.setAttribute("type", "PolyData")
    root.setAttribute("version", "0.1")

    if sys.byteorder == 'little':
        root.setAttribute("byte_order", "LittleEndian")
    else:
        root.setAttribute("byte_order", "BigEndian")

    doc.appendChild(root)

    polydata = doc.createElementNS("VTK", "PolyData")
    root.appendChild(polydata)

    for part in parts.iter_single():
        mesh = part.mesh

        piece = doc.createElementNS("VTK", "Piece")
        piece.setAttribute("NumberOfPoints", str(len(part.nodes)))
        piece.setAttribute("NumberOfPolys", str(len(mesh.polygons)))
        polydata.appendChild(piece)

        # Add the points
        points = doc.createElementNS("VTK", "Points")
        piece.appendChild(points)

        # DataArray containing point coordinates
        points.appendChild(vtk_da(doc, part.nodes))

        # Now define the polygons
        polys = doc.createElementNS("VTK", "Polys")
        piece.appendChild(polys)
        polys.appendChild(vtk_da(doc, mesh.polygons.flatten(), "connectivity"))
        polys.appendChild(vtk_da(doc, np.cumsum([len(y) for y in mesh.polygons]), "offsets"))

        if solution is None:
            continue

        # The current and charge distributions, are currently defined at a
        # single point per cell, hence they are CellData
        basis = basis_container[part]

        celldata = doc.createElementNS("VTK", "CellData")
        # Iterate over multiple quantities (e.g. electric and magnetic current)
        for sol in solution.lookup[0]:
            # Get the current at the centre of the triangle, and its divergence
            centre, current, charge = basis.interpolate_function(solution[sol, part],
                                                                 return_scalar=True,
                                                                 nodes=part.nodes)
            try:
                scalar_name = scalar_names[sol]
            except KeyError:
                scalar_name = "scalar_"+sol

            # The real and imaginary parts of the charge and current
            celldata.appendChild(vtk_da(doc, charge.real, scalar_name+"_real"))
            celldata.appendChild(vtk_da(doc, charge.imag, scalar_name+"_imag"))
            celldata.appendChild(vtk_da(doc, current.real, sol+"_real"))
            celldata.appendChild(vtk_da(doc, current.imag, sol+"_imag"))

        piece.appendChild(celldata)

    with open(filename, "w") as outfile:
        doc.writexml(outfile, newl='\n', addindent='  ')
