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
"""Routines which are specific to operation within the IPython notebook"""

from six.moves import StringIO
import os.path as osp
from IPython.display import HTML, display
import numpy as np
import uuid
import json
import matplotlib

from openmodes.mesh import combine_mesh
from openmodes import template_env

from pkg_resources import resource_filename
three_js_dir = resource_filename('openmodes', osp.join('external', 'three.js'))
static_dir = resource_filename('openmodes', 'static')


def init_3d():
    """Initialise 3D plots within the IPython notebook, by injecting the
    required javascript libraries.
    """

    library_javascript = StringIO()

    library_javascript.write("""
    <p>Loading javascript for 3D plot in browser</p>
    <script type="text/javascript">
    /* Beginning of javascript injected by OpenModes */
    var openmodes_javascript_injected = true;
    """)

    three_js_libraries = ("three.min.js", "OrbitControls.js",
                          "Lut.js", "Detector.js", "CanvasRenderer.js",
                          "Projector.js")

    # Include required parts of three.js inline
    for library in three_js_libraries:
        with open(osp.join(three_js_dir, library)) as infile:
            library_javascript.write(infile.read())

    # include my custom javascript inline
    with open(osp.join(static_dir, "three_js_plot.js")) as infile:
        library_javascript.write(infile.read())

    library_javascript.write(
                "/* End of javascript injected by OpenModes */\n</script>\n")

    display(HTML(library_javascript.getvalue()))


def plot_3d(parts_list, charges, currents, centres, width=700, height=500,
            wireframe=False, skip_webgl=False):
    """Create a 3D plot in the IPython notebook

    Parameters
    ----------
    parts_list : list of Part
        All the parts to plot
    charges : list of ndarray
        The charge distribution on each part
    currents : list of ndarray
        The current distribution on each part
    centres : list of ndarray
        The centre of each triangle, where the current vector is located
    width : integer, optional
        The width of the plot
    height : integer, optional
        The height of the plot
    wireframe : bool, optional
        Whether the plot should initially show a wireframe view
    skip_webgl : bool, optional
        Do not attempt to use webgl rendering, always use slower canvas
        rendering

    Returns
    -------
    H : HTML
        An HTML object containing the necessary HTML, CSS and javascript
        to show the plot
    """

    meshes = [part.mesh for part in parts_list]
    nodes = [part.nodes for part in parts_list]
    full_mesh = combine_mesh(meshes, nodes)

    # combine the meshes
    # scale all nodes so that the maximum size is known
    mesh_scale = 100/full_mesh.fast_size()
    full_mesh.nodes = full_mesh.nodes*mesh_scale

    # generate a javascript representation of the object
    geometry_name = "geometry_"+str(uuid.uuid4()).replace('-', '')
    geometry_javascript = StringIO()
    geometry_javascript.write("var %s = " % geometry_name)

    geometry_tree = {'nodes': full_mesh.nodes.tolist(),
                     'triangles': full_mesh.polygons.tolist()}

    # include the charge information if it is present
    if charges is not None:
        charges = np.hstack(charges)
        geometry_tree['charge'] = {'real': charges.real.tolist(),
                                   'imag': charges.imag.tolist(),
                                   'abs':  abs(charges).tolist(),
                                   'phase': np.angle(charges, deg=True).tolist()}

    # Include the current information if it is present. Vectors will be of
    # the form (length, x, y, z), where the 3 cartesian components are scaled
    # to be a normal vector
    if currents is not None:
        currents = np.vstack(currents)
        lengths_real = np.sqrt(np.sum(currents.real**2, axis=1))
        lengths_imag = np.sqrt(np.sum(currents.imag**2, axis=1))
        currents.real /= lengths_real[:, None]
        currents.imag /= lengths_imag[:, None]

        current_real = np.hstack((lengths_real[:, None], currents.real)).tolist()
        current_imag = np.hstack((lengths_imag[:, None], currents.imag)).tolist()
        geometry_tree['current'] = {'real': current_real, 'imag': current_imag}
        geometry_tree['centres'] = (np.vstack(centres)*mesh_scale).tolist()

    json.dump(geometry_tree, geometry_javascript)

    geometry_javascript.write(';')

    html_source = template_env.get_template('three_js_plot.html')
    html_generated = html_source.render({'geometry_javascript': geometry_javascript,
                                         'geometry_name': geometry_name,
                                         'canvas_width': width,
                                         'canvas_height': height,
                                         'initial_wireframe': wireframe,
                                         'skip_webgl': skip_webgl,
                                         'current_vector_len': np.median(full_mesh.edge_lens)})

    display(HTML(html_generated))


def matplotlib_defaults():
    "Set some nicer defaults for matplotlib plots for ipython notebooks"
    rcp = matplotlib.rcParams
    rcp['figure.figsize'] = (8, 5)
    rcp['lines.linewidth'] = 1.0
    rcp['lines.markeredgewidth'] = 2.0
    rcp['axes.labelsize'] = 12
    rcp['font.size'] = 12
    rcp['patch.linewidth'] = 1.0
    rcp['figure.facecolor'] = 'white'
    rcp['figure.edgecolor'] = 'white'
