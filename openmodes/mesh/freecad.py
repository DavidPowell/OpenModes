# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
#  OpenModes - An eigenmode solver for open electromagnetic resonantors
#  Copyright (C) 2013-2015 David Powell
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
# -----------------------------------------------------------------------------
"Create meshes using FreeCAD"

from __future__ import division

import numpy as np

import FreeCAD
import Part
import MeshPart


def freecad_mesh(geometry):
    mesh = MeshPart.meshFromShape(geometry, Fineness=2, SecondOrder=0,
                                  Optimize=1, AllowQuad=0)

    result = {}

    # Point IDs might be in arbitrary order, so create a lookup to be sure
    # that we have the correct point ids
    point_ids = {}
    points = []
    for point_count, point in enumerate(mesh.Points):
        points.append([point.x, point.y, point.z])
        point_ids[point.Index] = point_count

    result['nodes'] = np.array(points, dtype=np.float64)

    triangles = []
    for facet_count, facet in enumerate(mesh.Facets):
        if len(facet.PointIndices) != 3:
            raise NotImplementedError("Only triangles currently supported")
        triangles.append([point_ids[n] for n in facet.PointIndices])

    result['triangles'] = np.array(triangles)

    return result


def box(x, y, z, rounding=None):
    """Create a box, optionally rounding the edges

    Parameters
    ----------
    x, y, z: float
        Size of the box in 3 dimensions
    rounding: float, optional
        If specified, the edges will be rounded with this radius
    """
    box = Part.makeBox(x, y, z)

    centre = FreeCAD.Vector(-0.5*x, -0.5*y, -0.5*z)
    box.Placement.Base = centre

    if rounding is not None:
        box = box.makeFillet(rounding, box.Edges)
    return box


#bar = box(500, 150, 200, 10)
#mesh = freecad_mesh(bar)

#MeshPart.meshFromShape(box,GrowthRate=0.3,SegPerEdge=1,SegPerRadius=2,SecondOrder=0,Optimize=1,AllowQuad=0)