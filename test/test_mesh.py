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

import openmodes
import os.path as osp


def test_closed():
    "Check whether the meshes are closed"

    # For each file, list whether each of the meshes it contains is closed.
    # There may be multiple meshes per file.
    # TODO: get all geometries working
    mesh_values = [#('asymmetric_ring.geo', (False, False)),
                   # ('canonical_spiral.geo', (False,)),
                   ('circle.geo', (False,)),
                   ('circled_cross.geo', (False,)),
                   ('closed_ring.geo', (False,)),
                   ('cross.geo', (False,)),
                   ('ellipsoid.geo', (True,)),
                   ('horseshoe_rect.geo', (True,)),
                   ('isosceles.geo', (False,)),
                   ('rectangle.geo', (False,)),
                   ('single.geo', (False,)),
                   ('sphere.geo', (True,)),
                   ('SRR.geo', (False,)),
                   # ('torus.geo', (True,)),
                   # ('v_antenna.geo', (False,)),
                   ]

    sim = openmodes.Simulation()

    for filename, closed in mesh_values:
        mesh = sim.load_mesh(osp.join(openmodes.geometry_dir, filename),
                             force_tuple=True)
        assert all(m.closed_surface == closed_val for (m, closed_val) in
                   zip(mesh, closed)), \
            ("%s closed_surface is not %s" % (filename, closed))

if __name__ == "__main__":
    test_closed()
