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
OpenModes - An eigenmode solver for open electromagnetic resonantors
"""

from openmodes.simulation import Simulation

# dynamically query the project version
from pkg_resources import get_distribution, DistributionNotFound
import os.path

try:
    _dist = get_distribution('openmodes')
    if not __file__.lower().startswith(os.path.join(_dist.location, 'openmodes')):
        # not installed, but there is another version that *is*
        raise DistributionNotFound
except DistributionNotFound:
    raise DistributionNotFound('Please install this project with setup.py')
else:
    __version__ = _dist.version

# allow the user to find the provided geometry files
from pkg_resources import resource_filename
geometry_dir = resource_filename('openmodes', 'geometry')


#__all__ = [openmodes.simulation.Simulation, openmodes.mesh.load_mesh]

