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
Routines to describe materials
"""
from __future__ import division

from openmodes.helpers import Identified, wrap_if_constant


class IsotropicMaterial(Identified):
    "An isotropic material with a given permittivity and permeability"
    def __init__(self, name, epsilon_r, mu_r):
        super(IsotropicMaterial, self).__init__()
        self.name = name
        self.epsilon_r = wrap_if_constant(epsilon_r)
        self.mu_r = wrap_if_constant(mu_r)

# a constant for free space
FreeSpace = IsotropicMaterial("Free space", 1.0, 1.0)

# a constant, indicating that this material is a perfect electric conductor
PecMaterial = "Perfect electric conductor"
