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

import numpy as np

from openmodes.helpers import Identified, wrap_if_constant
from openmodes.constants import eta_0


class IsotropicMaterial(Identified):
    "An isotropic material with a given permittivity and permeability"
    def __init__(self, name, epsilon_r, mu_r):
        super(IsotropicMaterial, self).__init__()
        self.name = name
        self.epsilon_r = wrap_if_constant(epsilon_r)
        self.mu_r = wrap_if_constant(mu_r)

    def eta(self, s):
        "Absolute impedance of the material"
        return eta_0*self.eta_r(s)

    def eta_r(self, s):
        "Impedance of the material relative to free space"
        return np.sqrt(self.mu_r(s)/self.epsilon_r(s))

# a constant for free space
FreeSpace = IsotropicMaterial("Free space", 1.0, 1.0)

# a constant for a perfect electric conductor
PecMaterial = IsotropicMaterial("Perfect electric conductor", -np.inf, -np.inf)
