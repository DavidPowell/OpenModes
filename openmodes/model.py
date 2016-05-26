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
"Fit scalar models to numerically calculated impedance data"

import numpy as np
from openmodes.impedance import ImpedanceMatrixLA, EfieImpedanceMatrixLA


class ModelMutualWeight(object):
    """A model where mutual terms come from directly weighting the mutual
    terms of the full impedance matrix"""

    def __init__(self, modes):
        """
        Parameters
        ----------
        modes : Modes object
            The modes object from which to create
        """
        self.modes = modes
        self.parts = list(modes.modes_of_parts.keys())
        self.parent_part = modes.parent_part
        self.macro_container = modes.macro_container
        self.symmetric = modes.operator.reciprocal
        self.impedance_class = ImpedanceMatrixLA

    def impedance_self(self, s, part_o, Z_full):
        "Self impedance of one part"
        s_o = self.modes.s[0, part_o]
        Z_full.matrices['Z'][part_o, part_o] = np.diag(s_o*(s-s_o)/s)
        # TODO: impedance derivative

    def impedance_mutual(self, s, part_o, part_s, Z_full):
        "Impedance between two parts, by weighting matrix"
        vl = self.modes.vl[:, part_o, :, part_o]
        vr = self.modes.vr[:, part_s, :, part_s]
        Z_full[part_o, part_s] = self.modes.operator.impedance(s, part_o, part_s).weight(vr, vl)

    def impedance(self, s):
        """Impedance matrix

        Parameters
        ----------
        s : complex
            Frequency at which to calculate impedance
        """
        Z = self.impedance_class(self.parent_part, self.parent_part,
                                 self.macro_container, ('modes',), ('modes',))

        Z.md['s'] = s
        Z.md['symmetric'] = self.symmetric
        Z.md['operator'] = self.modes.operator

        for count_o, part_o in enumerate(self.parts):
            for count_s, part_s in enumerate(self.parts):
                if count_o == count_s:
                    self.impedance_self(s, part_o, Z)
                elif self.symmetric and (count_o > count_s):
                    # account for symmetry of operator
                    Z[part_o, part_s] = Z[part_s, part_o].T
                else:
                    self.impedance_mutual(s, part_o, part_s, Z)
        return Z


class EfieModelMutualWeight(ModelMutualWeight):
    """A model where mutual terms come from directly weighting the mutual
    terms of the full impedance matrix"""

    def __init__(self, modes):
        ModelMutualWeight.__init__(self, modes)
        self.impedance_class = EfieImpedanceMatrixLA

    def impedance_self(self, s, part_o, Z_full):
        "Self impedance of one part"
        s_o = self.modes.s[0, part_o]
        Z_full.matrices['S'][part_o, part_o] = np.diag(s_o*(s-s_o))
        Z_full.matrices['L'][part_o, part_o] = 0.0
        # TODO: derivatives
