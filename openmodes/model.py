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

from __future__ import division
import numpy as np
from openmodes.impedance import ImpedanceMatrixLA, EfieImpedanceMatrixLA
from openmodes.modes import SplitModes


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
        self.vl = self.modes.vl
        self.vr = self.modes.vr

    def impedance_self(self, s, part_o, Z_full):
        "Self impedance of one part"
        s_o = self.modes.s[0, part_o]
        Z_full.matrices['Z'][part_o, part_o] = np.diag(s_o*(s-s_o)/s)
        # TODO: impedance derivative

    def impedance_mutual(self, s, part_o, part_s, Z_full):
        "Impedance between two parts, by weighting matrix"
        vl = self.vl[:, part_o, :, part_o]
        vr = self.vr[:, part_s, :, part_s]
        z_weighted = self.modes.operator.impedance(s, part_o, part_s).weight(vr, vl)

        # If the model has the same impedance class as the full matrix,
        # then store all sub-matrices. Otherwise just get the combined value.
        try:
            Z_full[part_o, part_s] = z_weighted
        except KeyError:
            Z_full.matrices['Z'][part_o, part_s] = z_weighted.val().simple_view()

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


class ModelSplit(ModelMutualWeight):
    "A model of modes which have been split into real and imaginary parts"
    def __init__(self, modes):
        if not isinstance(modes, SplitModes):
            modes = modes.split_real_imag()
        super(ModelSplit, self).__init__(modes)

    def impedance_self(self, s, part_o, Z_full):
        "Self impedance of one part"
        s_o = self.modes.s[0, part_o]
        num_modes = len(s_o)//2
        s_r = s_o[:num_modes]
        s_i = s_o[num_modes:]
        Z_self = np.diag(s_r + (s_i**2 - s_r**2)/s)*0.5
        Z_mutual = np.diag(s_i - 2*s_r*s_i/s)*0.5
        Z_full.matrices['Z'][part_o, part_o] = np.vstack((np.hstack((Z_self, Z_mutual)),
                                                          np.hstack((Z_mutual, -Z_self))))


class EfieModelSplit(EfieModelMutualWeight):
    "A model of modes which have been split into real and imaginary parts"
    def __init__(self, modes):
        if not isinstance(modes, SplitModes):
            modes = modes.split_real_imag()
        super(EfieModelSplit, self).__init__(modes)

    def impedance_self(self, s, part_o, Z_full):
        "Self impedance of one part"
        s_o = self.modes.s[0, part_o]
        num_modes = len(s_o)//2
        s_r = s_o[:num_modes]
        s_i = s_o[num_modes:]
        Z_self = np.diag(s_r*s + (s_i**2 - s_r**2))*0.5
        Z_mutual = np.diag(s_i*s - 2*s_r*s_i)*0.5
        Z_full.matrices['S'][part_o, part_o] = np.vstack((np.hstack((Z_self, Z_mutual)),
                                                          np.hstack((Z_mutual, -Z_self))))
        Z_full.matrices['L'][part_o, part_o] = 0.0
