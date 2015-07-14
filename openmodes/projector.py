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
"Classes for projecting a vector or matrix onto a set of modes"

from __future__ import division

import numpy as np

from openmodes.array import LookupArray
from openmodes.impedance import ImpedanceMatrixLA
from openmodes.basis import BasisContainer, MacroBasis
from openmodes.parts import Part


class Projector(object):
    "A class for projecting a matrix or vector onto a set of modes"

    def __init__(self, modes, parent_part, operator, orig_container):
        if not isinstance(list(modes.keys())[0], Part):
            modes = {parent_part: modes}

        self.basis_container = BasisContainer(MacroBasis, global_args = {'right_basis': modes,
                                                                         'left_basis': modes})
        self.basis_container.lowest_parts = set(modes.keys())

        # The Macro basis functions should know which actual part they were
        # defined on, because their solutions are defined on a per part basis
#        for parent_part in modes.keys():
#            for part in parent_part.iter_single():
#                self.basis_container.set_args(part, {'part': part})#, 'parent_part': parent_part})

        # TODO: unknowns and sources can come from modes
        right = LookupArray((operator.unknowns, (parent_part, orig_container),
                             ('modes',), (parent_part, self.basis_container)),
                            dtype=np.complex128)
        left = LookupArray((('modes',), (parent_part, self.basis_container),
                            operator.sources, (parent_part, orig_container)),
                           dtype=np.complex128)
        right[:] = 0.0
        left[:] = 0.0

        for part_num, (part, mode) in enumerate(modes.items()):
            right[:, part, :, part] = mode['vr'][:, :, None, :]
            try:
                left[:, part, :, part] = mode['vl'][None, :, :, :]
            except KeyError:
                old_shape = mode['vr'].shape
                left[:, part, :, part] = mode['vr'].T.reshape((1, old_shape[2], old_shape[0], old_shape[1]))

        self.left = left
        self.right = right

    def __call__(self, original):
        "Perform a projection operation on an array"
        if isinstance(original, ImpedanceMatrixLA):
            # Reduced impedance matrix will throw away all information about
            # sub-matrices
            new_matrices = {'Z': self.left.dot(original.val().dot(self.right)).simple_view()}
            return ImpedanceMatrixLA(original.part_o, original.part_s, self.basis_container,
                                     ('modes',), ('modes',), original.md, new_matrices, original.der)

        # We are projecting onto either a 1D or 2D array
        left_proj = self.left.dot(original)

        try:
            right_proj = left_proj.dot(self.right)
        except NotImplementedError:
            return left_proj

        return right_proj

    def expand(self, original):
        "Expand a projected vector to the original basis"
        return self.right.dot(original)
