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
from openmodes.basis import BasisContainer, MacroBasis
from openmodes.helpers import cached_property


def is_real_pole(s):
    """Apply a threshold to determine if a pole should be treated as purely
    real, in which case its conjugate should not be used in models"""
    return abs(s.imag) < 1e-3*abs(s.real)


class AbstractModes(object):
    """A class for holding a set of modes, enabling a matrix or vector to be
    easily projected onto them"""

    def __init__(self, parent_part, modes_of_parts, operator,
                 orig_container, macro_container=None):
        """
        parent_part: Part
            The Part containing all parts for which modes are defined, but no
            other parts.
        modes_of_parts: dictionary
            keys are Parts
            values are dictionaries, with elements
                's': array of freqs, 'vl', 'vr', arrays of left and right
                eigenvectors. Note that these should be simple 2D numpy arrays,
                not LookupArrays
        unknowns, sources: tuple of str
            The variable names for the unknown and source quantities
        orig_container: BasisContainer
            The container for the basis functions used to find the modes
        macro_container: BasisContainer, optional
            A container for the macro basis functions. Should be provided if
            there is an existing container which should be re-used.
        """
        self.parent_part = parent_part
        self.operator = operator
        self.orig_container = orig_container
        self.modes_of_parts = modes_of_parts

        # a container for the macro basis functions
        if macro_container is None:
            macro_container = BasisContainer(MacroBasis, global_args = {'modes_of_parts': modes_of_parts})
            macro_container.lowest_parts = set(modes_of_parts.keys())

        self.macro_container = macro_container

    def __len__(self):
        return sum(len(modes['s']) for modes in self.modes_of_parts.values())

    @cached_property
    def s(self):
        res = LookupArray((('modes',), (self.parent_part, self.macro_container)),
                         dtype=np.complex128)
        for part, modes in self.modes_of_parts.items():
            res[:, part] = modes['s']
        return res

    @cached_property
    def vr(self):
        "The right eigenvectors"

        res = LookupArray((self.operator.unknowns, (self.parent_part, self.orig_container),
                          ('modes',), (self.parent_part, self.macro_container)),
                         dtype=np.complex128)
        res[:] = 0.0

        for part, modes in self.modes_of_parts.items():
            res[:, part, :, part] = modes['vr'].reshape(res[:, part, :, part].shape)
        return res

    @cached_property
    def vl(self):
        "The left eigenvectors"
        res = LookupArray((('modes',), (self.parent_part, self.macro_container),
                          self.operator.sources, (self.parent_part, self.orig_container)),
                         dtype=np.complex128)
        res[:] = 0.0

        for part, modes in self.modes_of_parts.items():
            res[:, part, :, part] = modes['vl'].reshape(res[:, part, :, part].shape)

        return res

    def __getitem__(self, part):
        "Get the modes for one of the sub-parts"

        # TODO: implement for an intermediate level sub-part for which modes
        # were not calculated directly, only of its children.
        sub_modes = {part: self.modes_of_parts[part]}
        return Modes(part, sub_modes, self.operator,
                     self.orig_container, self.macro_container)

    def select(self, criteria):
        """Select a sub-set of modes based on the given criteria

        Parameters
        ----------
        criteria: list, of dict(Part, list)
            Typically this will be a list of desired mode numbers
            If there are multiple parts, then this can instead be a dictionary
            with different criteria per Part.
        """

        # for now, criteria is just a list of mode numbers
        new = {}
        for part, original in self.modes_of_parts.items():
            if isinstance(criteria, dict):
                part_criteria = criteria[part]
            elif isinstance(criteria, LookupArray):
                part_criteria = criteria[:, part][0]
            else:
                part_criteria = criteria

            new[part] = {}
            new[part]['s'] = original['s'][part_criteria]
            new[part]['vr'] = original['vr'][:, part_criteria]
            new[part]['vl'] = original['vl'][part_criteria, :]

        return self.__class__(self.parent_part, new, self.operator, self.orig_container)


class Modes(AbstractModes):

    def add_conjugates(self):
        """Create a new set of modes including the conjugate poles, with the
        nearly real poles to be exactly real. Does not check whether conjugate
        poles have already been added"""

        new = {}
        for part, original in self.modes_of_parts.items():
            # first attempt for a single part
            real_poles = is_real_pole(original['s'])
            complex_poles = np.logical_not(real_poles)

            new[part] = {}
            new[part]['s'] = np.hstack((original['s'][real_poles].real,
                                        original['s'][complex_poles],
                                        original['s'][complex_poles].conj()))

            new[part]['vr'] = np.hstack((original['vr'][:, real_poles],
                                         original['vr'][:, complex_poles],
                                         original['vr'][:, complex_poles].conj()))

            new[part]['vl'] = np.vstack((original['vl'][real_poles, :],
                                         original['vl'][complex_poles, :],
                                         original['vl'][complex_poles, :].conj()))

        return ConjugateModes(self.parent_part, new, self.operator,
                              self.orig_container)

    def split_real_imag(self):
        """Create a new set of basis currents by splitting the real and
        imaginary currents of each mode"""

        new = {}
        for part, original in self.modes_of_parts.items():
            # first attempt for a single part
            real_poles = is_real_pole(original['s'])
            complex_poles = np.logical_not(real_poles)

            new[part] = {}
            new[part]['s'] = np.hstack((original['s'][real_poles].real,
                                        original['s'][complex_poles].real,
                                        np.zeros_like(original['s'][real_poles].imag),
                                        original['s'][complex_poles].imag))

            new[part]['vr'] = np.hstack((original['vr'][:, real_poles].real,
                                         original['vr'][:, complex_poles].real,
                                         np.zeros_like(original['vr'][:, real_poles].imag),
                                         original['vr'][:, complex_poles].imag))

            new[part]['vl'] = np.vstack((original['vl'][real_poles, :].real,
                                         original['vl'][complex_poles, :].real,
                                         np.zeros_like(original['vl'][real_poles, :].imag),
                                         original['vl'][complex_poles, :].imag))

        return SplitModes(self.parent_part, new, self.operator,
                          self.orig_container)


class ConjugateModes(AbstractModes):
    "A class holding modes along with their complex conjugates"


class SplitModes(AbstractModes):
    "A class holding modes by splitting them into real and imaginary parts"


def match_degenerate_modes(modes, threshold=1e-2):
    "Determine which modes are degenerate, within a certain threshold"
    # TODO: should only be done for a single Part

    s = modes.s[0]
    matched = []
    unmatched = range(len(s))

    while len(unmatched) > 0:
        current = unmatched.pop()
        ds = np.abs((s[current]-s[unmatched])/s[current])
        matches = np.where(ds < threshold)[0]
        current_group = [current]
        # Traverse in reverse order so that popping does not invalidate other
        # elements
        for m in reversed(matches):
            current_group.append(unmatched[m])
            unmatched.pop(m)
        matched.append(current_group)
    return matched