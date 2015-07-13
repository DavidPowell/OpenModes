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

import os.path as osp
from openmodes.array import LookupArray
import openmodes
import numpy as np


def test_indexing():
    "Basic test for array indexing"

    sim = openmodes.Simulation()
    mesh = sim.load_mesh(osp.join(openmodes.geometry_dir, "SRR.geo"))

    group1 = sim.place_part()
    group2 = sim.place_part()

    srr1 = sim.place_part(mesh, parent=group1)
    srr2 = sim.place_part(mesh, parent=group1)
    srr3 = sim.place_part(mesh, parent=group2)

    basis = sim.basis_container[srr1]
    basis_len = len(basis)

    A = LookupArray(((group1, sim.basis_container),
                     (group2, sim.basis_container), 5, 3))

    assert(A.shape == (2*basis_len, basis_len, 5, 3))

    A[group1, srr3] = 22.5
    assert(np.all(A[srr1, :] == 22.5))
    assert(np.all(A[srr2] == 22.5))

    V = LookupArray((("E", "H"), (sim.parts, sim.basis_container)),
                    dtype=np.complex128)

    V["E", group1] = -4.7+22j
    V["H", srr1] = 5.2
    V["H", srr2] = 6.7

    assert(np.all(V["E", group1] == V["E"][group1]))
    assert(np.all(V["E", srr1] == -4.7+22j))
    assert(np.all(V["E", srr2].imag == 22))

if __name__ == "__main__":
    test_indexing()
