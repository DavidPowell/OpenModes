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
"""Helper routines for tests"""

import numpy as np


def write_1d_complex(filename, data):
    "Write a 1D complex array to a text file"
    with open(filename, "wt") as outfile:
        for d in data:
            outfile.write("%.8e %.8e\n" % (d.real, d.imag))


def read_1d_complex(filename):
    "Read a 1D complex array from a text file"

    data = []
    with open(filename, "rt") as infile:
        for d in infile:
            d = d.split()
            data.append(float(d[0])+1j*float(d[1]))

    return np.array(data, dtype=np.complex128)


def write_2d_real(filename, data):
    "Write a 2D real array to a text file"
    with open(filename, "wt") as outfile:
        for row in data:
            for d in row:
                outfile.write("%.8e " % d)
            outfile.write("\n")


def read_2d_real(filename):
    "Read a 2D real array from a text file"

    data = []
    with open(filename, "rt") as infile:
        for row in infile:
            row = row.split()
            data.append([float(d) for d in row])

    return np.array(data, dtype=np.float64)
