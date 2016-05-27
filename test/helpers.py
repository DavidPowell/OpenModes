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
import os.path as osp
import os
import pickle
from numpy.testing import assert_allclose


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


def get_test_dir(tests_filename):
    location, fname = osp.split(tests_filename)
    fname = osp.splitext(fname)[0]
    test_dir = osp.join(location, 'reference', fname)
    if not osp.exists(test_dir):
        os.makedirs(test_dir)
    return test_dir


def get_input_dir(tests_filename, name=None):
    location, fname = osp.split(tests_filename)
    fname = osp.splitext(fname)[0]
    input_dir = osp.join(location, 'input', fname)
    if name is not None:
        input_dir = osp.join(input_dir, name)
    if not osp.exists(input_dir):
        os.makedirs(input_dir)
    return input_dir


def compare_ref(val, reference, rtol, name):
    "Compare a value against a stored reference"
    if isinstance(val, np.ndarray):
        # Compare numpy arrays
        assert_allclose(val, reference, rtol=rtol,
                        err_msg='Result "{}" differs from reference'.format(name))
    elif isinstance(val, dict):
        # Compare dictionaries, ensuring keys and all values are the same
        assert(set(val.keys()) == set(reference.keys()))
        for key in val.keys():
            compare_ref(val[key], reference[key], rtol, "{}[{}]".format(name, key))
    else:
        raise ValueError("Unable to compare {} of type {} to reference".format(name, type(val)))


def run_test(func, tests_filename):
    "Run a test function and check the output against reference"
    results = func()
    test_dir = get_test_dir(tests_filename)

    with open(osp.join(test_dir, results['name'])+".pickle", "rb") as infile:
        reference = pickle.load(infile)

    assert(set(reference.keys()) == set(results['results'].keys()))
    for name, val in results['results'].items():
        try:
            rtol = results['rtol'][name]
        except:
            rtol = 1e-7

        compare_ref(val, reference[name], rtol, name)


def create_reference(func, tests_filename):
    "Run a test function and generate reference output"
    results = func()
    test_dir = get_test_dir(tests_filename)

    with open(osp.join(test_dir, results['name'])+".pickle", "wb") as outfile:
        pickle.dump(results['results'], outfile, protocol=2, fix_imports=True)
