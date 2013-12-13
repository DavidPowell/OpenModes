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
Convert all the IPython notebook example worksheets into .rst files which
can be included in the sphinx documentation

As of IPython v1.1, the quality of the generated .rst files is very poor,
so a development version of IPython is recommended.

Files are renamed from .rst to .txt
"""

import os.path as osp
import os
import subprocess

os.chdir(osp.join("doc", "examples"))

example_dir = osp.join("..", "..", "examples")

notebooks =[f for f in os.listdir(example_dir) if f.endswith(".ipynb")]

for notebook in notebooks:
    print notebook
    subprocess.call(["ipython", "nbconvert", "--to", "rst", osp.join(example_dir, notebook)])
    name_root = osp.splitext(notebook)[0]
    os.rename(name_root+".rst", name_root+".txt")
   
