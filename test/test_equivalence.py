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
"Tests for equivalence class code"

from openmodes.helpers import equivalence

relations = ((0, 1), (1, 2), (2, 3), (3, 0),
             (7, 12), (12, 9), (9, 4))

print equivalence(relations)


relations = ((1, 2), (0, 1), (2, 3), (3, 3),
             (7, 12), (12, 9), (9, 4),
            ("f", 14), ("h", "f"), (14, 15))

print equivalence(relations)
