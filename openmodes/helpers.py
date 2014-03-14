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

import functools


def inc_slice(s, inc):
    """Increment a slice so that it starts at the current stop, and the current
    stop is incremented by some amount"""
    return slice(s.stop, s.stop+inc)


def cached_property(f):
    """Wrap a function which represents some property that is expensive to
    calculate but is likely to be reused."""

    f.cache = {}

    @functools.wraps(f)
    def caching_wrapper(*args, **kwargs):
        key = str(args) + str(kwargs)
        # print key
        if key not in f.cache:
            f.cache[key] = f(*args, **kwargs)
        return f.cache[key]

    return property(caching_wrapper)


class MeshError(Exception):
    "An exeception indicating a failure generating or reading the mesh"
    pass
