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
import uuid
import weakref
import numpy as np
import numbers


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


class Identified(object):
    "An object which can be uniquely identified by an id number"

    def __init__(self):
        self.id = uuid.uuid4()

    def __hash__(self):
        return self.id.__hash__()

    def __eq__(self, other):
        return hasattr(other, 'id') and (self.id == other.id)

    def __repr__(self):
        "Represent the object by its id, in addition to its memory address"
        return ("<%s at 0x%08x with id %s>" % (str(self.__class__)[8:-2],
                                               id(self),
                                               str(self.id)))


class PicklableRef(object):
    """A weak reference which can be pickled. This is achieved by
    creating a strong reference to the object at pickling time, then restoring
    the weak reference when unpickling. Note that unless the object being
    referenced is also pickled and referenced after unpickling, the weak
    reference will be dead after unpickling.
    """

    def __init__(self, obj, callback=None):
        self.ref = weakref.ref(obj, callback)

    def __call__(self):
        return self.ref()

    def __getstate__(self):
        return {'ref': self.ref()}

    def __setstate__(self, state):
        self.ref = weakref.ref(state['ref'])


def memoize(obj):
    """A simple decorator to memoize function calls. Pays particular attention
    to numpy arrays and objects which are subclasses of Identified. It is
    assumed that in such cases, the object does not change if its `id` is the
    same"""
    cache = obj.cache = {}

    def get_key(item):
        if isinstance(item, (basestring, numbers.Number)):
            return item
        elif isinstance(item, Identified):
            return str(item.id)
        elif isinstance(item, np.ndarray):
            return item.tostring()
        else:
            return str(item)

    @functools.wraps(obj)
    def memoizer(*args, **kwargs):
        key_arg = tuple(get_key(arg) for arg in args)
        key_kwarg = tuple((kw, get_key(arg)) for (kw, arg)
                          in kwargs.iteritems())
        key = (key_arg, key_kwarg)

        if key not in cache:
            cache[key] = obj(*args, **kwargs)
        return cache[key]
    return memoizer
