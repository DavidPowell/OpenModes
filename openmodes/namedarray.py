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
"""A custom class for indexing array elements with any hashable object"""

import numpy as np

class NamedArray(np.ndarray):
    """
    A subclass of a numpy array, where certain ranges can be specified by a
    a name, which can be any hashable type.
    
    For explanation of subclassing numpy arrays, see
    http://docs.scipy.org/doc/numpy/user/basics.subclassing.html
    """
    
    def __new__(subtype, shape, ranges, dtype=float):
        # NB: shape could also contain the data
        obj = np.ndarray.__new__(subtype, shape, dtype)
        
        if isinstance(ranges, dict):
            # force ranges to be a tuple
            ranges = ranges,
        elif not isinstance(ranges, tuple):
            raise ValueError("Invalid ranges")
        obj.ranges = ranges
        # Finally, we must return the newly created object:
        return obj

#    def __array_finalize__(self, obj):
#        "Function is called when creating array from view as well"
#        if obj is None: return
#        
#        # ranges are only valid when explictly creating a matrix. They are
#        # not added when taking a view
        
    def __getslice__(self, start, stop):
        "Needed due to CPython bug"
        return self.__getitem__(slice(start, stop))

    def __setslice__(self, start, stop, val):
        "Needed due to CPython bug"
        self.__setitem__(slice(start, stop), val)

    def __getitem__(self, idx):
        """Gets an item or items from the array. Any of the indices may be the
        name of a range, in addition to all the usual fancy indexing options"""
        if not hasattr(self, 'ranges'):
            # if ranges is not set, just behave like a normal array
            return super(NamedArray, self).__getitem__(idx)

        if not isinstance(idx, tuple):
            # force a single index to be a tuple
            idx = idx,
        new_idx = []

        # Find out how many fancy indices were used, to know what dimensions
        # to broadcase the array over
        num_ranges_used = sum(not isinstance(idn, slice) and (idn in sec) 
                              for (idn, sec) in zip(idx, self.ranges))

        for count, (idn, sec) in enumerate(zip(idx, self.ranges)):
            try:
                new_idx.append(sec[idn][(slice(None),)+(None,)*(num_ranges_used-count-1)])
            except (KeyError, TypeError):
                # index item not found in list
                new_idx.append(idn)
        return super(NamedArray, self).__getitem__(tuple(new_idx))


    def __setitem__(self, idx, value):
        """Gets an item or items in the array. Any of the indices may be the
        name of a range, in addition to all the usual fancy indexing options"""
        if not hasattr(self, 'ranges'):
            # if ranges is not set, just behave like a normal array
            return super(NamedArray, self).__setitem__(idx, value)

        if not isinstance(idx, tuple):
            # force a single index to be a tuple
            idx = idx,
        new_idx = []

        # Find out how many fancy indices were used, to know what dimensions
        # to broadcase the array over
        num_ranges_used = sum(not isinstance(idn, slice) and (idn in sec) 
                              for (idn, sec) in zip(idx, self.ranges))

        for count, (idn, sec) in enumerate(zip(idx, self.ranges)):
            try:
                new_idx.append(sec[idn][(slice(None),)+(None,)*(num_ranges_used-count-1)])
            except (KeyError, TypeError):
                # index item not found in list
                new_idx.append(idn)
        super(NamedArray, self).__setitem__(tuple(new_idx), value)

if __name__ == "__main__":
    ranges = {'A' : np.array([0, 1, 2, 3, 8, 9, 10]),
                'B' : np.array([4, 5, 6, 7, 11])}
                
    arr = NamedArray(12, ranges)
    arr[:] = np.arange(12)
    print(arr)
    arr['B'] *= -2
    print(arr)
    
    ranges = ({'A' : np.array([0, 1, 2]), 'B' : np.array([3,4])},
              {'C' : np.array([0, 2]), 'D' : np.array([1, 3])})
    b = NamedArray((5, 4), ranges)
    b[:] = np.arange(20).reshape(5, 4)
    print(b)
    print(b['A', 'C'])
    b['A', 'D'] = [[-1, -2], [-3, -4], [-5, -6]]
    print(b)

