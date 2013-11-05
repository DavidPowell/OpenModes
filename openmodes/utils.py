# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 18:17:43 2013

@author: dap124
"""

import numpy as np

# the real data type which is used by fortran code
#def fortran_real_type():
#    return np.float64
#    #return np.float32
    
#f_real = fortran_real_type()
fortran_real_type = np.float64


#class observable(object):
#    """A simple implementation of the observer pattern which provides no
#    details of the event
#    
#    adapted from http://stackoverflow.com/a/1925836/482420
#    
#    Also, the callbacks are destroyed if the object is pickled
#    """
#    def __init__(self):
#        self.callbacks = []
#        
#    def add_observer(self, callback):
#        self.callbacks.append(callback)
#        
#    def notify(self):
#        for callback in self.callbacks:
#            callback()
#            
#    def __getstate__(self):
#        d = dict(self.__dict__)
#        del d['callbacks']
#        return d
        
        
class SingularSparse(object):
    """A sparse matrix class for holding A and phi arrays with the same 
    sparsity pattern to store singular triangle impedances"""
    def __init__(self):
        self.rows = {}
        
    def __setitem__(self, index, item):
        """Add an item, which will be stored in a dictionary of dictionaries. 
        Item is assumed to be (A, phi)"""
        
        row, col = index
        try:
            self.rows[row][col] = item
        except KeyError:
            self.rows[row] = {col: item}
    
    def iteritems(self):
        for row, row_dict in self.rows.iteritems():
            for col, item in row_dict.iteritems():
                yield ((row, col), item)
    
    def tocsr(self):
        """Convert the matrix to compressed sparse row format, with 
        common index array and two data arrays for A and phi"""
        A_data = []
        phi_data = []
        indices = []
        indptr = [0]
        
        data_index = 0

        num_rows = max(self.rows.keys())+1

        for row in xrange(num_rows):
            if row in self.rows:
                # the row exists, so process it
                for col, item in self.rows[row].iteritems():
                    A_data.append(item[0])
                    phi_data.append(item[1])
                    indices.append(col)
                    
                    data_index = data_index + 1
            # regardless of whether the row exists, update the index pointer
            indptr.append(data_index)
            
        
        return (np.array(A_data, dtype=fortran_real_type, order="F"), 
                np.array(phi_data, dtype=fortran_real_type, order="F"),
                np.array(indices, dtype=np.int32, order="F"), 
                np.array(indptr, dtype=np.int32, order="F"))
