# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 15:06:48 2013

@author: dap124
"""
import numpy as np

#cimport "complex.h"
# this defines the external function's interface.  Why the out needs
# to be defined as 'int *out' and not 'int **out' I do not know...
cdef extern:
    void test_wrapped(float complex a) nogil

def test(complex a):
    test_wrapped(a)
