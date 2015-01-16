# distutils: language = c++
# cython: boundscheck=False, wraparound=False

import numpy as np
cimport numpy as np

cdef extern from "TaylorDuffy.h" nogil:
    cdef cppclass TaylorDuffyArgStruct:
        # mandatory input fields 
        int WhichCase

        double *V1, *V2, *V3
        double *V2P, *V3P

        int NumPKs
        int *PIndex
        int *KIndex
        double complex *KParam

        # output fields
        double complex *Result, *Error
        int nCalls

        # optional input fields
        double *Q, *QP
        double *nHat

        double AbsTol, RelTol
        int MaxEval
        int ForceOnceIntegrable

    # values for the WhichK parameter to the Taylor_xx routines
    cdef enum:
        TD_RP                   #0
        TD_HELMHOLTZ            #1
        TD_GRADHELMHOLTZ        #2
        TD_HIGHK_HELMHOLTZ      #3
        TD_HIGHK_GRADHELMHOLTZ  #4
        NUMKS                   #5

    # values for the WhichP parameter to the Taylor_xx routines
    cdef enum:
        TD_UNITY               #0
        TD_RNORMAL             #1
        TD_PMCHWG1             #2
        TD_PMCHWC              #3
        TD_NMULLERG1           #4
        TD_NMULLERG2           #5
        TD_NMULLERC            #6
        NUMPS                  #7

    # values for the WhichCase parameter. note the values correspond to
    # the numbers of common vertices in each case.
    cdef enum:
        TD_COMMONVERTEX        #1
        TD_COMMONEDGE          #2
        TD_COMMONTRIANGLE      #3        

    void InitTaylorDuffyArgs(TaylorDuffyArgStruct *Args)
    void TaylorDuffy(TaylorDuffyArgStruct *Args)


# Standard C++ routines for dealing with complex numbers
cdef extern from "complex" nogil:
    double real(double complex)
    double imag(double complex)


def taylor_duffy(double[:, ::1] nodes_o, double[:, ::1] nodes_s,
                 int unique_s_1 = -1, int unique_s_2 = -1, int max_eval=1000,
                 double rel_tol = 1e-10):

    cdef np.ndarray[np.float64_t, ndim=2] res_A_np = np.empty((3, 3), np.float64)
    cdef double[:, ::1] res_A = res_A_np

    cdef TaylorDuffyArgStruct TDArgs
    # output buffers
    cdef double complex Result[1], Error[1]

    cdef int count_o, count_s
    cdef double res_phi

    with nogil:
        InitTaylorDuffyArgs(&TDArgs)

        # Determine how many nodes are in common
        if unique_s_2 == -1:
            if unique_s_1 == -1:
                TDArgs.WhichCase = TD_COMMONTRIANGLE
            else:
                TDArgs.WhichCase = TD_COMMONEDGE
                TDArgs.V3P = &nodes_s[unique_s_1, 0]
        else:
            TDArgs.WhichCase = TD_COMMONVERTEX
            TDArgs.V2P = &nodes_s[unique_s_1, 0]
            TDArgs.V3P = &nodes_s[unique_s_2, 0]

        TDArgs.V1 = &nodes_o[0, 0]
        TDArgs.V2 = &nodes_o[1, 0]
        TDArgs.V3 = &nodes_o[2, 0]

        # specification of which integrals we want
        TDArgs.NumPKs = 1
        TDArgs.KIndex = [TD_RP]
        TDArgs.KParam = [ -1.0+0j]

        TDArgs.Result = Result
        TDArgs.Error = Error 

        TDArgs.MaxEval = max_eval
        TDArgs.RelTol = rel_tol

        # evaluate the scalar potential term
        TDArgs.PIndex = [TD_UNITY]
        TaylorDuffy( &TDArgs )
        res_phi = real(Result[0])

        # evaluate the vector potential terms
        TDArgs.PIndex = [TD_PMCHWG1]
        for count_o in xrange(3):
            TDArgs.Q = &nodes_o[count_o, 0]
            for count_s in xrange(3):
                TDArgs.QP = &nodes_s[count_s, 0]
                TaylorDuffy( &TDArgs )
                res_A[count_o, count_s] = real(Result[0])

    return res_A_np, res_phi
