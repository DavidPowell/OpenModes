# distutils: language = c++
# cython: boundscheck=False, wraparound=False

import numpy as np
cimport numpy as np
from numpy import pi

cdef extern from "TaylorDuffy.h" nogil:
    cdef cppclass TaylorDuffyArgStruct:
        # mandatory input fields 
        int WhichCase

        double *V1
        double *V2
        double *V3
        double *V2P
        double *V3P

        int NumPKs
        int *PIndex
        int *KIndex
        double complex *KParam

        # output fields
        double complex *Result
        double complex *Error
        int nCalls

        # optional input fields
        double *Q
        double *QP
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

# An easy way to set the relevant vertex pointers
cdef void set_vertex(TaylorDuffyArgStruct *Args, int which_vertex, double* address) nogil:
    if which_vertex == 0:
        Args.V1 = address
    elif which_vertex == 1:
        Args.V2 = address
    elif which_vertex == 2:
        Args.V3 = address
    elif which_vertex == 3:
        # note that V3P must be set before V2P!
        Args.V3P = address
    else:
        Args.V2P = address

# Definitions for which form of singularity to extract
cpdef enum:
    SING_T_EFIE
    SING_N_EFIE
    SING_T_MFIE
    SING_N_MFIE

def taylor_duffy(double[:, ::1] nodes, int[::1] triangle_o,
                 int[::1] triangle_s, int which_form, int max_eval=1000,
                 double rel_tol = 1e-10):

    cdef np.ndarray[np.float64_t, ndim=2] res_A_np = np.empty((3, 3), np.float64)
    cdef double[:, ::1] res_A = res_A_np

    cdef TaylorDuffyArgStruct TDArgs
    # output buffers
    cdef double complex Result[1]
    cdef double complex Error[1]

    cdef int count_o, count_s
    cdef double res_phi

    cdef int obs_only[2]
    cdef int source_only[2]
    cdef int common[3]
    cdef int obs_only_count = 0, source_only_count = 0, common_count = 0

    cdef int v_count
    cdef int node

    if which_form not in (SING_T_EFIE, SING_N_MFIE):
        raise ValueError("Unkown singularity form")

    with nogil:
        InitTaylorDuffyArgs(&TDArgs)

        # Determine which nodes are in common, or observer only
        for count_o in xrange(3):
            node = triangle_o[count_o]
            if (node == triangle_s[0] or node == triangle_s[1] or
                node == triangle_s[2]):
                common[common_count] = node
                common_count += 1
            else:
                obs_only[obs_only_count] = node
                obs_only_count += 1

        # Determine which nodes are in the source only
        for count_o in xrange(3):
            node = triangle_s[count_o]
            if (node != triangle_o[0] and node != triangle_o[1] and
                node != triangle_o[2]):
                source_only[source_only_count] = node
                source_only_count += 1

        # Work out the relevant case
        if common_count == 1:
            TDArgs.WhichCase = TD_COMMONVERTEX
        elif common_count == 2:
            TDArgs.WhichCase = TD_COMMONEDGE
        else:
            TDArgs.WhichCase = TD_COMMONTRIANGLE
            
        # point the V pointers to the relevant nodes
        v_count = 0
        for count_o in xrange(common_count):
            set_vertex(&TDArgs, v_count, &nodes[common[count_o], 0])
            v_count += 1
            
        for count_o in xrange(obs_only_count):
            set_vertex(&TDArgs, v_count, &nodes[obs_only[count_o], 0])
            v_count += 1

        for count_o in xrange(source_only_count):
            set_vertex(&TDArgs, v_count, &nodes[source_only[count_o], 0])
            v_count += 1
            
        # specification of which integrals we want
        TDArgs.NumPKs = 1
        TDArgs.KIndex = [TD_RP]
        TDArgs.KParam = [ -1.0+0j]

        TDArgs.Result = Result
        TDArgs.Error = Error 

        TDArgs.MaxEval = max_eval
        TDArgs.RelTol = rel_tol

        # evaluate the scalar potential term if using EFIE
        if which_form == SING_T_EFIE:
            TDArgs.PIndex = [TD_UNITY]
            TaylorDuffy( &TDArgs )
            res_phi = real(Result[0])

        # choose which vector potential term to return
        if which_form == SING_T_EFIE:
            TDArgs.PIndex = [TD_PMCHWG1]
        elif which_form == SING_N_MFIE:
            TDArgs.PIndex = [TD_NMULLERC]
            
        # evaluate the vector potential terms
        for count_o in xrange(3):
            TDArgs.Q = &nodes[triangle_o[count_o], 0]
            for count_s in xrange(3):
                TDArgs.QP = &nodes[triangle_s[count_s], 0]
                TaylorDuffy( &TDArgs )
                res_A[count_o, count_s] = real(Result[0])

    if which_form in (SING_T_EFIE, SING_N_EFIE):
        return res_phi*4*pi, res_A_np*4*pi
    else:
        return res_A_np*4*pi
