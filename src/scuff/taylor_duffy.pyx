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
    OPERATOR_EFIE
    OPERATOR_MFIE

cdef extern from "math.h" nogil:
    double M_PI

cdef enum:
    MAX_TERMS = 2

def taylor_duffy(double[:, ::1] nodes, int[::1] triangle_o,
                 int[::1] triangle_s, int which_operator, bint tangential_form,
                 int num_terms, double[::1] normal = None,
                 int max_eval=1000, double rel_tol = 1e-10):
    """Calculate singular terms for RWG basis functions using the Taylor-Duffy 
    method. Wraps C++ code from scuff-EM.
    
    Parameters
    ----------
    nodes: ndarray(num_nodes, 3) of double
        A list containing a set of unique nodes
    triangle_o: ndarray(3) of int
        The node indices of the observer triangle
    triangle_s: ndarray(3) of int
        The node indices of the source triangle
    which_operator: int
        Either OPERATOR_EFIE or OPERATOR MFIE specifying whether the EFIE or
        MFIE operator terms are required
    tangential_form: bool
        If True, the tangential form of the operator is used, otherwise the
        n x form is used
    normal: array(3) of double
        The normal to the observer triangle
    num_terms: int
        The number of singular terms to extract
    max_eval: int
        The maximum number of times to evaulate the function during
        adaptive integration
    rel_tol: double
        The desired relative tolerance in the integrals
    """

    # The output arrays, and memoryviews into them for speed
    cdef np.ndarray[np.float64_t, ndim=3] res_A_np = np.empty((num_terms, 3, 3), np.float64)
    cdef double[:, :, ::1] res_A = res_A_np
    
    cdef np.ndarray[np.float64_t, ndim=1] res_phi_np = np.empty(num_terms, np.float64)
    cdef double[::1] res_phi = res_phi_np

    cdef TaylorDuffyArgStruct TDArgs
    
    # input buffers
    cdef int PIndex[MAX_TERMS]
    cdef int KIndex[MAX_TERMS]
    cdef double complex KParam[MAX_TERMS]
    
    # output buffers
    cdef double complex Result[MAX_TERMS]
    cdef double complex Error[MAX_TERMS]

    # counters to sort out which node belongs to which triangle(s) and to
    # order them correctly
    cdef int obs_only[2]
    cdef int source_only[2]
    cdef int common[3]
    cdef int obs_only_count = 0, source_only_count = 0, common_count = 0

    cdef int v_count
    cdef int node
    cdef int n, start_term
    cdef int count_o, count_s

    if which_operator not in (OPERATOR_EFIE, OPERATOR_MFIE):
        raise ValueError("Unkown operator %s" % str(which_operator))
        
    if num_terms > MAX_TERMS:
        raise ValueError("Too many singular terms: %d" % num_terms)

    if not tangential_form and (normal is None):
        raise ValueError("Normal required for n x operator form")

    with nogil:
        InitTaylorDuffyArgs(&TDArgs)

        # Determine which nodes are in common, or observer only
        for count_o in xrange(3):
            node = triangle_o[count_o]
            if node in (triangle_s[0], triangle_s[1], triangle_s[2]):
                common[common_count] = node
                common_count += 1
            else:
                obs_only[obs_only_count] = node
                obs_only_count += 1

        # Determine which nodes are in the source only
        for count_s in xrange(3):
            node = triangle_s[count_s]
            if node not in (triangle_o[0], triangle_o[1], triangle_o[2]):
                source_only[source_only_count] = node
                source_only_count += 1

        # The relevant case is just the number of common vertices
        TDArgs.WhichCase = common_count
            
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
        TDArgs.NumPKs = num_terms
        
        TDArgs.KIndex = KIndex
        for n in xrange(num_terms):
            KIndex[n] = TD_RP

        TDArgs.PIndex = PIndex
        TDArgs.KParam = KParam

        TDArgs.Result = Result
        TDArgs.Error = Error 

        TDArgs.MaxEval = max_eval
        TDArgs.RelTol = rel_tol

        # polynomial terms are -1, 1, for EFIE, -3, -1 for MFIE
        if which_operator == OPERATOR_EFIE:
            start_term = -1
        else:
            start_term = -3
            
        for n in xrange(num_terms):
            KParam[n] = <double complex>(start_term+2*n)

        # evaluate the scalar potential term if using EFIE
        if which_operator == OPERATOR_EFIE:
            for n in xrange(num_terms):
                PIndex[n] = TD_UNITY

            TaylorDuffy( &TDArgs )
            for n in xrange(num_terms):
                res_phi[n] = real(Result[n])*4*M_PI

        # Check the normals, and pass them
        if not tangential_form:
            TDArgs.nHat = &normal[0]

        # choose which vector potential term to calculate
        for n in xrange(num_terms):
            if which_operator == OPERATOR_EFIE:
                if tangential_form:
                    PIndex[n] = TD_PMCHWG1
                else:
                    PIndex[n] = TD_NMULLERG1
            else:
                if tangential_form:
                    PIndex[n] = TD_PMCHWC
                else:
                    PIndex[n] = TD_NMULLERC
            
        # evaluate the vector potential terms
        for count_o in xrange(3):
            TDArgs.Q = &nodes[triangle_o[count_o], 0]
            for count_s in xrange(3):
                TDArgs.QP = &nodes[triangle_s[count_s], 0]
                TaylorDuffy( &TDArgs )
                for n in xrange(num_terms):
                    res_A[n, count_o, count_s] = real(Result[n])*4*M_PI

    if which_operator == OPERATOR_EFIE:
        return res_phi_np, res_A_np
    else:
        return (res_A_np,)
