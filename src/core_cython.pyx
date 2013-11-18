# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 10:23:51 2013

@author: dap124
"""

# C99 routines for dealing with complex numbers
cdef extern from "complex.h" nogil:
    double creal(double complex)
    double cimag(double complex)
    double complex _Complex_I

import cython

from cython.parallel import prange, parallel
from cython.view cimport contiguous

cimport openmp

import numpy as np
cimport numpy as np

from libc.math cimport exp, sin, cos
from libc.stdlib cimport malloc, free

# under mingw32, it seems that complex functions are not implemented, so 
# synthesize them from real functions
cdef double complex cexp(double complex z) nogil:
    #cdef double r, i
    return exp(creal(z))*(cos(cimag(z)) + _Complex_I*sin(cimag(z)))

ctypedef np.int_t int_t
ctypedef np.complex128_t complex_t
ctypedef np.float64_t real_t

cdef class IrregularIntArray:
    """An array equivalent to a list of lists of integers
    
    This structure does not support adding or subtracting elements, so for the
    creation process build a list of lists, then convert to IrregularIntArray
    """
    
    cdef int total_length, num_lists
    cdef int* item_storage
    cdef int* offset_storage
    
    def __cinit__(self, list_of_lists):
        cdef int item_counter, list_counter, offset
        self.total_length = sum(len(l) for l in list_of_lists)
        self.num_lists = len(list_of_lists)
        
        self.offset_storage = <int *>malloc((self.num_lists+1)*sizeof(int))
        self.item_storage = <int *>malloc(self.total_length*sizeof(int))
        
        offset = 0
        for list_counter, current_list in enumerate(list_of_lists):
            self.offset_storage[list_counter] = offset
            for item in current_list:
                self.item_storage[offset] = item
                offset += 1
                
        # put a reference beyond the array end to enable safe checking of offsets
        self.offset_storage[self.num_lists+1] = self.total_length+1
      
    def __getitem__(self, item):
        # does not check for negative indices
        cdef int which_list, which_item, index
        which_list, which_item = item
        assert(which_list < self.num_lists) # "Invalid list number specified")
        index = self.offset_storage[which_list] + which_item
        assert(index < self.offset_storage[which_list+1])#, "Invalid list offset specified")
        return self.item_storage[index]
          
    def __dealloc__(self):
        free(<void*>self.item_storage)
        free(<void*>self.offset_storage)
                
#@cython.boundscheck(False)
#@cython.wraparound(False)
#def triangle_face_to_loop_star(IrregularIntArray basis_tri_p, 
#                               IrregularIntArray basis_tri_m, 
#                               IrregularIntArray basis_node_p, 
#                               IrregularIntArray basis_node_m, 
#                         complex[:, :, :, :] vector_face,
#                         complex[:, :] scalar_face):
#    """Take interaction terms which are defined between triangle
#    faces and convert them to a loop-star basis"""
#
#    cdef int m, n, p_p, p_m, q_p, q_m, ip_p, ip_m, iq_p, iq_m
#
#    cdef int num_triangles = vector_face.shape[0]
#    cdef int num_basis = basis_tri_p.shape[0]
#
#    cdef np.ndarray[complex_t, ndim=2] vector_rwg = np.zeros([num_basis, num_basis], dtype=np.complex128) 
#    #cdef complex[:, ::1] vector_rwg = np.empty([num_basis, num_basis], dtype=np.complex128) 
#    cdef np.ndarray[complex_t, ndim=2] scalar_rwg = np.zeros([num_basis, num_basis], dtype=np.complex128)
#    #cdef complex[:, ::1] scalar_rwg = np.empty([num_basis, num_basis], dtype=np.complex128)
#
#    with nogil:
#        for m in range(num_basis): # m is the index of the observer edge
#            # assume that there are an equal number of plus and minus triangles
#            # within each loop or star element
#            p_p = basis_tri_p[m]
#            p_m = basis_tri_m[m] # observer triangles
#    
#            ip_p = basis_node_p[m]
#            ip_m = basis_node_m[m] # observer unshared nodes
#            
#            for n in range(num_basis): # n is the index of the source
#                q_p = basis_tri_p[n]
#                q_m = basis_tri_m[n] # source triangles
#                
#                iq_p = basis_node_p[n]
#                iq_m = basis_node_m[n] # source unshared nodes
#    
#                vector_rwg[m, n] = ( 
#                      vector_face[p_p, q_p, ip_p, iq_p] - vector_face[p_p, q_m, ip_p, iq_m]
#                    - vector_face[p_m, q_p, ip_m, iq_p] + vector_face[p_m, q_m, ip_m, iq_m])
#                    
#                scalar_rwg[m, n] = (
#                    - scalar_face[p_m, q_p] + scalar_face[p_m, q_m] 
#                    + scalar_face[p_p, q_p] - scalar_face[p_p, q_m])
#
#    return vector_rwg, scalar_rwg
            


@cython.boundscheck(False)
@cython.wraparound(False)
def triangle_face_to_rwg(int[:] basis_tri_p, int[:] basis_tri_m, 
                         int[:] basis_node_p, int[:] basis_node_m, 
                         complex[:, :, :, :] vector_face,
                         complex[:, :] scalar_face):
    """Take interaction terms which are defined between triangle
    faces and convert them to RWG basis"""

    cdef int m, n, p_p, p_m, q_p, q_m, ip_p, ip_m, iq_p, iq_m

    cdef int num_triangles = vector_face.shape[0]
    cdef int num_basis = basis_tri_p.shape[0]

    cdef np.ndarray[complex_t, ndim=2] vector_rwg = np.empty([num_basis, num_basis], dtype=np.complex128) 
    #cdef complex[:, ::1] vector_rwg = np.empty([num_basis, num_basis], dtype=np.complex128) 
    cdef np.ndarray[complex_t, ndim=2] scalar_rwg = np.empty([num_basis, num_basis], dtype=np.complex128)
    #cdef complex[:, ::1] scalar_rwg = np.empty([num_basis, num_basis], dtype=np.complex128)

    with nogil:
        for m in range(num_basis): # m is the index of the observer edge
            p_p = basis_tri_p[m]
            p_m = basis_tri_m[m] # observer triangles
    
            ip_p = basis_node_p[m]
            ip_m = basis_node_m[m] # observer unshared nodes
            
            for n in range(num_basis): # n is the index of the source
                q_p = basis_tri_p[n]
                q_m = basis_tri_m[n] # source triangles
                
                iq_p = basis_node_p[n]
                iq_m = basis_node_m[n] # source unshared nodes
    
                vector_rwg[m, n] = ( 
                      vector_face[p_p, q_p, ip_p, iq_p] - vector_face[p_p, q_m, ip_p, iq_m]
                    - vector_face[p_m, q_p, ip_m, iq_p] + vector_face[p_m, q_m, ip_m, iq_m])
                    
                scalar_rwg[m, n] = (
                    - scalar_face[p_m, q_p] + scalar_face[p_m, q_m] 
                    + scalar_face[p_p, q_p] - scalar_face[p_p, q_m])

    return vector_rwg, scalar_rwg

@cython.boundscheck(False) 
@cython.wraparound(False)
cdef inline complex dot_product_complex_real(int length, complex* a, double* b) nogil:
    """An optimized dot-product for cython code"""
    cdef int counter
    cdef complex result = 0.0
    
    for counter in range(length):
        result += a[counter]*b[counter]
        
    return result
 
#from math cimport exp
    
#from cython cimport view
#from cython.view cimport array as cvarray
   
@cython.boundscheck(False) 
@cython.wraparound(False)
#cdef source_integral_plane_wave(float[:, :] xi_eta_o, float[:] weights_o, 
#                               float[:, :] nodes_o, complex[::1] jk_inc, complex[:] e_inc):
cdef inline void source_integral_plane_wave(double[:, ::contiguous] xi_eta_o, double[::contiguous] weights_o, 
                               double[:, ::contiguous] nodes_o, complex[::contiguous] jk_inc, 
                               complex[::contiguous] e_inc, complex[::contiguous] I) nogil:
    """ Inner product of source field with testing function to give source "voltage"
    !
    ! xi_eta_s/o - list of coordinate pairs in source/observer triangle
    ! weights_s/o - the integration weights of the source and observer
    ! nodes_s/o - the nodes of the source and observer triangles
    ! k_0 - free space wavenumber
    ! nodes - the position of the triangle nodes"""

    cdef double xi_o, eta_o, zeta_o, w_o
    
    cdef double r_o[3]
    cdef double rho_o[3][3]
    cdef complex e_r[3]

    cdef int n_o = xi_eta_o.shape[0]
    cdef int index_a, index_b, count_o
    cdef complex jkr, exp_jkr

    # way expensive!
    #cdef np.ndarray[complex_t, ndim=1] I = np.zeros(3, dtype=np.complex128) 
    I[:] = 0.0

    for count_o in range(n_o):

        w_o = weights_o[count_o]

        # Barycentric coordinates of the observer
        xi_o = xi_eta_o[count_o, 0]
        eta_o = xi_eta_o[count_o, 1]
        zeta_o = 1.0 - eta_o - xi_o

        # Cartesian coordinates of the observer
        for index_a in range(3):
            r_o[index_a] = xi_o*nodes_o[0, index_a] + eta_o*nodes_o[1, index_a] + zeta_o*nodes_o[2, index_a]
#        r_o[:] = xi_o*nodes_o[:, 0] + eta_o*nodes_o[:, 2] + zeta_o*nodes_o[:, 3]

        # Vector rho within the observer triangle
        for index_a in range(3):
            for index_b in range(3):
                rho_o[index_a][index_b] = r_o[index_b] - nodes_o[index_a, index_b]

        jkr = jk_inc[0]*r_o[0]+jk_inc[1]*r_o[1]+jk_inc[2]*r_o[2]
        #jkr = dot_product_complex_real(3, &jk_inc[0], &r_o[0])

        # calculate the incident electric field
        exp_jkr = cexp(-jkr)
        for index_a in range(3):
            e_r[index_a] = exp_jkr*e_inc[index_a]
            #e_r[index_a] = e_inc[index_a]


        #forall (uu=1:3) I(uu) = I(uu) + dot_product(rho_o(uu, :), e_r)*w_o
        for index_a in range(3):
            #I[index_a] += dot_product_complex_real(3, &e_r[0], &rho_o[index_a][0])*w_o
            I[index_a] += (e_r[0]*rho_o[index_a][0]+e_r[1]*rho_o[index_a][1]+e_r[2]*rho_o[index_a][2])*w_o

    return
    

@cython.boundscheck(False) 
@cython.wraparound(False)
def voltage_plane_wave_serial(double[:, ::contiguous] nodes, int[:, ::contiguous] triangle_nodes, 
                       int[::contiguous] basis_tri_p, int[::contiguous] basis_tri_m, 
                       int[::contiguous] basis_node_p, int[::contiguous] basis_node_m,
                       double[:, ::contiguous] xi_eta_eval, double[::contiguous] weights, 
                       complex[::contiguous] e_inc, complex[::contiguous] jk_inc):
    """
    ! Calculate the voltage term, assuming a plane-wave incidence
    !
    ! Note that this assumes a free-space background

    Currently this routine is ~20x slower than the corresponding fortran routine

    integer, intent(in) :: num_nodes, num_triangles, num_basis, num_integration

    real(WP), intent(in), dimension(0:num_nodes-1, 0:2) :: nodes
    integer, intent(in), dimension(0:num_triangles-1, 0:2) :: triangle_nodes
    integer, intent(in), dimension(0:num_basis-1) :: basis_tri_p
    integer, intent(in), dimension(0:num_basis-1) :: basis_tri_m
    integer, intent(in), dimension(0:num_basis-1) :: basis_node_p
    integer, intent(in), dimension(0:num_basis-1) :: basis_node_m

    real(WP), intent(in), dimension(0:num_integration-1, 0:1) :: xi_eta_eval
    real(WP), intent(in), dimension(0:num_integration-1) :: weights

    complex(WP), intent(in), dimension(3) :: jk_inc
    complex(WP), intent(in), dimension(3) :: e_inc

    complex(WP), intent(out), dimension(0:num_basis-1) :: V
    """

    #real(WP), dimension(0:2, 0:2) :: nodes_p, nodes_q
    #complex(WP), dimension(0:2, 0:num_triangles-1) :: V_face
    cdef int num_triangles = triangle_nodes.shape[0]
    cdef int num_basis = basis_tri_p.shape[0]
    #cdef view.array nodes_p
    #cdef int[3] which_nodes
    cdef int which_node
    #cdef float nodes_p[3][3]

    #cdef complex [::1] V = np.zeros(num_basis, dtype=np.complex128) 
    cdef np.ndarray[dtype=complex_t, ndim=1] V = np.zeros(num_basis, dtype=np.complex128)
    cdef complex[::contiguous] V_view = V

 
    #cdef np.ndarray[complex_t, ndim=2] V_face = np.zeros((num_triangles, 3), dtype=np.complex128) 
    cdef complex [:, ::contiguous] V_face = np.zeros((num_triangles, 3), dtype=np.complex128) 
    #cdef np.ndarray[float, ndim=2] nodes_p = np.zeros((3, 3), dtype=np.int)
    #cdef double thread_storage[openmp.omp_get_num_threads()][3][3] # statically allocate the storage - any point?
    #cdef double nodes_p_storage[3][3] # statically allocate the storage - any point?
    #cdef double* nodes_p_storage # statically allocate the storage - any point?
    #cdef double [:, :, :] nodes_p #= nodes_p_storage # assigning the view is always expensive
    cdef double nodes_p_storage[3][3]
    cdef double [:, ::contiguous] nodes_p = nodes_p_storage #np.empty((3, 3), dtype=np.float64) 
    #cdef view.array nodes_p = np.zeros((3, 3), dtype=np.int) 

    cdef int p, p_p, p_m, ip_p, ip_m, m, q
    #cdef int n_threads = openmp.omp_get_max_threads()
    #cdef int thread_id


    #with nogil, parallel():

    #nodes_p_storage = <double*>malloc(3*3*sizeof(double))

    #nodes_p = nodes_p_storage

    #with gil:

    #nodes_p_storage = <double*>malloc(n_threads*3*3*sizeof(double))
    #nodes_p = <double[:n_threads,:3,:3]> nodes_p_storage


    # calculate all the integrations for each face pair
    #$OMP PARALLEL DO SCHEDULE(DYNAMIC) DEFAULT(SHARED) &
    #$OMP PRIVATE (p, nodes_p)
    #for p in range(num_triangles): # p is the index of the observer face:
    #with nogil:
    for p in range(num_triangles): # p is the index of the observer face:
        for m in range(3): # m is the index of the observer face's nodes
            #which_nodes[m] = triangle_nodes[p, m]
            #which_node = triangle_nodes[p, m]
            for q in range(3): # index of the cartesian component
                #nodes_p[m, q] = nodes[which_node, q]
                nodes_p[m, q] = nodes[triangle_nodes[p, m], q]
                #nodes_p[m][q] = nodes[triangle_nodes[p, m], q]
            
        # perform testing of the incident field
        #V_face[p, :] = source_integral_plane_wave(xi_eta_eval, weights, nodes_p, jk_inc, e_inc)
        source_integral_plane_wave(xi_eta_eval, weights, nodes_p, jk_inc, e_inc, V_face[p, :])

    # now build up the source vector in terms of the basis vectors
    for m in range(num_basis): # m is the index of the observer edge
        p_p = basis_tri_p[m]
        p_m = basis_tri_m[m] # observer triangles

        ip_p = basis_node_p[m]
        ip_m = basis_node_m[m] # observer unshared nodes

        #V[m] = (V_face[ip_p, p_p]-V_face[ip_m, p_m])
        V_view[m] = (V_face[p_p, ip_p]-V_face[p_m, ip_m])

    #free(nodes_p_storage)    



    return V

@cython.boundscheck(False) 
@cython.wraparound(False)
#cdef source_integral_plane_wave(float[:, :] xi_eta_o, float[:] weights_o, 
#                               float[:, :] nodes_o, complex[::1] jk_inc, complex[:] e_inc):
cdef inline void source_integral_plane_wave_pointer(int n_o, double * xi_eta_o, double* weights_o, 
                               double* nodes_o, complex* jk_inc, 
                               complex* e_inc, complex* I) nogil:
    """ Inner product of source field with testing function to give source "voltage"
    !
    ! xi_eta_s/o - list of coordinate pairs in source/observer triangle
    ! weights_s/o - the integration weights of the source and observer
    ! nodes_s/o - the nodes of the source and observer triangles
    ! k_0 - free space wavenumber
    ! nodes - the position of the triangle nodes"""

    cdef double xi_o, eta_o, zeta_o, w_o
    
    cdef double r_o[3]
    cdef double rho_o[3][3]
    cdef complex e_r[3]

    #cdef int n_o = xi_eta_o.shape[0]
    cdef int index_a, index_b, count_o
    cdef complex jkr, exp_jkr

    # way expensive!
    #cdef np.ndarray[complex_t, ndim=1] I = np.zeros(3, dtype=np.complex128) 
    #I[:] = 0.0

    for count_o in range(n_o):

        w_o = weights_o[count_o]

        # Barycentric coordinates of the observer
        xi_o = xi_eta_o[count_o*2+0]
        eta_o = xi_eta_o[count_o*2+1]
        zeta_o = 1.0 - eta_o - xi_o

        # Cartesian coordinates of the observer
        for index_a in range(3):
            r_o[index_a] = xi_o*nodes_o[0*3+index_a] + eta_o*nodes_o[1*3+index_a] + zeta_o*nodes_o[2*3+index_a]
#        r_o[:] = xi_o*nodes_o[:, 0] + eta_o*nodes_o[:, 2] + zeta_o*nodes_o[:, 3]

        # Vector rho within the observer triangle
        for index_a in range(3):
            for index_b in range(3):
                rho_o[index_a][index_b] = r_o[index_b] - nodes_o[index_a*3+index_b]

        jkr = jk_inc[0]*r_o[0]+jk_inc[1]*r_o[1]+jk_inc[2]*r_o[2]
        #jkr = dot_product_complex_real(3, &jk_inc[0], &r_o[0])

        # calculate the incident electric field
        exp_jkr = cexp(-jkr)
        for index_a in range(3):
            e_r[index_a] = exp_jkr*e_inc[index_a]
            #e_r[index_a] = e_inc[index_a]


        #forall (uu=1:3) I(uu) = I(uu) + dot_product(rho_o(uu, :), e_r)*w_o
        for index_a in range(3):
            #I[index_a] += dot_product_complex_real(3, &e_r[0], &rho_o[index_a][0])*w_o
            I[index_a] += (e_r[0]*rho_o[index_a][0]+e_r[1]*rho_o[index_a][1]+e_r[2]*rho_o[index_a][2])*w_o

    return

    
@cython.boundscheck(False) 
@cython.wraparound(False)
def voltage_plane_wave_parallel(double[:, ::contiguous] nodes, int[:, ::contiguous] triangle_nodes, 
                       int[::contiguous] basis_tri_p, int[::contiguous] basis_tri_m, 
                       int[::contiguous] basis_node_p, int[::contiguous] basis_node_m,
                       double[:, ::contiguous] xi_eta_eval, double[::contiguous] weights, 
                       complex[::contiguous] e_inc, complex[::contiguous] jk_inc):
    """
    ! Calculate the voltage term, assuming a plane-wave incidence
    !
    ! Note that this assumes a free-space background

    Currently this routine is ~20x slower than the corresponding fortran routine

    integer, intent(in) :: num_nodes, num_triangles, num_basis, num_integration

    real(WP), intent(in), dimension(0:num_nodes-1, 0:2) :: nodes
    integer, intent(in), dimension(0:num_triangles-1, 0:2) :: triangle_nodes
    integer, intent(in), dimension(0:num_basis-1) :: basis_tri_p
    integer, intent(in), dimension(0:num_basis-1) :: basis_tri_m
    integer, intent(in), dimension(0:num_basis-1) :: basis_node_p
    integer, intent(in), dimension(0:num_basis-1) :: basis_node_m

    real(WP), intent(in), dimension(0:num_integration-1, 0:1) :: xi_eta_eval
    real(WP), intent(in), dimension(0:num_integration-1) :: weights

    complex(WP), intent(in), dimension(3) :: jk_inc
    complex(WP), intent(in), dimension(3) :: e_inc

    complex(WP), intent(out), dimension(0:num_basis-1) :: V
    """

    cdef int num_triangles = triangle_nodes.shape[0]
    cdef int num_basis = basis_tri_p.shape[0]
    cdef int n_o = weights.shape[0]
    cdef int which_node
    cdef np.ndarray[dtype=complex_t, ndim=1] V = np.zeros(num_basis, dtype=np.complex128) 

    cdef complex* e_inc_ptr = &e_inc[0]
    cdef complex* jk_inc_ptr = &jk_inc[0]
    cdef double* xi_eta_ptr = &xi_eta_eval[0, 0]
    cdef double* weights_ptr = &weights[0]
 
    cdef complex* V_face = <complex*>malloc(num_triangles*3*sizeof(complex))
    cdef double* nodes_p # statically allocate the storage - any point?

    cdef int p, p_p, p_m, ip_p, ip_m, m, q
    cdef int thread_id


    with nogil, parallel():
        nodes_p = <double*>malloc(3*3*sizeof(double))

        for p in prange(num_triangles, schedule='dynamic'): # p is the index of the observer face:
            for m in range(3): # m is the index of the observer face's nodes
                for q in range(3): # index of the cartesian component
                    nodes_p[m*3+q] = nodes[triangle_nodes[p, m], q]
                
            source_integral_plane_wave_pointer(n_o, xi_eta_ptr, weights_ptr, nodes_p, jk_inc_ptr, e_inc_ptr, V_face+p*3)

        # now build up the source vector in terms of the basis vectors
        for m in prange(num_basis): # m is the index of the observer edge
            p_p = basis_tri_p[m]
            p_m = basis_tri_m[m] # observer triangles
    
            ip_p = basis_node_p[m]
            ip_m = basis_node_m[m] # observer unshared nodes
    
            #V[m] = (V_face[ip_p, p_p]-V_face[ip_m, p_m])
            V[m] = (V_face[p_p*3+ip_p]-V_face[p_m*3+ip_m])

        free(nodes_p)    

    free(V_face)

    return V