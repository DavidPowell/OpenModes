
module constants
    implicit none
	
    integer, parameter :: WP = 8 ! working precision, can be 4 or 8
    ! when adjusting working precision, need to change line in file .f2py_f2cmap
    integer, parameter :: DP = 8
    integer, parameter :: SP = 4	
end module

module core_for

    use constants
    use iso_c_binding
    implicit none

    real(DP), parameter :: c = 299792458.0_DP
    !real(DP), parameter :: epsilon_0 = 8.8541878176203892e-12_DP
    !real(DP), parameter :: mu_0 =1.2566370614359173e-06_DP
    !real(DP), parameter :: pi = 3.1415926535897931_DP

    interface
        pure function e_source(r)
            use constants
            implicit none
            real(WP), dimension(3), intent(in) :: r
            complex(WP), dimension(3) :: e_source
        end function

        subroutine arcioni_singular(nodes, I_A, I_phi)
            ! Calculate singular MOM integrals as per Arcioni, IEEE MTT 45 p436
            use constants
            implicit none
        
            real(WP), dimension(3, 3), intent(in) :: nodes
            real(WP), intent(out) :: I_phi
            real(WP), dimension(3, 3), intent(out) :: I_A  
        end subroutine

        pure function cross_product(a, b)
            use constants
            implicit none
            real(WP), intent(in), dimension(:) :: a, b
            real(WP), dimension(3) :: cross_product
        end function

        pure function mag(vector)
            use constants
            implicit none
        
            real(WP), intent(in), dimension(:) :: vector
            real(WP) :: mag
        end function

        pure function scr_index(row, col, indices, indptr)
            ! Convert compressed sparse row notation into an index within an array
            ! row, col - row and column into the sparse array
            ! indices, indptr - arrays of indices and index pointers
            ! NB: everything is assumed ZERO BASED!!
            ! Indices are not assumed to be sorted within the column
        
            implicit none
            integer, intent(in) :: row, col
            integer, dimension(0:), intent(in) :: indices, indptr
            
            integer :: scr_index
        end function


        subroutine source_integral_plane_wave(n_o, xi_eta_o, weights_o, nodes_o, jk_inc, e_inc, I)
            ! Inner product of source field with testing function to give source "voltage"
            !
            ! xi_eta_s/o - list of coordinate pairs in source/observer triangle
            ! weights_s/o - the integration weights of the source and observer
            ! nodes_s/o - the nodes of the source and observer triangles
            ! k_0 - free space wavenumber
            ! nodes - the position of the triangle nodes

            use constants
            implicit none

            integer, intent(in) :: n_o
            ! f2py intent(hide) :: n_o
            real(WP), dimension(3, 3), intent(in) :: nodes_o

            real(WP), intent(in), dimension(0:n_o-1, 2) :: xi_eta_o
            real(WP), intent(in), dimension(0:n_o-1) :: weights_o
            complex(WP), intent(in), dimension(3) :: jk_inc
            complex(WP), intent(in), dimension(3) :: e_inc

            complex(WP), intent(out), dimension(:) :: I

        end subroutine

    end interface

contains



end module core_for


!!use iso_c_binding
!
!subroutine test_wrapped(a) bind(c, name='test_wrapped_')
!    use constants
!    use iso_c_binding
!      
!    implicit none
!	
!    complex(c_float_complex), value, intent(in) :: a
!	
!    print *, "input value is ", a
!
!end subroutine
!

pure function cross_product(a, b)
    use constants
    implicit none
    real(WP), intent(in), dimension(:) :: a, b
    real(WP), dimension(3) :: cross_product

    cross_product(1) = a(2)*b(3) - a(3)*b(2)
    cross_product(2) = a(3)*b(1) - a(1)*b(3)
    cross_product(3) = a(1)*b(2) - a(2)*b(1)

end function

pure function mag(vector)
    use constants
    implicit none

    real(WP), intent(in), dimension(:) :: vector
    real(WP) :: mag

    mag = sqrt(dot_product(vector, vector))
end function

pure function scr_index(row, col, indices, indptr)
    ! Convert compressed sparse row notation into an index within an array
    ! row, col - row and column into the sparse array
    ! indices, indptr - arrays of indices and index pointers
    ! NB: everything is assumed ZERO BASED!!
    ! Indices are not assumed to be sorted within the column

    implicit none
    integer, intent(in) :: row, col
    integer, dimension(0:), intent(in) :: indices, indptr
    
    integer :: scr_index

    integer :: n
    do n = indptr(row),indptr(row+1)
        if (indices(n)==col) then
            scr_index = n
            return
        end if
    end do
    scr_index = -1
    return ! value not found, so return an error code

end function

subroutine face_integrals_complex(n_s, xi_eta_s, weights_s, nodes_s_in, n_o, xi_eta_o, &
        weights_o, nodes_o_in, jk_0, I_A, I_phi)
    ! Fully integrated over source and observer, vector kernel of the MOM for RWG basis functions
    ! NB: includes the 1/4A**2 prefactor
    !
    ! xi_eta_s/o - list of coordinate pairs in source/observer triangle
    ! weights_s/o - the integration weights of the source and observer
    ! nodes_s/o - the nodes of the source and observer triangles
    ! jk_0 - *complex* free space wavenumber, j*k_0
    ! nodes - the position of the triangle nodes

    use core_for
    implicit none

    integer, intent(in) :: n_s, n_o
    ! f2py intent(hide) :: n_s, n_o
    real(WP), dimension(3, 3), intent(in) :: nodes_s_in, nodes_o_in
    complex(WP), intent(in) :: jk_0

    real(WP), intent(in), dimension(0:n_s-1, 2) :: xi_eta_s
    real(WP), intent(in), dimension(0:n_s-1) :: weights_s

    real(WP), intent(in), dimension(0:n_o-1, 2) :: xi_eta_o
    real(WP), intent(in), dimension(0:n_o-1) :: weights_o

    complex(WP), intent(out), dimension(3, 3) :: I_A
    complex(WP), intent(out) :: I_phi

    real(WP) :: xi_s, eta_s, zeta_s, xi_o, eta_o, zeta_o, R, w_s, w_o
    real(WP), dimension(3) :: r_s, r_o
    real(WP), dimension(3, 3) :: rho_s, rho_o
    complex(WP) :: g
    integer :: count_s, count_o, uu, vv !, ww
    real(WP), dimension(3, 3) :: nodes_s, nodes_o

    real(WP), dimension(3, 0:n_s-1) :: r_s_table
    real(WP), dimension(3, 3, 0:n_s-1) :: rho_s_table

    ! explictly copying the output arrays gives some small speedup,
    ! possibly by avoiding access to the shared target array
    complex(WP) :: I_phi_int
    complex(WP), dimension(3, 3) :: I_A_int

    
    ! transpose for speed
    nodes_s = transpose(nodes_s_in)
    nodes_o = transpose(nodes_o_in)

    I_A_int = 0.0
    I_phi_int = 0.0

    ! The loop over the source is repeated many times. Therefore pre-calculate the source
    ! quantities to optimise speed (gives minor benefit)

    do count_s = 0,n_s-1

        !w_s = weights_s(count_s)
        xi_s = xi_eta_s(count_s, 1)
        eta_s = xi_eta_s(count_s, 2)

        zeta_s = 1.0 - eta_s - xi_s
        r_s = xi_s*nodes_s(:, 1) + eta_s*nodes_s(:, 2) + zeta_s*nodes_s(:, 3)
        r_s_table(:, count_s) = r_s

        forall (uu=1:3) rho_s_table(:, uu, count_s) = r_s - nodes_s(:, uu)

    end do

    do count_o = 0,n_o-1

        w_o = weights_o(count_o)

        ! Barycentric coordinates of the observer
        xi_o = xi_eta_o(count_o, 1)
        eta_o = xi_eta_o(count_o, 2)
        zeta_o = 1.0 - eta_o - xi_o

        ! Cartesian coordinates of the observer
        r_o = xi_o*nodes_o(:, 1) + eta_o*nodes_o(:, 2) + zeta_o*nodes_o(:, 3)

        ! Vector rho within the observer triangle
        forall (uu=1:3) rho_o(:, uu) = r_o - nodes_o(:, uu)

        do count_s = 0,n_s-1
    
            w_s = weights_s(count_s)

            r_s = r_s_table(:, count_s)
            rho_s = rho_s_table(:, :, count_s)
              
            R = sqrt(sum((r_s - r_o)**2))
            g =  exp(-jk_0*R)/R
     
            I_phi_int = I_phi_int + g*w_s*w_o

            I_A_int = I_A_int + g*w_s*w_o*matmul(transpose(rho_o), rho_s)

        end do
    end do

    I_A = I_A_int
    I_phi = I_phi_int

end subroutine face_integrals_complex

subroutine source_integral_plane_wave(n_o, xi_eta_o, weights_o, nodes_o, &
                                      jk_inc, e_inc, I)
    ! Inner product of source field with testing function to give source "voltage"
    !
    ! xi_eta_s/o - list of coordinate pairs in source/observer triangle
    ! weights_s/o - the integration weights of the source and observer
    ! nodes_s/o - the nodes of the source and observer triangles
    ! k_0 - free space wavenumber
    ! nodes - the position of the triangle nodes

    use constants
    !use core_for
    implicit none

    integer, intent(in) :: n_o
    ! f2py intent(hide) :: n_o
    real(WP), dimension(3, 3), intent(in) :: nodes_o

    real(WP), intent(in), dimension(0:n_o-1, 2) :: xi_eta_o
    real(WP), intent(in), dimension(0:n_o-1) :: weights_o
    complex(WP), intent(in), dimension(3) :: jk_inc
    complex(WP), intent(in), dimension(3) :: e_inc

    complex(WP), intent(out), dimension(:) :: I

    real(WP) :: xi_o, eta_o, zeta_o, w_o
    real(WP), dimension(3) :: r_o
    real(WP), dimension(3, 3) :: rho_o
    complex(WP), dimension(3) :: e_r
    integer :: count_o, uu

    I = 0.0

    do count_o = 0,n_o-1

        w_o = weights_o(count_o)

        ! Barycentric coordinates of the observer
        xi_o = xi_eta_o(count_o, 1)
        eta_o = xi_eta_o(count_o, 2)
        zeta_o = 1.0 - eta_o - xi_o

        ! Cartesian coordinates of the observer
        r_o = xi_o*nodes_o(1, :) + eta_o*nodes_o(2, :) + zeta_o*nodes_o(3, :)

        ! Vector rho within the observer triangle
        forall (uu=1:3) rho_o(uu, :) = r_o - nodes_o(uu, :)

        ! calculate the incident electric field
        e_r = exp(-dot_product(jk_inc, r_o))*e_inc

        forall (uu=1:3) I(uu) = I(uu) + dot_product(rho_o(uu, :), e_r)*w_o
    end do

end subroutine source_integral_plane_wave


subroutine arcioni_singular(nodes, I_A, I_phi)
    ! Calculate singular 1/R term of the MOM integrals as per
    ! P. Arcioni, M. Bressan, and L. Perregrini, 
    ! IEEE Trans. Microw. Theory Tech. 45, 436 (1997).
    !
    ! Only works for the self impedance term on the same triangle
    !
    ! nodes - the three nodes of the triangle
    ! I_A - (3x3) the integrated vector potential terms
    ! I_phi - the integrated scalar potential
    use constants
    use core_for, only : cross_product, mag
    implicit none

    real(WP), dimension(3, 3), intent(in) :: nodes
    real(WP), intent(out) :: I_phi
    real(WP), dimension(3, 3), intent(out) :: I_A    
 
    real(WP) :: aa, bb, cc, loga, logb, logc, p, Area
    real(WP), dimension(3, 3) :: v
    real(WP), dimension(3) :: l, logl
    integer :: m, n

    v(:, 1) = nodes(3, :)-nodes(2, :)
    v(:, 2) = nodes(3, :)-nodes(1, :)
    v(:, 3) = nodes(2, :)-nodes(1, :)

    Area = 0.5*mag(cross_product(v(:, 1), v(:, 2)))

    forall (n=1:3) l(n) = mag(v(:, n)) ! lengths of the sides
    p = 0.5*sum(l) ! half perimeter
    logl = log(1.0-l/p)/l ! the log terms which appear in expressions below

    I_phi = -4.0/3.0*sum(logl)

    do m=1,3
        do n=1,m
            if (m==n) then
                ! index the appropriate side lengths
                aa = l(m)
                loga = logl(m)
                bb =      l(mod(m, 3)+1)
                logb = logl(mod(m, 3)+1)
                cc =      l(mod(m+1, 3)+1)
                logc = logl(mod(m+1, 3)+1)

                I_A(m,m) = 1.0/30.0*( &
                (10 + 3*(cc**2-aa**2)/bb**2 - 3*(aa**2-bb**2)/cc**2)*aa - &
                ( 5 - 3*(aa**2-bb**2)/cc**2 - 2*(bb**2-cc**2)/aa**2)*bb - &
                ( 5 + 3*(cc**2-aa**2)/bb**2 + 2*(bb**2-cc**2)/aa**2)*cc + &
                (aa**2 - 3*bb**2 - 3*cc**2 - 8*Area**2/aa**2)*2*loga + &
                (aa**2 - 2*bb**2 - 4*cc**2 + 6*Area**2/bb**2)*4*logb + &
                (aa**2 - 4*bb**2 - 2*cc**2 + 6*Area**2/cc**2)*4*logc )

            else
                aa = l(6-m-n) ! indexing of a is a little bit black magic!
                loga = logl(6-m-n)
                bb = l(m)
                logb = logl(m)
                cc = l(n)
                logc = logl(n)

                I_A(m,n) = 1.0/60.0*(&
                (-10 + (cc**2-aa**2)/bb**2 -   (aa**2-bb**2)/cc**2)*aa + &
                (  5 + (aa**2-bb**2)/cc**2 - 6*(bb**2-cc**2)/aa**2)*bb + &
                (  5 - (cc**2-aa**2)/bb**2 + 6*(bb**2-cc**2)/aa**2)*cc + &
                (2*aa**2 -   bb**2 -  cc**2 + 4*Area**2/aa**2)*12*loga + &
                (9*aa**2 - 3*bb**2 -  cc**2 + 4*Area**2/bb**2)* 2*logb + &
                (9*aa**2 -   bb**2 -3*cc**2 + 4*Area**2/cc**2)* 2*logc )
            end if
        end do
    end do

    ! the identical components below diagonal
    I_A(1,2) = I_A(2,1)
    I_A(1,3) = I_A(3,1)
    I_A(2,3) = I_A(3,2)

    I_phi = I_phi*0.25
    I_A = I_A*0.25

end subroutine

pure subroutine face_integrals_smooth_complex(n_s, n_s2, xi_eta_s, weights_s, &
                nodes_s, n_o, xi_eta_o, weights_o, nodes_o, jk_0, I_A, I_phi)
    ! Integrate the smooth part of the kernel, currently excludes only the 1/R part
    ! Fully integrated over source and observer, vector kernel of the MOM for RWG basis functions
    ! with the singular term(s) 1/R and R subtracted
    ! NB: includes the 1/4A**2 prefactor
    !
    ! xi_eta_s/o - list of coordinate pairs in source/observer triangle
    ! weights_s/o - the integration weights of the source and observer
    ! nodes_s/o - the nodes of the source and observer triangles
    ! k_0 - free space wavenumber
    ! nodes - the position of the triangle nodes
    !
    ! Note that weights_s has an additional dimension, which will normally be 
    ! of length one, allowance is made for the use of the singularity 
    ! technique, in which case the additional dimensions will be equal to the 
    ! number of observer integration points. This feature is currently not
    ! implemented 

    use constants
    implicit none

    integer, intent(in) :: n_s, n_s2, n_o
    ! f2py intent(hide) :: n_s, n_s2, n_o
    real(WP), dimension(3, 3), intent(in) :: nodes_s, nodes_o
    complex(WP), intent(in) :: jk_0

    real(WP), intent(in), dimension(0:n_s2-1, 0:n_s-1, 2) :: xi_eta_s
    real(WP), intent(in), dimension(0:n_s2-1, 0:n_s-1) :: weights_s

    real(WP), intent(in), dimension(0:n_o-1, 2) :: xi_eta_o
    real(WP), intent(in), dimension(0:n_o-1) :: weights_o

    complex(WP), intent(out), dimension(3, 3) :: I_A
    complex(WP), intent(out) :: I_phi

    real(WP) :: xi_s, eta_s, zeta_s, xi_o, eta_o, zeta_o, R, w_s, w_o
    real(WP), dimension(3) :: r_s, r_o
    real(WP), dimension(3, 3) :: rho_s, rho_o
    complex(WP) :: g
    integer :: count_s, count_o, uu, vv
    
    I_A = 0.0
    I_phi = 0.0

    do count_o = 0,n_o-1

        w_o = weights_o(count_o)

        ! Barycentric coordinates of the observer
        xi_o = xi_eta_o(count_o, 1)
        eta_o = xi_eta_o(count_o, 2)
        zeta_o = 1.0 - eta_o - xi_o

        ! Cartesian coordinates of the observer
        r_o = xi_o*nodes_o(1, :) + eta_o*nodes_o(2, :) + zeta_o*nodes_o(3, :)

        ! Vector rho within the observer triangle
        forall (uu=1:3) rho_o(uu, :) = r_o - nodes_o(uu, :)    

        do count_s = 0,n_s-1
    
            ! for singular integrations, the source quadrature depends on the observation point
            w_s = weights_s(0, count_s)
            xi_s = xi_eta_s(0, count_s, 1)
            eta_s = xi_eta_s(0, count_s, 2)

            zeta_s = 1.0 - eta_s - xi_s
            r_s = xi_s*nodes_s(1, :) + eta_s*nodes_s(2, :) + zeta_s*nodes_s(3, :)
    
    
            forall (uu=1:3) rho_s(uu, :) = r_s - nodes_s(uu, :)
              
            R = sqrt(sum((r_s - r_o)**2))

            ! give the explicit limit for R=0 
            ! (could use a Taylor expansion for small k_0*R?)
            if (abs(jk_0*R) < 1e-8) then
                g = -jk_0
            else
                g = (exp(-jk_0*R) - 1.0)/R
            end if

            I_phi = I_phi + g*w_s*w_o

            forall (uu=1:3, vv=1:3) I_A(uu, vv) = I_A(uu, vv) + g*dot_product(rho_o(uu, :), rho_s(vv, :))*w_s*w_o
        end do
    end do

end subroutine face_integrals_smooth_complex


subroutine set_threads(n)
    ! Set the number of openmp threads
    ! -1 signifies to have as many threads as CPUs
    use omp_lib
    !use mkl_service
    implicit none

    integer, intent(in) :: n

    if (n == -1) then
        call omp_set_num_threads(omp_get_num_procs())
    else
        call omp_set_num_threads(n)
    end if
end subroutine
    
subroutine get_threads(n)
    ! Get the number of openmp threads
    use omp_lib
    implicit none

    integer, intent(out) :: n

    n = omp_get_max_threads()
end subroutine

subroutine Z_EFIE_faces(num_nodes, num_triangles, num_integration, num_singular, nodes, triangle_nodes, &
                                s, xi_eta_eval, weights, phi_precalc, A_precalc, indices_precalc, indptr_precalc, &
                                A_face, phi_face)
    ! Calculate the face to face interaction terms used to build the impedance matrix
    !
    ! As per Rao, Wilton, Glisson, IEEE Trans AP-30, 409 (1982)
    ! Uses impedance extraction techqnique of Hanninen, precalculated
    !
    ! nodes - position of all the triangle nodes
    ! basis_tri_p/m - the positive and negative triangles for each basis function
    ! basis_node_p/m - the free nodes for each basis function
    ! omega - evaulation frequency in rad/s
    ! s - complex frequency
    ! xi_eta_eval, weights - quadrature rule over the triangle (weights normalised to 0.5)
    ! A_precalc, phi_precalc - precalculated 1/R singular terms

    use core_for
    implicit none

    integer, intent(in) :: num_nodes, num_triangles, num_integration, num_singular
    ! f2py intent(hide) :: num_nodes, num_triangles, num_integration, num_singular

    real(WP), intent(in), dimension(0:num_nodes-1, 0:2) :: nodes
    integer, intent(in), dimension(0:num_triangles-1, 0:2) :: triangle_nodes

    complex(WP), intent(in) :: s

    real(WP), intent(in), dimension(0:num_integration-1, 0:1) :: xi_eta_eval
    real(WP), intent(in), dimension(0:num_integration-1) :: weights

    real(WP), intent(in), dimension(0:num_singular-1) :: phi_precalc
    real(WP), intent(in), dimension(0:num_singular-1, 3, 3) :: A_precalc
    integer, intent(in), dimension(0:num_singular-1) :: indices_precalc
    integer, intent(in), dimension(0:num_triangles) :: indptr_precalc

    complex(WP), intent(out), dimension(0:num_triangles-1, 0:num_triangles-1, 0:2, 0:2) :: A_face
    complex(WP), intent(out), dimension(0:num_triangles-1, 0:num_triangles-1) :: phi_face
    
    
    complex(WP) :: jk_0 
    
    real(WP), dimension(0:2, 0:2) :: nodes_p, nodes_q
    complex(WP) :: A_part, phi_part
    complex(WP), dimension(3, 3) :: I_A
    complex(WP) :: I_phi

    integer :: p, q, q_p, q_m, p_p, p_m, ip_p, ip_m, iq_p, iq_m, m, n, index_singular

    jk_0 = s/c

    ! calculate all the integrations for each face pair
    !$OMP PARALLEL DO SCHEDULE(DYNAMIC) DEFAULT(SHARED) &
    !$OMP PRIVATE (p, q, nodes_p, nodes_q, I_A, I_phi)
    do p = 0,num_triangles-1 ! p is the index of the observer face:
        nodes_p = nodes(triangle_nodes(p, :), :)
        do q = 0,p ! q is the index of the source face, need for elements below diagonal

            nodes_q = nodes(triangle_nodes(q, :), :)
            if (any(triangle_nodes(p, :) == triangle_nodes(q, :))) then
                ! triangles have one or more common nodes, perform singularity extraction
                call face_integrals_smooth_complex(num_integration, 1, xi_eta_eval, weights, nodes_q, &
                                    num_integration, xi_eta_eval, weights, nodes_p, jk_0, I_A, I_phi)
        
                ! the singular 1/R components are pre-calculated
                index_singular = scr_index(p, q, indices_precalc, indptr_precalc)

                I_A = I_A + A_precalc(index_singular, :, :)
                I_phi = I_phi + phi_precalc(index_singular)
        
            else
                ! just perform regular integration
                ! As per RWG, triangle area must be cancelled in the integration
                ! for non-singular terms the weights are unity and we DON't want to scale to triangle area
                call face_integrals_complex(num_integration, xi_eta_eval, weights, nodes_q, &
                                    num_integration, xi_eta_eval, weights, nodes_p, jk_0, I_A, I_phi)
            end if

            ! by symmetry of Galerkin procedure, transposed components are identical (but transposed node indices)
            A_face(p, q, :, :) = I_A
            A_face(q, p, :, :) = transpose(I_A)
            phi_face(p, q) = I_phi
            phi_face(q, p) = I_phi

        end do
    end do
    !$OMP END PARALLEL DO

end subroutine Z_EFIE_faces

subroutine voltage_plane_wave(num_nodes, num_triangles, num_basis, num_integration, nodes, triangle_nodes, &
                                basis_tri_p, basis_tri_m, basis_node_p, basis_node_m, &
                                xi_eta_eval, weights, e_inc, jk_inc, V)
    ! Calculate the voltage term, assuming a plane-wave incidence
    !
    ! Note that this assumes a free-space background


    use core_for
    implicit none

    integer, intent(in) :: num_nodes, num_triangles, num_basis, num_integration
    ! f2py intent(hide) :: num_nodes, num_triangles, num_basis, num_integration

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

    real(WP), dimension(0:2, 0:2) :: nodes_p, nodes_q
    complex(WP), dimension(0:2, 0:num_triangles-1) :: V_face

    integer :: p, p_p, p_m, ip_p, ip_m, m

    ! calculate all the integrations for each face pair
    !$OMP PARALLEL DO SCHEDULE(DYNAMIC) DEFAULT(SHARED) &
    !$OMP PRIVATE (p, nodes_p)
    do p = 0,num_triangles-1 ! p is the index of the observer face:
        nodes_p = nodes(triangle_nodes(p, :), :)
        ! perform testing of the incident field
        call source_integral_plane_wave(num_integration, xi_eta_eval, weights, nodes_p, jk_inc, e_inc, V_face(:, p))
    end do
    !$OMP END PARALLEL DO

    ! now build up the source vector in terms of the basis vectors
    do m=0,num_basis-1 ! m is the index of the observer edge
        p_p = basis_tri_p(m)
        p_m = basis_tri_m(m) ! observer triangles

        ip_p = basis_node_p(m)
        ip_m = basis_node_m(m) ! observer unshared nodes

        V(m) = (V_face(ip_p, p_p)-V_face(ip_m, p_m))

    end do

end subroutine

subroutine face_to_rwg(num_triangles, num_basis, basis_tri_p, basis_tri_m, basis_node_p, basis_node_m, &
                        vector_face, scalar_face, vector_rwg, scalar_rwg)
    ! take quantities which are defined as interaction between faces and convert them to rwg basis

    use core_for
    
    integer, intent(in) :: num_triangles, num_basis
    ! f2py intent(hide) :: num_triangles, num_basis
    
    integer, intent(in), dimension(0:num_basis-1) :: basis_tri_p
    integer, intent(in), dimension(0:num_basis-1) :: basis_tri_m
    integer, intent(in), dimension(0:num_basis-1) :: basis_node_p
    integer, intent(in), dimension(0:num_basis-1) :: basis_node_m    

    complex(WP), intent(in), dimension(0:num_triangles-1, 0:num_triangles-1, 0:2, 0:2) :: vector_face
    complex(WP), intent(in), dimension(0:num_triangles-1, 0:num_triangles-1) :: scalar_face
    
    complex(WP), intent(out), dimension(0:num_basis-1, 0:num_basis-1) :: vector_rwg, scalar_rwg

 
    integer m, n, p_p, p_m, q_p, q_m, ip_p, ip_m, iq_p, iq_m

    do m=0,num_basis-1 ! m is the index of the observer edge
        p_p = basis_tri_p(m)
        p_m = basis_tri_m(m) ! observer triangles

        ip_p = basis_node_p(m)
        ip_m = basis_node_m(m) ! observer unshared nodes
        
        do n = 0,num_basis-1 ! n is the index of the source
            q_p = basis_tri_p(n)
            q_m = basis_tri_m(n) ! source triangles
            
            iq_p = basis_node_p(n)
            iq_m = basis_node_m(n) ! source unshared nodes

            vector_rwg(m, n) = ( &
                  vector_face(p_p, q_p, ip_p, iq_p) - vector_face(p_p, q_m, ip_p, iq_m) &
                - vector_face(p_m, q_p, ip_m, iq_p) + vector_face(p_m, q_m, ip_m, iq_m))
                
            scalar_rwg(m, n) = ( &
                - scalar_face(p_m, q_p) + scalar_face(p_m, q_m) &
                + scalar_face(p_p, q_p) - scalar_face(p_p, q_m))

        end do
    end do

end subroutine face_to_rwg


subroutine face_integrals_hanninen(nodes_s, n_o, xi_eta_o, weights_o, &
                                   nodes_o, I_A, I_phi)
    ! Fully integrated over source and observer the singular part of the MOM 
    ! for RWG basis functions
    ! NB: includes the 1/4A**2 prefactor
    ! Use method from Hanninen PIER 63 243
    !
    ! xi_eta_o - list of coordinate pairs in source/observer triangle
    ! weights_o - the integration weights of the source and observer
    ! but for singular integrals will be equal to the number of observer integration points
    ! nodes_s/o - the nodes of the source and observer triangles
    ! nodes - the position of the triangle nodes
    !
    ! Need to calculate I_S_m3, and h, giving a different formula for I_S_m1
    use core_for
    implicit none

    integer, intent(in) :: n_o
    ! f2py intent(hide) :: n_o
    real(WP), dimension(3, 3), intent(in) :: nodes_s, nodes_o
    real(WP), intent(in), dimension(0:n_o-1, 2) :: xi_eta_o
    real(WP), intent(in), dimension(0:n_o-1) :: weights_o

    real(WP), intent(out), dimension(3, 3) :: I_A
    real(WP), intent(out) :: I_phi

    real(WP) :: xi_o, eta_o, zeta_o, w_o
    real(WP), dimension(3) :: r_o, rho_o
    real(WP), dimension(3, 3) :: m_hat, a
    integer :: count_s, count_o, uu, vv
    real(WP), dimension(3) :: I_L_m1, I_L_1 !, I_L_3
    real(WP) :: I_S_m3_h, I_S_m1, I_S_1, x, y

    real(WP), dimension(3) :: p1, p2, s_hat, n_hat, t
    real(WP) :: R_m, R_p, s_m, s_p, h, area_s_2
    
    I_A = 0.0
    I_phi = 0.0

    n_hat = cross_product(nodes_S(1, :) - nodes_s(2, :), nodes_s(1, :)-nodes_s(3,:))
    area_s_2 = mag(n_hat)
    n_hat = n_hat/area_s_2
    h = dot_product(r_o-p1, n_hat) ! currently assumed zero in some formulas

    do count_o = 0,n_o-1

        w_o = weights_o(count_o)

        ! Barycentric coordinates of the observer within the observer's triangle
        xi_o = xi_eta_o(count_o, 1)
        eta_o = xi_eta_o(count_o, 2)
        zeta_o = 1.0 - eta_o - xi_o

        ! Cartesian coordinates of the observer
        r_o = xi_o*nodes_o(1, :) + eta_o*nodes_o(2, :) + zeta_o*nodes_o(3, :)

        ! Confusing Notation - this is projection of observer onto the plane of the source triangle
        rho_o = r_o-n_hat*dot_product(r_o, n_hat)

        ! Apply recursive formulae of Hanninen

        ! iterate over all source edges (which are numbered by their opposite node)
        do count_s = 1,3

            ! select the nodes on the edge opposite the current node
            p1 = nodes_s(mod(count_s, 3)+1, :)
            p2 = nodes_s(mod(count_s+1, 3)+1, :)

            R_m = mag(r_o-p1)
            R_p = mag(r_o-p2)

            s_hat = (p2 - p1)/mag(p2 - p1)
            s_m = dot_product(p1 - r_o, s_hat)
            s_p = dot_product(p2 - r_o, s_hat)

            !t(count_s) = dot_product(r_o-p1, 1.0-n_hat-s_hat)
            m_hat(:, count_s) = cross_product(s_hat, n_hat)
            t(count_s) = dot_product(r_o-p1, m_hat(:, count_s)) ! sign of t ???

            if (abs(R_m + s_m) > abs(R_p - s_p)) then
                I_L_m1(count_s) = log((R_p + s_p)/(R_m + s_m))
            else
                I_L_m1(count_s) = log((R_m - s_m)/(R_p - s_p))
            end if

            ! apply recursion formula for the line
            ! using the expression for R_0
            I_L_1(count_s) = 0.5*((R_p**2 - s_p**2)*I_L_m1(count_s) + s_p*R_p - s_m*R_m)

            !I_L_3(count_s) = 3.0/4.0*(R_p**2 - s_p**2)*I_L_m1(count_s) + 1/4.0*(s_p*R_p - s_m*R_m)
        end do

        ! Find the initial area integral (zero for h=0)
        !a = (nodes_s-r_o)/mag(nodes_s-r_o) ! nodes need to be listed in a specific order?????
        forall (uu=1:3) a(:, uu) = (nodes_s(uu, :)-r_o)/mag(nodes_s(uu, :)-r_o)
        x = 1 + dot_product(a(:, 1), a(:, 2)) + dot_product(a(:, 1), a(:, 3)) + dot_product(a(:, 2), a(:, 3))
        y = abs(dot_product(a(:, 1), cross_product(a(:, 2), a(:, 3))))
        I_S_m3_h = 2*atan2(y, x)

        ! currently assumes that source and observer are in the same plane, so that h=0
        I_S_m1 = -sum(t*I_L_m1) - h*I_S_m3_h
        I_S_1 = -0.5*sum(t*I_L_1) + h**2/3.0*I_S_m1

        ! Final results do not have explicit h dependance
        I_phi = I_phi + w_o*(I_S_m1) ! eq (65)

        ! check ordering of source vs observer triangle node
        ! eq (70)
        forall (uu=1:3, vv=1:3) I_A(uu, vv) = I_A(uu, vv) + w_o*( &
            !dot_product((m_hat(:, 1)*I_L_1(1)+m_hat(:, 2)*I_L_1(2)+m_hat(:, 3)*I_L_1(3)) + &
            !     (rho_o-nodes_s(vv, :))*I_S_m1,(r_o - nodes_o(uu, :))))! + &
            dot_product(matmul(I_L_1, transpose(m_hat)) + (rho_o-nodes_s(vv, :))*I_S_m1, &
                        (r_o - nodes_o(uu, :)) ) )! + &
    end do
    I_phi = I_phi/area_s_2
    I_A = I_A/area_s_2

end subroutine face_integrals_hanninen

subroutine triangle_face_to_rwg(num_triangles, num_basis, basis_tri_p, basis_tri_m, basis_node_p, basis_node_m, &
                        vector_face, scalar_face, vector_rwg, scalar_rwg)
    ! take quantities which are defined as interaction between faces and convert them to rwg basis

    use core_for
    
    integer, intent(in) :: num_triangles, num_basis
    ! f2py intent(hide) :: num_triangles, num_basis
    
    integer, intent(in), dimension(0:num_basis-1) :: basis_tri_p
    integer, intent(in), dimension(0:num_basis-1) :: basis_tri_m
    integer, intent(in), dimension(0:num_basis-1) :: basis_node_p
    integer, intent(in), dimension(0:num_basis-1) :: basis_node_m    

    complex(WP), intent(in), dimension(0:num_triangles-1, 0:num_triangles-1, 0:2, 0:2) :: vector_face
    complex(WP), intent(in), dimension(0:num_triangles-1, 0:num_triangles-1) :: scalar_face
    
    complex(WP), intent(out), dimension(0:num_basis-1, 0:num_basis-1) :: vector_rwg, scalar_rwg

 
    integer m, n, p_p, p_m, q_p, q_m, ip_p, ip_m, iq_p, iq_m

    do m=0,num_basis-1 ! m is the index of the observer edge
        p_p = basis_tri_p(m)
        p_m = basis_tri_m(m) ! observer triangles

        ip_p = basis_node_p(m)
        ip_m = basis_node_m(m) ! observer unshared nodes
        
        do n = 0,num_basis-1 ! n is the index of the source
            q_p = basis_tri_p(n)
            q_m = basis_tri_m(n) ! source triangles
            
            iq_p = basis_node_p(n)
            iq_m = basis_node_m(n) ! source unshared nodes

            vector_rwg(m, n) = ( &
                  vector_face(p_p, q_p, ip_p, iq_p) - vector_face(p_p, q_m, ip_p, iq_m) &
                - vector_face(p_m, q_p, ip_m, iq_p) + vector_face(p_m, q_m, ip_m, iq_m))
                
            scalar_rwg(m, n) = ( &
                - scalar_face(p_m, q_p) + scalar_face(p_m, q_m) &
                + scalar_face(p_p, q_p) - scalar_face(p_p, q_m))

        end do
    end do

end subroutine triangle_face_to_rwg
