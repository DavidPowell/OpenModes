! OpenModes - An eigenmode solver for open electromagnetic resonantors
! Copyright (C) 2013 David Powell
!
! This program is free software: you can redistribute it and/or modify
! it under the terms of the GNU General Public License as published by
! the Free Software Foundation, either version 3 of the License, or
! (at your option) any later version.
!
! This program is distributed in the hope that it will be useful,
! but WITHOUT ANY WARRANTY; without even the implied warranty of
! MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
! GNU General Public License for more details.
!
! You should have received a copy of the GNU General Public License
! along with this program.  If not, see <http://www.gnu.org/licenses/>.


module core_for

    use constants
    implicit none

    real(DP), parameter :: c = 299792458.0_DP
    !real(DP), parameter :: epsilon_0 = 8.8541878176203892e-12_DP
    !real(DP), parameter :: mu_0 =1.2566370614359173e-06_DP
    real(DP), parameter :: pi = 3.1415926535897931_DP

    interface

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

        pure subroutine face_integrals_smooth_complex(n_s, n_s2, xi_eta_s, weights_s, &
                        nodes_s, n_o, xi_eta_o, weights_o, nodes_o, jk_0, I_A, I_phi)
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
        end subroutine

        subroutine face_unit_integral(n, xi_eta, weights, nodes_in, normal, T_form, I_Z)
            ! Double integral of Dirac delta function over a single triangle face
            ! NB: includes only the 1/2A prefactor, may need an additional 1/2A
            !
            ! xi_eta - list of coordinate pairs in the triangle
            ! weights - the integration weights
            ! nodes - the position of the triangle nodes
            ! normal - the vector of the triangle normal
            ! T_form - if true, the tangential form is used, otherwise the n x form

            use constants        
            implicit none
        
            integer, intent(in) :: n
            real(WP), dimension(3, 3), intent(in) :: nodes_in
        
            real(WP), intent(in), dimension(0:n-1, 2) :: xi_eta
            real(WP), intent(in), dimension(0:n-1) :: weights
            logical, intent(in) :: T_form
        
            real(WP), intent(in), dimension(3) :: normal
        
            complex(WP), intent(out), dimension(3, 3) :: I_Z
        end subroutine

        subroutine face_integral_MFIE(n_s, xi_eta_s, weights_s, nodes_s_in, n_o, xi_eta_o, &
                weights_o, nodes_o_in, jk_0, normal, T_form, I_Z)
            ! Fully integrated over source and observer, vector kernel of the MOM for RWG basis functions
            ! NB: includes the 1/4A**2 prefactor
            !
            ! xi_eta_s/o - list of coordinate pairs in source/observer triangle
            ! weights_s/o - the integration weights of the source and observer
            ! nodes_s/o - the nodes of the source and observer triangles
            ! jk_0 - *complex* free space wavenumber, j*k_0
            ! nodes - the position of the triangle nodes
        
            use constants
            implicit none
        
            integer, intent(in) :: n_s, n_o
            real(WP), dimension(3, 3), intent(in) :: nodes_s_in, nodes_o_in
            complex(WP), intent(in) :: jk_0
        
            real(WP), intent(in), dimension(0:n_s-1, 2) :: xi_eta_s
            real(WP), intent(in), dimension(0:n_s-1) :: weights_s
        
            real(WP), intent(in), dimension(0:n_o-1, 2) :: xi_eta_o
            real(WP), intent(in), dimension(0:n_o-1) :: weights_o
        
            real(WP), intent(in), dimension(3) :: normal
            logical, intent(in) :: T_form
        
            complex(WP), intent(out), dimension(3, 3) :: I_z
        end subroutine



    end interface

contains

end module core_for


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
    use vectors!, only : cross_product, mag
    !use core_for
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


subroutine Z_EFIE_faces_mutual(num_nodes_o, num_triangles_o, num_nodes_s, num_triangles_s, &
                               num_integration, nodes_o, triangle_nodes_o, nodes_s, triangle_nodes_s, &
                                s, xi_eta_eval, weights, &
                                A_face, phi_face)
    ! Calculate the face to face interaction terms used to build the impedance matrix
    ! For mutual coupling terms between different parts
    !
    ! As per Rao, Wilton, Glisson, IEEE Trans AP-30, 409 (1982)
    ! Uses impedance extraction techqnique of Hanninen, precalculated
    !
    ! nodes - position of all the triangle nodes
    ! omega - evaulation frequency in rad/s
    ! s - complex frequency
    ! xi_eta_eval, weights - quadrature rule over the triangle (weights normalised to 0.5)

    use core_for
    implicit none

    integer, intent(in) :: num_nodes_o, num_triangles_o, num_nodes_s, num_triangles_s, num_integration

    real(WP), intent(in), dimension(0:num_nodes_o-1, 0:2) :: nodes_o
    integer, intent(in), dimension(0:num_triangles_o-1, 0:2) :: triangle_nodes_o
    real(WP), intent(in), dimension(0:num_nodes_s-1, 0:2) :: nodes_s
    integer, intent(in), dimension(0:num_triangles_s-1, 0:2) :: triangle_nodes_s

    complex(WP), intent(in) :: s

    real(WP), intent(in), dimension(0:num_integration-1, 0:1) :: xi_eta_eval
    real(WP), intent(in), dimension(0:num_integration-1) :: weights

    complex(WP), intent(out), dimension(0:num_triangles_o-1, 0:2, 0:num_triangles_s-1, 0:2) :: A_face
    complex(WP), intent(out), dimension(0:num_triangles_o-1, 0:num_triangles_s-1) :: phi_face
    
    
    complex(WP) :: jk_0 
    
    real(WP), dimension(0:2, 0:2) :: nodes_p, nodes_q
    complex(WP), dimension(3, 3) :: I_A
    complex(WP) :: I_phi

    integer :: p, q

    jk_0 = s/c

    ! calculate all the integrations for each face pair
    !$OMP PARALLEL DO SCHEDULE(DYNAMIC) DEFAULT(SHARED) &
    !$OMP PRIVATE (p, q, nodes_p, nodes_q, I_A, I_phi)
    do p = 0,num_triangles_o-1 ! p is the index of the observer face:
        nodes_p = nodes_o(triangle_nodes_o(p, :), :)
        do q = 0,num_triangles_s-1 ! q is the index of the source face

            nodes_q = nodes_s(triangle_nodes_s(q, :), :)
            ! just perform regular integration
            ! As per RWG, triangle area must be cancelled in the integration
            ! for non-singular terms the weights are unity and we DON't want to scale to triangle area
            call EFIE_face_integrals(num_integration, xi_eta_eval, weights, nodes_q, &
                                num_integration, xi_eta_eval, weights, nodes_p, jk_0, .FALSE., I_A, I_phi)
            ! by symmetry of Galerkin procedure, transposed components are identical (but transposed node indices)
            A_face(p, :, q, :) = I_A
            phi_face(p, q) = I_phi

        end do
    end do
    !$OMP END PARALLEL DO

end subroutine Z_EFIE_faces_mutual


subroutine EFIE_face_integrals(n_s, xi_eta_s, weights_s, nodes_s_in, n_o, xi_eta_o, &
        weights_o, nodes_o_in, jk_0, singular, I_A, I_phi)
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

    logical, intent(in) :: singular

    complex(WP), intent(out), dimension(3, 3) :: I_A
    complex(WP), intent(out) :: I_phi

    real(WP) :: xi_s, eta_s, zeta_s, xi_o, eta_o, zeta_o, R, w_s, w_o
    real(WP), dimension(3) :: r_s, r_o
    real(WP), dimension(3, 3) :: rho_s, rho_o
    complex(WP) :: g
    integer :: count_s, count_o, uu!, vv !, ww
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

            if (singular) then 
                ! give the explicit limit for R=0 
                ! (could use a Taylor expansion for small k_0*R?)
                if (abs(jk_0*R) < 1e-8) then
                    g = -jk_0
                else
                    g = (exp(-jk_0*R) - 1.0)/R
                end if
            else
                g = exp(-jk_0*R)/R
            end if
     
            I_phi_int = I_phi_int + g*w_s*w_o

            I_A_int = I_A_int + g*w_s*w_o*matmul(transpose(rho_o), rho_s)

        end do
    end do

    I_A = I_A_int
    I_phi = I_phi_int

end subroutine EFIE_face_integrals


subroutine Z_EFIE_faces_self(num_nodes, num_triangles, num_integration, num_singular, degree_singular, &
                                nodes, triangle_nodes, s, xi_eta_eval, weights, phi_precalc, A_precalc, &
                                indices_precalc, indptr_precalc, A_face, phi_face)
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

    integer, intent(in) :: num_nodes, num_triangles, num_integration, num_singular, degree_singular
    ! f2py intent(hide) :: num_nodes, num_triangles, num_integration, num_singular

    real(WP), intent(in), dimension(0:num_nodes-1, 0:2) :: nodes
    integer, intent(in), dimension(0:num_triangles-1, 0:2) :: triangle_nodes

    complex(WP), intent(in) :: s

    real(WP), intent(in), dimension(0:num_integration-1, 0:1) :: xi_eta_eval
    real(WP), intent(in), dimension(0:num_integration-1) :: weights

    real(WP), intent(in), dimension(0:num_singular-1, 0:degree_singular-1) :: phi_precalc
    real(WP), intent(in), dimension(0:num_singular-1, 0:degree_singular-1, 3, 3) :: A_precalc
    integer, intent(in), dimension(0:num_singular-1) :: indices_precalc
    integer, intent(in), dimension(0:num_triangles) :: indptr_precalc

    complex(WP), intent(out), dimension(0:num_triangles-1, 0:2, 0:num_triangles-1, 0:2) :: A_face
    complex(WP), intent(out), dimension(0:num_triangles-1, 0:num_triangles-1) :: phi_face
    
    
    complex(WP) :: jk_0 
    
    real(WP), dimension(0:2, 0:2) :: nodes_p, nodes_q
    complex(WP), dimension(3, 3) :: I_A
    complex(WP) :: I_phi

    integer :: p, q, index_singular

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
                call EFIE_face_integrals(num_integration, xi_eta_eval, weights, nodes_q, &
                                    num_integration, xi_eta_eval, weights, nodes_p, jk_0, .TRUE., I_A, I_phi)
        
                ! the singular 1/R components are pre-calculated
                index_singular = scr_index(p, q, indices_precalc, indptr_precalc)

                I_A = I_A + A_precalc(index_singular, 0, :, :)
                I_phi = I_phi + phi_precalc(index_singular, 0)
        
            else
                ! just perform regular integration
                ! As per RWG, triangle area must be cancelled in the integration
                ! for non-singular terms the weights are unity and we DON't want to scale to triangle area
                call EFIE_face_integrals(num_integration, xi_eta_eval, weights, nodes_q, &
                                    num_integration, xi_eta_eval, weights, nodes_p, jk_0, .FALSE., I_A, I_phi)
            end if

            ! by symmetry of Galerkin procedure, transposed components are identical (but transposed node indices)
            A_face(p, :, q, :) = I_A
            A_face(q, :, p, :) = transpose(I_A)
            phi_face(p, q) = I_phi
            phi_face(q, p) = I_phi

        end do
    end do
    !$OMP END PARALLEL DO

end subroutine Z_EFIE_faces_self


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
    use constants
    use vectors
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

    n_hat = cross_product(nodes_s(2, :) - nodes_s(1, :), nodes_s(3, :)-nodes_s(1,:))
    area_s_2 = mag(n_hat)
    n_hat = n_hat/area_s_2

    do count_o = 0,n_o-1

        w_o = weights_o(count_o)

        ! Barycentric coordinates of the observer within the observer's triangle
        xi_o = xi_eta_o(count_o, 1)
        eta_o = xi_eta_o(count_o, 2)
        zeta_o = 1.0 - eta_o - xi_o

        ! Cartesian coordinates of the observer
        r_o = xi_o*nodes_o(1, :) + eta_o*nodes_o(2, :) + zeta_o*nodes_o(3, :)

        ! Out of plane coordinate of observer
        h = abs(dot_product(r_o-nodes_s(1, :), n_hat))

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
            I_L_1(count_s) = 0.5*((t(count_s)**2 + h**2)*I_L_m1(count_s) + s_p*R_p - s_m*R_m)

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


subroutine face_integral_MFIE(n_s, xi_eta_s, weights_s, nodes_s_in, n_o, xi_eta_o, &
        weights_o, nodes_o_in, jk_0, normal, T_form, I_Z)
    ! Fully integrated over source and observer, vector kernel of the MOM for RWG basis functions
    ! NB: includes the 1/4A**2 prefactor
    !
    ! xi_eta_s/o - list of coordinate pairs in source/observer triangle
    ! weights_s/o - the integration weights of the source and observer
    ! nodes_s/o - the nodes of the source and observer triangles
    ! jk_0 - *complex* free space wavenumber, j*k_0
    ! nodes - the position of the triangle nodes

    use core_for
    use vectors
    implicit none

    integer, intent(in) :: n_s, n_o
    real(WP), dimension(3, 3), intent(in) :: nodes_s_in, nodes_o_in
    complex(WP), intent(in) :: jk_0

    real(WP), intent(in), dimension(0:n_s-1, 2) :: xi_eta_s
    real(WP), intent(in), dimension(0:n_s-1) :: weights_s

    real(WP), intent(in), dimension(0:n_o-1, 2) :: xi_eta_o
    real(WP), intent(in), dimension(0:n_o-1) :: weights_o

    real(WP), intent(in), dimension(3) :: normal
    logical, intent(in) :: T_form

    complex(WP), intent(out), dimension(3, 3) :: I_z

    real(WP) :: xi_s, eta_s, zeta_s, xi_o, eta_o, zeta_o, R, w_s, w_o
    real(WP), dimension(3) :: r_s, r_o
    real(WP), dimension(3, 3) :: rho_s, rho_o
    complex(WP) :: g
    integer :: count_s, count_o, uu, vv
    real(WP), dimension(3, 3) :: nodes_s, nodes_o

    real(WP), dimension(3, 0:n_s-1) :: r_s_table
    real(WP), dimension(3, 3, 0:n_s-1) :: rho_s_table

    ! explictly copying the output arrays gives some small speedup,
    ! possibly by avoiding access to the shared target array
    complex(WP), dimension(3, 3) :: I_Z_int

    
    ! transpose for speed
    nodes_s = transpose(nodes_s_in)
    nodes_o = transpose(nodes_o_in)

    I_Z_int = 0.0

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
            g = (1.0 + jk_0*R)*exp(-jk_0*R)/R**3

            
            if (T_form) then
                ! The tang RWG form
                forall (uu=1:3, vv=1:3) I_Z_int(uu, vv) = I_Z_int(uu, vv) + w_o*w_s*g*( &
                    dot_product(rho_o(:, uu), -cross_product(normal, cross_product(normal, cross_product(r_o - r_s, rho_s(:, vv))))))
                !forall (uu=1:3, vv=1:3) I_Z_int(uu, vv) = I_Z_int(uu, vv) + w_o*w_s*g*( &
                !    dot_product(rho_o(:, uu), cross_product(r_o - r_s, rho_s(:, vv))))


                !forall (uu=1:3, vv=1:3) I_Z_int(uu, vv) = I_Z_int(uu, vv) + w_o*w_s*g*( &
                !    dot_product(rho_o(:, uu), cross_product(r_o - r_s, rho_s(:, vv))))

                ! From Reid
                !forall (uu=1:3, vv=1:3) I_Z_int(uu, vv) = I_Z_int(uu, vv) + w_o*w_s*g*( &
                !    dot_product(r_o - r_s, cross_product(rho_o(:, uu), rho_s(:, vv))))

            else     
                ! The n x RWG form
                forall (uu=1:3, vv=1:3) I_Z_int(uu, vv) = I_Z_int(uu, vv) + w_o*w_s*g*( &
                    dot_product(rho_o(:, uu), cross_product(normal, cross_product(r_o - r_s, rho_s(:, vv)))))
            end if

        end do
    end do

    I_Z = I_Z_int

end subroutine face_integral_MFIE

subroutine face_unit_integral(n, xi_eta, weights, nodes_in, normal, n_cross, I_Z)
    ! Double integral of Dirac delta function over a single triangle face
    ! NB: includes only the 1/2A prefactor, may need an additional 1/2A
    !
    ! xi_eta - list of coordinate pairs in the triangle
    ! weights - the integration weights
    ! nodes - the position of the triangle nodes
    ! normal - the vector of the triangle normal
    ! n_cross - if true, n x RWG is used as the source function 

    use core_for
    use vectors
    implicit none

    integer, intent(in) :: n
    real(WP), dimension(3, 3), intent(in) :: nodes_in

    real(WP), intent(in), dimension(0:n-1, 2) :: xi_eta
    real(WP), intent(in), dimension(0:n-1) :: weights
    logical, intent(in) :: n_cross

    real(WP), intent(in), dimension(3) :: normal

    complex(WP), intent(out), dimension(3, 3) :: I_Z

    real(WP) :: xi, eta, zeta, w
    real(WP), dimension(3) :: r
    real(WP), dimension(3, 3) :: rho
    integer :: count_o, uu, vv
    real(WP), dimension(3, 3) :: nodes

    ! explictly copying the output arrays gives some small speedup,
    ! possibly by avoiding access to the shared target array
    complex(WP), dimension(3, 3) :: I_Z_int

    
    ! transpose for speed
    nodes = transpose(nodes_in)

    I_Z_int = 0.0

    ! The loop over the source is repeated many times. Therefore pre-calculate the source
    ! quantities to optimise speed (gives minor benefit)

    do count_o = 0,n-1

        w = weights(count_o)

        ! Barycentric coordinates of the observer
        xi = xi_eta(count_o, 1)
        eta = xi_eta(count_o, 2)
        zeta = 1.0 - eta - xi

        ! Cartesian coordinates of the observer
        r = xi*nodes(:, 1) + eta*nodes(:, 2) + zeta*nodes(:, 3)

        ! Vector rho within the observer triangle
        forall (uu=1:3) rho(:, uu) = r - nodes(:, uu)

        if (n_cross) then
            ! Use n x RWG source function
            forall (uu=1:3, vv=1:3) I_Z_int(uu, vv) = I_Z_int(uu, vv) - w*( &
                dot_product(rho(:, uu), cross_product(normal, rho(:, vv))))
        else
            ! RWG source function
            forall (uu=1:3, vv=1:3) I_Z_int(uu, vv) = I_Z_int(uu, vv) + w*( &
                dot_product(rho(:, uu), rho(:, vv)))
        end if
    end do

    I_Z = I_Z_int

end subroutine face_unit_integral

subroutine Z_MFIE_faces_self(num_nodes, num_triangles, num_integration, nodes, triangle_nodes, &
                                triangle_areas, s, xi_eta, weights, normals, T_form, Z_face)
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
    ! xi_eta, weights - quadrature rule over the triangle (weights normalised to 0.5)
    ! A_precalc, phi_precalc - precalculated 1/R singular terms

    use core_for
    implicit none

    integer, intent(in) :: num_nodes, num_triangles, num_integration
    ! f2py intent(hide) :: num_nodes, num_triangles, num_integration

    real(WP), intent(in), dimension(0:num_nodes-1, 0:2) :: nodes
    integer, intent(in), dimension(0:num_triangles-1, 0:2) :: triangle_nodes
    real(WP), intent(in), dimension(0:num_triangles-1) :: triangle_areas

    complex(WP), intent(in) :: s

    real(WP), intent(in), dimension(0:num_integration-1, 0:1) :: xi_eta
    real(WP), intent(in), dimension(0:num_integration-1) :: weights
    real(WP), intent(in), dimension(0:num_triangles-1, 0:2) :: normals
    logical, intent(in) :: T_form

    complex(WP), intent(out), dimension(0:num_triangles-1, 0:2, 0:num_triangles-1, 0:2) :: Z_face
    
    complex(WP) :: jk_0 
    
    real(WP), dimension(0:2, 0:2) :: nodes_p, nodes_q
    complex(WP), dimension(3, 3) :: I_Z

    integer :: p, q

    jk_0 = s/c

    ! calculate all the integrations for each face pair
    do p = 0,num_triangles-1 ! p is the index of the observer face:
        nodes_p = nodes(triangle_nodes(p, :), :)
        do q = 0,num_triangles-1 ! q is the index of the source face

            nodes_q = nodes(triangle_nodes(q, :), :)
            if (p == q) then
                ! diagonal self terms
                call face_unit_integral(num_integration, xi_eta, weights, nodes_p, normals(p, :), T_form, I_Z)
                I_Z = I_Z/4.0/triangle_areas(p)
            else
                ! just perform regular integration
                ! As per RWG, triangle area must be cancelled in the integration
                ! for non-singular terms the weights are unity and we DON't want to scale to triangle area
                call face_integral_MFIE(num_integration, xi_eta, weights, nodes_q, &
                                    num_integration, xi_eta, weights, nodes_p, jk_0, normals(p, :), T_form, I_Z)
                I_Z = I_Z/4.0/pi
            end if

            ! there is no symmetry for MFIE
            Z_face(p, :, q, :) = I_Z

        end do
    end do

end subroutine Z_MFIE_faces_self
