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

module constants
    implicit none
	
    integer, parameter :: WP = 8 ! working precision, can be 4 or 8
    ! when adjusting working precision, need to change line in file .f2py_f2cmap
    integer, parameter :: DP = 8
    integer, parameter :: SP = 4
end module


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

module vectors
!    interface
!
!        pure function cross_product(a, b)
!            use constants
!            implicit none
!            real(WP), intent(in), dimension(:) :: a, b
!            real(WP), dimension(3) :: cross_product
!        end function
!    
!        pure function mag(vector)
!            use constants
!            implicit none
!        
!            real(WP), intent(in), dimension(:) :: vector
!            real(WP) :: mag
!        end function
!    end interface

contains


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


end module vectors