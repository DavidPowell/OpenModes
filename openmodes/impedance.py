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
"Classes for holding impedance matrix objects"

from __future__ import division

# numpy and scipy
import numpy as np
import scipy.linalg as la

from openmodes.helpers import inc_slice
from openmodes.basis import get_combined_basis
from openmodes.eig import eig_newton_bordered
from openmodes.vector import VectorParts


class ImpedanceMatrix(object):
    """Holds an impedance matrix as a single object
    """

    reciprocal = False

    def __init__(self, s, Z, basis_o, basis_s, operator, part_o, part_s):
        self.s = s
        self.Z = Z

        self.operator = operator
        self.part_o = part_o
        self.part_s = part_s

        self.basis_o = basis_o
        self.basis_s = basis_s

        # prevent external modification, to allow caching
        Z.setflags(write=False)

    def __getitem__(self, index):
        """Evaluates all or part of the impedance matrix, and returns it as
        an array.
        """
        return self.Z[index]

    def solve(self, V, cache=True):
        """Solve for the current, given a voltage vector

        Parameters
        ----------
        V : ndarray
            The source vector
        cache : boolean, optional
            If True, cache the LU factorisation to avoid recalculating it
        """
        if cache and hasattr(self, "factored_matrix"):
            lu = self.factored_matrix
        else:
            lu = la.lu_factor(self[:])
            if cache:
                self.factored_matrix = lu

        if self.part_s is None:
            # e.g. if this is the result of a projection onto modes
            vector = np.empty_like(V)
        else:
            vector = VectorParts(self.part_s, self.operator.basis_container,
                                 dtype=np.complex128)

        vector[:] = la.lu_solve(lu, V)
        return vector

    def eigenmodes(self, num_modes=None, use_gram=None, start_j=None):
        """Calculate the eigenimpedance and eigencurrents of each part's modes

        The modes with the smallest imaginary part of their impedance will be
        returned.

        Note that the impedance matrix can easily be *ill-conditioned*.
        Therefore this routine can return junk results, particularly if the
        mesh is dense.

        Parameters
        ----------
        num_modes : integer, optional
            The number of modes to find for each part
        use_gram : boolean, optional
            Solve a generalised problem involving the Gram matrix, which scales
            out the basis functions to get the physical eigenimpedances
        start_j : ndarray, optional
            If specified, then iterative solutions will be found, starting
            from the vectors in this array

        Returns
        -------
        eigenimpedance : ndarray (num_modes)
            The scalar eigenimpedance for each mode
        eigencurrent : VectorParts (num_basis, num_modes)
            A vector containing the eigencurrents of each mode in its columns
        """

        if start_j is not None:
            # An iterative solution will be performed, based on the given
            # current distribution. In this case the Gram matrix is used and
            # the use_gram parameter is ignored
            eigencurrent = np.empty_like(start_j)
            num_modes = start_j.shape[1]
            eigenimpedance = np.empty(num_modes, np.complex128)

            G = self.basis_o.gram_matrix
            start_j /= np.sqrt(np.diag(start_j.T.dot(G.dot(start_j))))

            Z = self[:]
            for mode in xrange(num_modes):
                start_z = start_j[:, mode].dot(Z.dot(start_j[:, mode]))
                res = eig_newton_bordered(Z, start_z, start_j[:, mode], G=G)
                eigencurrent[:, mode] = res['eigvec']
                eigenimpedance[mode] = res['eigval']

            eigencurrent /= np.sqrt(np.diag(eigencurrent.T.dot(G.dot(eigencurrent))))

        else:
            # The direct solution, which may or may not use the Gram matrix

            if use_gram:
                G = self.basis_o.gram_matrix
                z_all, v_all = la.eig(self[:], G)
            else:
                z_all, v_all = la.eig(self[:])

            if start_j is None:
                which_z = np.argsort(abs(z_all.imag))[:num_modes]
            else:
                which_z = np.dot(start_j.T, v_all).argmax(1)

            eigenimpedance = z_all[which_z]
            v = v_all[:, which_z]

            if use_gram:
                eigencurrent = v/np.sqrt(np.diag(v.T.dot(G.dot(v))))
            else:
                eigencurrent = v/np.sqrt(np.sum(v**2, axis=0))

        return eigenimpedance, eigencurrent

    def weight(self, modes_o, modes_s=None, return_arrays=False):
        """Calculate a reduced impedance matrix based on the scalar impedance
        of the modes of each part, and the scalar coupling coefficients.

        Parameters
        ----------
        modes_o : ndarray
            The modal currents of the observer part
        modes_s : ndarray, optional
            The modal currents of the source part
        return_arrays : boolean, optional
            Return the impedance arrays directly, instead of constructing an
            `ImpedanceMatrix` object

        Returns
        -------
        L_red, S_red : ndarray
            the reduced inductance and susceptance matrices

        or,

        Z : ImpedanceMatrix
            the reduced impedance matrix object
        """
        if modes_s is None:
            modes_s = modes_o

        # TODO: special handling of self terms to zero off-diagonal terms?

        Z_red = modes_o.T.dot(self.Z.dot(modes_s))

        if return_arrays:
            return Z_red
        else:
            return self.__class__(self.s, Z_red, None, None,
                                  self.operator, None, None)

    def source_modes(self, V, num_modes, mode_currents):
        "Take a source field, and project it onto the modes of the system"

        # calculate separately
        V_red = np.zeros(num_modes, np.complex128)
        for i in xrange(num_modes):
            V_red[i] = mode_currents[:, i].dot(V)

        return V_red

    @property
    def shape(self):
        "The shape of all matrices"
        return self.Z.shape

    @property
    def T(self):
        "A transposed version of the impedance matrix"
        # note interchange of source and observer basis functions
        return self.__class__(self.s, self.Z.T, self.basis_s,
                              self.basis_o, self.operator, self.part_s,
                              self.part_o)

    @staticmethod
    def combine_parts(matrices, s, part_o, part_s):
        """Combine a set of impedance matrices for sub-parts for a single
        matrix

        Parameters
        ----------
        matrices : list of list of EfieImpedanceMatrix
            The impedance matrices to be combined
        s : complex
            The frequency at which the impedance was evaluated

        Returns
        -------
        impedance : EfieImpedanceMatrix
            An object containing the combined impedance matrices
        """

        total_rows = sum(M[0].shape[0] for M in matrices)
        total_cols = sum(M.shape[1] for M in matrices[0])
        Z_tot = np.empty((total_rows, total_cols), np.complex128)

        row_offset = 0
        for row in matrices:
            row_size = row[0].shape[0]
            col_offset = 0

            for matrix in row:
                col_size = matrix.shape[1]
                Z_tot[row_offset:row_offset+row_size,
                      col_offset:col_offset+col_size] = matrix.Z
                col_offset += col_size
            row_offset += row_size

        basis = get_combined_basis(basis_list=[m.basis_o for m in row])
        return ImpedanceMatrix(s, Z_tot, basis, basis, matrix.operator,
                               part_o, part_s)


class EfieImpedanceMatrix(ImpedanceMatrix):
    """Holds an impedance matrix from the electric field integral equation,
    which contains two separate parts corresponding to the vector and scalar
    potential.

    This is a single impedance matrix for the whole system. Note that elements
    of the matrix should not be modified after being added to this object.
    """

    reciprocal = True

    def __init__(self, s, L, S, basis_o, basis_s, operator, part_o, part_s):
        self.s = s
        assert(L.shape == S.shape)
        self.L = L
        self.S = S

        self.operator = operator
        self.part_o = part_o
        self.part_s = part_s

        self.basis_o = basis_o
        self.basis_s = basis_s

        # prevent external modification, to allow caching
        L.setflags(write=False)
        S.setflags(write=False)

    def __getitem__(self, index):
        """Evaluates all or part of the impedance matrix, and returns it as
        an array.
        """
        return self.s*self.L[index] + self.S[index]/self.s

    def weight(self, modes_o, modes_s=None, return_arrays=False):
        """Calculate a reduced impedance matrix based on the scalar impedance
        of the modes of each part, and the scalar coupling coefficients.

        Parameters
        ----------
        modes_o : ndarray
            The modal currents of the observer part
        modes_s : ndarray, optional
            The modal currents of the source part
        return_arrays : boolean, optional
            Return the impedance arrays directly, instead of constructing an
            `ImpedanceMatrix` object

        Returns
        -------
        L_red, S_red : ndarray
            the reduced inductance and susceptance matrices

        or,

        Z : ImpedanceMatrix
            the reduced impedance matrix object
        """
        if modes_s is None:
            modes_s = modes_o

        # TODO: special handling of self terms to zero off-diagonal terms?

        L_red = modes_o.T.dot(self.L.dot(modes_s))
        S_red = modes_o.T.dot(self.S.dot(modes_s))

        if return_arrays:
            return L_red, S_red
        else:
            return self.__class__(self.s, L_red, S_red, None, None,
                                  self.operator, None, None)

    @property
    def shape(self):
        "The shape of all matrices"
        return self.L.shape

    @property
    def T(self):
        "A transposed version of the impedance matrix"
        # note interchange of source and observer basis functions
        return self.__class__(self.s, self.L.T, self.S.T, self.basis_s,
                              self.basis_o, self.operator, self.part_s,
                              self.part_o)

    @staticmethod
    def combine_parts(matrices, s, part_o, part_s):
        """Combine a set of impedance matrices for sub-parts for a single
        matrix

        Parameters
        ----------
        matrices : list of list of EfieImpedanceMatrix
            The impedance matrices to be combined
        s : complex
            The frequency at which the impedance was evaluated

        Returns
        -------
        impedance : EfieImpedanceMatrix
            An object containing the combined impedance matrices
        """

        total_rows = sum(M[0].shape[0] for M in matrices)
        total_cols = sum(M.shape[1] for M in matrices[0])
        L_tot = np.empty((total_rows, total_cols), np.complex128)
        S_tot = np.empty_like(L_tot)

        row_offset = 0
        for row in matrices:
            row_size = row[0].shape[0]
            col_offset = 0

            for matrix in row:
                col_size = matrix.shape[1]
                L_tot[row_offset:row_offset+row_size,
                      col_offset:col_offset+col_size] = matrix.L
                S_tot[row_offset:row_offset+row_size,
                      col_offset:col_offset+col_size] = matrix.S
                col_offset += col_size
            row_offset += row_size

        basis = get_combined_basis(basis_list=[m.basis_o for m in row])
        return EfieImpedanceMatrix(s, L_tot, S_tot, basis, basis,
                                   matrix.operator, part_o, part_s)


class EfieImpedanceMatrixLoopStar(EfieImpedanceMatrix):
    """A specialised impedance matrix which contains the results calculated in
    a loop-star basis. It is able to report which regions of the impedance
    matrices correspond to the loops and stars.
    """

    @property
    def loop_range_o(self):
        return slice(0, self.basis_o.num_loops)

    @property
    def loop_range_s(self):
        return slice(0, self.basis_s.num_loops)

    @property
    def star_range_o(self):
        return slice(self.basis_o.num_loops, self.shape[0])

    @property
    def star_range_s(self):
        return slice(self.basis_s.num_loops, self.shape[1])

    @staticmethod
    def combine_parts(matrices, s, part_o, part_s):
        """Combine a set of impedance matrices for sub-parts for a single
        matrix

        Parameters
        ----------
        matrices : list of list of EfieImpedanceMatrix
            The impedance matrices to be combined
        s : complex
            The frequency at which the impedance was evaluated

        Returns
        -------
        Z : EfieLoopStarImpedanceMatrix
            An object containing the combined impedance matrices
        """

        total_rows = sum(M[0].shape[0] for M in matrices)
        total_cols = sum(M.shape[1] for M in matrices[0])
        L_tot = np.empty((total_rows, total_cols), np.complex128)
        S_tot = np.zeros_like(L_tot)

        basis = get_combined_basis(basis_list=[row[0].basis_o
                                               for row in matrices])

        loop_range_o = slice(0, 0)
        star_range_o = slice(basis.num_loops, basis.num_loops)

        for col_count, row in enumerate(matrices):
            m = row[0]
            loop_range_o = inc_slice(loop_range_o, m.basis_o.num_loops)
            star_range_o = inc_slice(star_range_o, m.basis_o.num_stars)

            loop_range_s = slice(0, 0)
            star_range_s = slice(basis.num_loops, basis.num_loops)

            for row_count, m in enumerate(row):
                loop_range_s = inc_slice(loop_range_s, m.basis_s.num_loops)
                star_range_s = inc_slice(star_range_s, m.basis_s.num_stars)

                # S only has stars
                S_tot[star_range_o, star_range_s] = m.S[m.star_range_o, m.star_range_s]

                # Some of these arrays may have one dimension of size zero if
                # there are no loops, but this is handled automatically.
                L_tot[loop_range_o, loop_range_s] = m.L[m.loop_range_o, m.loop_range_s]
                L_tot[loop_range_o, star_range_s] = m.L[m.loop_range_o, m.star_range_s]
                L_tot[star_range_o, loop_range_s] = m.L[m.star_range_o, m.loop_range_s]
                L_tot[star_range_o, star_range_s] = m.L[m.star_range_o, m.star_range_s]

        return EfieImpedanceMatrixLoopStar(s, L_tot, S_tot, basis, basis,
                                           m.operator, part_o, part_s)


class ImpedanceParts(object):
    """Holds a impedance matrices calculated at a specific frequency

    This consists of separate matrices for each part, and their mutual
    coupling terms.
    """

    def __init__(self, s, parent_part_o, parent_part_s, matrices,
                 impedance_class):
        """
        Parameters
        ----------
        s : complex
            complex frequency at which to calculate impedance (in rad/s)
        parent_part : CompositePart
            The main part which holds all other parts represented by this
            impedance matrix
        matrices : dictionary of ImpedanceMatrix
            The impedance matrix for each part, or mutual terms between them
        impedance_class : type
            The class of the impedance matrices of each part
        """
        self.s = s
        self.parent_part_o = parent_part_o
        self.parent_part_s = parent_part_s
        self.matrices = matrices
        self.impedance_class = impedance_class

    def __getitem__(self, index):
        """Allow self or mutual impedances of parts at any level to be
        accessed. If the impedance of a part is not found, then it will be
        constructed by combining the sub-parts

        Parameters
        ----------
        index : tuple, len 2
            A tuple containing the source and observer part.
        """
        if index == slice(None):
            index = (self.parent_part_o, self.parent_part_s)

        try:
            return self.matrices[index]
        except KeyError:
            if ((len(index) == 2) and (index[0] in self.parent_part_o) and
                    (index[1] in self.parent_part_s)):
                # a valid self or mutual term
                parent_o, parent_s = index
            else:
                raise KeyError("Invalid parts specified")

            combined = []
            for part_o in parent_o.iter_single():
                combined.append([])
                for part_s in parent_s.iter_single():
                    combined[-1].append(self.matrices[part_o, part_s])

            if (len(combined) == 1) and len(combined[0]) == 1:
                # don't bother combining if we only have a single matrix
                new_matrix = combined[0][0]
            else:
                new_matrix = self.impedance_class.combine_parts(combined,
                                                                self.s,
                                                                parent_o,
                                                                parent_s)
            self.matrices[parent_o, parent_s] = new_matrix
            if parent_s != parent_o and self.impedance_class.reciprocal:
                self.matrices[parent_s, parent_o] = new_matrix.T
            return new_matrix

    def solve(self, V, part=None, cache=True):
        """Solve a particular part in the system for a source vector vector

        Parameters
        ----------
        V : ndarray
            The vector for which to solve this impedance matrix
        part : Part, optional
            Can be used to specify a portion of this matrix to solve, which
            corresponds to a particular sub-part. If not specified, the
            full matrix will be solved
        """
        if part is None:
            if self.parent_part_o != self.parent_part_s:
                raise ValueError("Cannot solve partial impedance matrix")
            else:
                part = self.parent_part_o
        return self[part, part].solve(V, cache=cache)

    def eigenmodes(self, part=None, num_modes=None, use_gram=None,
                   start_j=None):
        """Calculate the eigenimpedance and eigencurrents of each part's modes

        The modes with the smallest imaginary part of their impedance will be
        returned.

        Note that the impedance matrix can easily be *ill-conditioned*.
        Therefore this routine can return junk results, particularly if the
        mesh is dense.

        Parameters
        ----------
        part : Part
            The part for which to find the eigenmodes. If not specified, the
            eigenmodes of the whole system will be calculated
        num_modes : integer, optional
            The number of modes to find for each part
        use_gram : boolean, optional
            Solve a generalised problem involving the Gram matrix, which scales
            out the basis functions to get the physical eigenimpedances
        start_j : ndarray, optional
            If specified, then iterative solutions will be found, starting
            from the vectors in this array

        Returns
        -------
        eigenimpedance : ndarray (num_modes)
            The scalar eigenimpedance for each mode
        eigencurrent : VectorParts (num_basis, num_modes)
            A vector containing the eigencurrents of each mode in its columns
        """
        if part is None:
            if self.parent_part_o != self.parent_part_s:
                raise ValueError("Cannot get eigenvalues of partial "
                                 "impedance matrix")
            else:
                part = self.parent_part_o
        return self[part, part].eigenmodes(num_modes, use_gram, start_j)

    def weight(self, part_modes):
        """Calculate a reduced impedance matrix based on the scalar impedance
        of the modes of each part, and the scalar coupling coefficients. These
        are determined by projecting segments of the impedance matrix onto
        the provided modes.

        Parameters
        ----------
        part_modes : sequence of tuples of (part, modes_vec)
            For each part `modes_vec` will be used as modes in which to base
            the reduced model. Multiple modes should be represented by mutiple
            columns of `modes_vec`.

        Returns
        -------
        Z : ImpedanceMatrix
            the reduced impedance matrix object

        Note that indirectly including a part more than once in part_modes
        (e.g. by including it and its parent part) will yield invalid results.
        """

        # calculate modal impedances for each part separately, and include
        # coupling between all modess of different parts
        L_red = []
        S_red = []

        for part_o, modes_o in part_modes:
            L_row = []
            S_row = []
            for part_s, modes_s in part_modes:
                L, S = self[part_o, part_s].weight(modes_o, modes_s,
                                                   return_arrays=True)
                L_row.append(L)
                S_row.append(S)
            L_red.append(np.hstack(L_row))
            S_red.append(np.hstack(S_row))

        L_red = np.vstack(L_red)
        S_red = np.vstack(S_red)
        return self.impedance_class(self.s, L_red, S_red, None, None,
                                    self[part_o, part_s].operator, None, None)
