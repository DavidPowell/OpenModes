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

from __future__ import division

# numpy and scipy
import numpy as np
import scipy.linalg as la
import itertools


class DummyBasis(object):
    """A minimal basis function data structure for combined impedance matrix
    objects. Does not contain full basis functions, but serves as a place
    holder for information which would be obtained form them
    """
    def __init__(self, **kwargs):
        for kw, val in kwargs.iteritems():
            setattr(self, kw, val)


# TODO: ImpedanceMatrix may need to know about number of loops and stars?
class EfieImpedanceMatrix(object):
    """Holds an impedance matrix from the electric field integral equation,
    which contains two separate parts corresponding to the vector and scalar
    potential.

    This is a single impedance matrix for the whole system. Note that elements
    of the matrix should not be modified after being added to this object.
    """

    def __init__(self, s, L, S, basis_o, basis_s):
        self.s = s
        assert(L.shape == S.shape)
        self.L = L
        self.S = S

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
            lu = la.lu_factor(self[:], overwrite_a=True)
            if cache:
                self.factored_matrix = lu
        return la.lu_solve(lu, V)

    def eigenmodes(self, num_modes, use_gram=False):
        """Calculate the eigenimpedance and eigencurrents of each part's modes

        The modes with the smallest imaginary part of their impedance will be
        returned.

        Note that the impedance matrix can easily be *ill-conditioned*.
        Therefore this routine can return junk results, particularly if the
        mesh is dense.

        Parameters
        ----------
        num_modes : integer
            The number of modes to find for each part
        use_gram : boolean, optional
            Solve a generalised problem involving the Gram matrix, which scales
            out the basis functions to get the physical eigenimpedances
        """

        if use_gram:
            Gw, Gv = self.basis_o.gram_factored
            Gwm = np.diag(1.0/Gw)
            Zd = Gwm.dot(Gv.T.dot(self[:].dot(Gv.dot(Gwm))))
            z_all, v_all = la.eig(Zd)
            #G = self.basis_o.gram_matrix
            #z_all, v_all = la.eig(self[:], G)
        else:
            z_all, v_all = la.eig(self[:])

        which_z = np.argsort(abs(z_all.imag))[:num_modes]
        eigenimpedance = z_all[which_z]

        v = v_all[:, which_z]
        eigencurrent = v/np.sqrt(np.sum(v**2, axis=0))

        if use_gram:
            eigencurrent = Gv.dot(Gwm.dot(eigencurrent))

        return eigenimpedance, eigencurrent

    def impedance_modes(self, num_modes, mode_currents_o,
                        mode_currents_s=None, return_arrays=False):
        """Calculate a reduced impedance matrix based on the scalar impedance
        of the modes of each part, and the scalar coupling coefficients.

        Parameters
        ----------
        s : number
            complex frequency at which to calculate impedance (in rad/s)
        num_modes : integer
            The number of modes to take into account for each part
        mode_currents_o : array
            The modal currents of the observer part
        mode_currents_s : array, optional
            The modal currents of the source part (only for off-diagonal terms
            where the source differs from the observer)
        return_arrays : boolean, optional
            Return the impedance arrays directly, instead of constructing an
            `ImpedanceMatrix` object

        Returns
        -------
        L_red, S_red : ndarray
            the reduced inductance and susceptance matrices
        """

        # Parts are already combined, so we are talking about modes of
        # the complete coupled system
        L_red = np.zeros((num_modes, num_modes), np.complex128)
        S_red = np.zeros_like(L_red)

        if mode_currents_s is None:
            for i in xrange(num_modes):
                # only diagonal terms are non-zero
                L_red[i, i] = mode_currents_o[:, i].dot(self.L.dot(mode_currents_o[:, i]))
                S_red[i, i] = mode_currents_o[:, i].dot(self.S.dot(mode_currents_o[:, i]))
        else:
            for i, j in itertools.product(xrange(num_modes), xrange(num_modes)):
                L_red[i, j] = mode_currents_o[:, i].dot(self.L.dot(mode_currents_s[:, j]))
                S_red[i, j] = mode_currents_o[:, i].dot(self.S.dot(mode_currents_s[:, j]))

        if return_arrays:
            return L_red, S_red
        else:
            return EfieImpedanceMatrix(self.s, L_red, S_red, None, None)

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
        return self.L.shape

    @property
    def T(self):
        "A transposed version of the impedance matrix"
        # note interchange of source and observer basis functions
        return self.__class__(self.s, self.L.T, self.S.T, self.basis_s,
                              self.basis_o)


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


class ImpedanceParts(object):
    """Holds a impedance matrices calculated at a specific frequency

    This consists of separate matrices for each part, and their mutual
    coupling terms.
    """
    # TODO: needs to be made agnostic regarding the type of impedance
    # matrix which it contains

    def __init__(self, s, num_parts, matrices):
        """
        Parameters
        ----------
        s : complex
            complex frequency at which to calculate impedance (in rad/s)
        matrices : list of list of ImpedanceMatrix
            The impedance matrix for each part, or mutual terms between them
        num_parts : int
            The number of parts in the system
        """
        self.s = s
        self.num_parts = num_parts
        self.matrices = matrices

    def __getitem__(self, index):
        """Allow matrices of individual parts to be accessed"""
        try:
            return self.matrices[index]
        except TypeError:
            if type(index[0]) == slice:
                raise TypeError("Cannot slice the first dimension")
            return self.matrices[index[0]][index[1]]

    def combine_parts(self, V=None):
        """Evaluate the self and mutual impedances of all parts combined into
        a pair of matrices for the whole system.

        Parameters
        ----------
        V : list of arrays, optional
            The corresponding voltages, which can also be combined in the
            same fashion

        Returns
        -------
        impedance : impedance object of appropriate type
            An object containing the combined impedance matrices
        V : array
            If given as an input, the voltage vector will also be combined and
            returned as an output
        """

        total_size = sum(M[0].shape[0] for M in self.matrices)
        L_tot = np.empty((total_size, total_size), np.complex128)
        S_tot = np.empty_like(L_tot)

        row_offset = 0
        for row in self.matrices:
            row_size = row[0].shape[0]
            col_offset = 0
            for matrix in row:
                col_size = matrix.shape[1]
                L_tot[row_offset:row_offset+row_size, col_offset:col_offset+col_size] = matrix.L
                S_tot[row_offset:row_offset+row_size, col_offset:col_offset+col_size] = matrix.S
                col_offset += col_size
            row_offset += row_size

        # TODO: Note that combined impedance matrices lose all information
        # about basis functions, so it has to be put into a dummy basis
        # function object
        basis = DummyBasis()
        Z = EfieImpedanceMatrix(self.s, L_tot, S_tot, basis, basis)

        if V is not None:
            return Z, np.hstack(V)
        else:
            return Z

    def eigenmodes(self, num_modes, use_gram=False):
        """Calculate the eigenimpedance and eigencurrents of each part's modes

        The modes with the smallest imaginary part of their impedance will be
        returned.

        Note that the impedance matrix is typically *ill-conditioned*.
        Therefore this routine can return junk results, particularly if the
        mesh is dense.

        Parameters
        ----------
        num_modes : integer
            The number of modes to find for each part
        use_gram : boolean, optional
            Solve a generalised problem involving the Gram matrix, which scales
            out the basis functions to get the physical eigenimpedances   
        """

        # TODO: cache this if parts are identical (should be upstream caching
        # of L and S for this to work)
        mode_impedances = []
        mode_currents = []
        for count in xrange(self.num_parts):
            eig_z, eig_current = self.matrices[count][count].eigenmodes(num_modes, use_gram)

            mode_impedances.append(eig_z)
            mode_currents.append(eig_current)

        return mode_impedances, mode_currents

    def impedance_modes(self, num_modes, mode_currents, combine=True):
        """Calculate a reduced impedance matrix based on the scalar impedance
        of the modes of each part, and the scalar coupling coefficients.

        Parameters
        ----------
        s : number
            complex frequency at which to calculate impedance (in rad/s)
        num_modes : integer
            The number of modes to take into account for each part
        mode_currents : list
            The modal currents of each part

        Returns
        -------
        L_red, S_red : ndarray
            the reduced inductance and susceptance matrices
        """

        # calculate modal impedances for each part separately, and include
        # coupling between all modes of different parts
        num_parts = self.num_parts
        L_red = np.zeros((num_parts, num_modes, num_parts, num_modes),
                         np.complex128)
        S_red = np.zeros_like(L_red)

        for i, j in itertools.product(xrange(num_parts), xrange(num_parts)):
            # The mutual impedance terms of modes within the
            # same resonator have L and S exactly cancelling,
            # so currently they are not calculated
            M = self.matrices[i][j]

            if i == j:
                # explicitly handle the diagonal cse
                L, S = M.impedance_modes(num_modes, mode_currents[i],
                                         return_arrays=True)
            else:
                L, S = M.impedance_modes(num_modes, mode_currents[i],
                                         mode_currents[j], return_arrays=True)
            L_red[i, :, j, :] = L
            S_red[i, :, j, :] = S

        if combine:
            L_red = L_red.reshape((num_parts*num_modes, num_parts*num_modes))
            S_red = S_red.reshape((num_parts*num_modes, num_parts*num_modes))
            return EfieImpedanceMatrix(self.s, L_red, S_red, None, None)
        else:
            raise NotImplementedError

    def source_modes(self, V, num_modes, mode_currents, combine=True):
        "Take a source field, and project it onto the modes of each part"

        V_red = np.zeros((self.num_parts, num_modes), np.complex128)

        for i in xrange(self.num_parts):
            V_red[i] = self.matrices[i][i].source_modes(V[i], num_modes,
                                                        mode_currents[i])

        if combine:
            V_red = V_red.reshape(self.num_parts*num_modes)

        return V_red


def inc_slice(s, inc):
    """Increment a slice so that it starts at the current stop, and the current
    stop is incremented by some amount"""
    return slice(s.stop, s.stop+inc)


class ImpedancePartsLoopStar(ImpedanceParts):
    """A specialised list of impedance matrices for holding results
    calculated with loop-star basis functions. Ensures that when results are
    combined, loops and stars are represented as global blocks for the whole
    system, rather than being separated for each individual part
    """

    def combine_parts(self, V=None):
        """Evaluate the self and mutual impedances of all parts combined into
        a pair of matrices for the whole system. Loops and stars are all
        combined into global blocks.

        Parameters
        ----------
        V : list of arrays, optional
            The corresponding voltages, which can also be combined in the
            same fashion

        Returns
        -------
        impedance : EfieImpedanceMatrixLoopStar
            An object containing the combined impedance matrices
        V : array
            If given as an input, the voltage vector will also be combined and
            returned as an output
        """

        total_size = sum(M[0].shape[0] for M in self.matrices)
        num_loops = sum(M[0].basis_o.num_loops for M in self.matrices)
        L_tot = np.empty((total_size, total_size), np.complex128)
        S_tot = np.zeros_like(L_tot)
        if V is not None:
            V_tot = np.empty(total_size, np.complex128)

        loop_range_o = slice(0, 0)
        star_range_o = slice(num_loops, num_loops)

        for col_count, row in enumerate(self.matrices):
            m = row[0]
            loop_range_o = inc_slice(loop_range_o, m.basis_o.num_loops)
            star_range_o = inc_slice(star_range_o, m.basis_o.num_stars)

            loop_range_s = slice(0, 0)
            star_range_s = slice(num_loops, num_loops)

            if V is not None:
                V_tot[loop_range_o] = V[col_count][m.loop_range_o]
                V_tot[star_range_o] = V[col_count][m.star_range_o]

            for m in row:
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

        basis = DummyBasis(num_loops=num_loops)
        Z = EfieImpedanceMatrixLoopStar(self.s, L_tot, S_tot, basis, basis)

        if V is not None:
            return Z, V_tot
        else:
            return Z
