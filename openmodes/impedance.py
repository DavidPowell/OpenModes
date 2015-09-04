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
from openmodes.array import LookupArray


class AbstractImpedanceMatrix(object):
    """An abstract base class for impedance matrices of a single part. The
    sub-matrices are stored in the `matrices` element.

    At minimum, a subclass must override __getitem__, in order to evaulate the
    combination of matrices.
    """

    def __init__(self, matrices, metadata):
        """
        Parameters
        ----------
        matrices : dict
            The component matrices which are held, with their names as keys
        metadata : dict
            The metadata for this element. At minimum, it must contain the
            elements 's', 'operator', 'part_o', 'part_s', 'basis_o', 'basis_s'
            'symmetric'
        """
        # TODO: Some kind of traits library could replace these dicts
        self.matrices = matrices
        self.md = metadata

        if not all(x in metadata for x in ('s', 'operator', 'part_o', 'part_s',
                                           'basis_o', 'basis_s', 'symmetric')):
            raise ValueError("Incomplete metadata for impedance matrix: %s" %
                             metadata)

        self.shape = None
        # prevent external modification, to allow caching
        for matrix in self.matrices.values():
            matrix.setflags(write=False)
            if self.shape is None:
                self.shape = matrix.shape
            elif matrix.shape != self.shape:
                raise ValueError("Matrices have inconsistent shapes")

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

        if self.md['part_s'] is None:
            # e.g. if this is the result of a projection onto modes
            vector = np.empty_like(V)
        else:
            vector = VectorParts(self.md['part_s'],
                                 self.md['operator'].basis_container,
                                 dtype=np.complex128)

        vector[:] = la.lu_solve(lu, V)
        return vector

    def eigenmodes(self, num_modes=None, use_gram=None, start_vec=None,
                   start_l_vec=None):
        """Calculate the eigenimpedance and eigencurrents of each part's modes

        The modes with the smallest imaginary part of their impedance will be
        returned.

        Note that the EFIE impedance matrix can easily be *ill-conditioned*.
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

        symmetric = self.md['symmetric']
        if start_vec is not None:

            if (not symmetric) and start_l_vec is None:
                raise ValueError("For non-symmetric operator, must give"
                                 "initial estimates for both left and right"
                                 "eigenvectors")

            # An iterative solution will be performed, based on the given
            # current distribution. In this case the Gram matrix is used and
            # the use_gram parameter is ignored
            v_r = np.empty_like(start_vec)
            num_modes = start_vec.shape[1]
            z = np.empty(num_modes, np.complex128)

            G = self.md['basis_o'].gram_matrix

            if symmetric:
                start_vec /= np.sqrt(np.diag(start_vec.T.dot(G.dot(start_vec))))
                start_l_vec = None
            else:
                start_vec /= np.diag(start_l_vec.T.dot(G.dot(start_vec)))
                v_l = np.empty_like(v_r)

            Z = self[:]
            for mode in range(num_modes):
                if symmetric:
                    start_z = start_vec[:, mode].dot(Z.dot(start_vec[:, mode]))
                    res = eig_newton_bordered(Z, start_z, start_vec[:, mode],
                                              B=G)
                else:
                    start_z = start_l_vec[:, mode].dot(Z.dot(start_vec[:, mode]))
                    res = eig_newton_bordered(Z, start_z, start_vec[:, mode], B=G,
                                              vl_0=start_l_vec[:, mode], w_tol=1e-13)
                    v_l[:, mode] = res['vl']

                v_r[:, mode] = res['vr']
                z[mode] = res['w']
        else:
            # The direct solution, which may or may not use the Gram matrix

            G = self.md['basis_o'].gram_matrix
            if symmetric:
                z_all, v_r_all = la.eig(self[:], G)
            else:
                z_all, v_r_all, v_l_all = la.eig(self[:], G, left=True)

            which_z = np.argsort(abs(z_all))[:num_modes]

            z = z_all[which_z]
            v_r = v_r_all[:, which_z]

            if symmetric:
                # Normalisation for symmetric system ensures that projector
                # applied multiple times has no additional effect.
                v_r /= np.sqrt(np.diag(v_r.T.dot(G.dot(v_r))))
            else:
                v_r = v_r_all[:, which_z]
                v_l = v_l_all[:, which_z].conjugate()

                # First scale the left eigenvectors so that dyadic is correct
                v_l /= np.diag(v_l.T.dot(G.dot(v_r)))

        # results should already be scaled
        if symmetric:
            return z, v_r
        else:
            return z, v_r, v_l

    def weight(self, modes_o, modes_s=None):
        """Calculate a reduced impedance matrix based on the scalar impedance
        of the modes of each part, and the scalar coupling coefficients.

        Parameters
        ----------
        modes_o : ndarray
            The modal currents of the observer part
        modes_s : ndarray, optional
            The modal currents of the source part

        Returns
        -------
        Z : ImpedanceMatrix
            the reduced impedance matrix object of the same class
        """
        if modes_s is None:
            modes_s = modes_o

        # TODO: special handling of self terms to zero off-diagonal terms?

        matrices_red = {}
        for key, val in self.matrices.items():
            matrices_red[key] = modes_o.T.dot(val.dot(modes_s))

        return self.__class__(matrices_red, self.md)

    def source_modes(self, V, num_modes, mode_currents):
        "Take a source field, and project it onto the modes of the system"

        # calculate separately
        V_red = np.zeros(num_modes, np.complex128)
        for i in range(num_modes):
            V_red[i] = mode_currents[:, i].dot(V)

        return V_red

    @property
    def T(self):
        "A transposed version of the impedance matrix"
        # note interchange of source and observer basis functions
        matrices_T = {key: val.T for key, val in self.matrices.items()}
        return self.__class__(matrices_T, self.md)

    @staticmethod
    def combine_parts(matrices, s, part_o, part_s):
        """Combine a set of impedance matrices for sub-parts for a single
        matrix

        Parameters
        ----------
        matrices : list of list of impedance matrix objects
            The impedance matrices to be combined, should all be of the same
            class
        s : complex
            The frequency at which the impedance was evaluated

        Returns
        -------
        impedance : impedance matrix object
            An object containing the combined impedance matrices, of the same
            class as those provided
        """

        total_rows = sum(M[0].shape[0] for M in matrices)
        total_cols = sum(M.shape[1] for M in matrices[0])

        matrices_tot = {}
        for key in matrices[0][0].matrices:
            matrices_tot[key] = np.empty((total_rows, total_cols),
                                         np.complex128)

        row_offset = 0
        for row in matrices:
            row_size = row[0].shape[0]
            col_offset = 0

            for matrix in row:
                col_size = matrix.shape[1]

                for key, val in matrix.matrices:
                    matrices_tot[key][row_offset:row_offset+row_size,
                                 col_offset:col_offset+col_size] = matrix[key]
                col_offset += col_size
            row_offset += row_size

        basis = get_combined_basis(basis_list=[m.basis_o for m in row])

        metadata = matrix.metadata.copy()
        metadata['basis_o'] = basis
        metadata['basis_s'] = basis
        metadata['part_s'] = part_s
        metadata['part_o'] = part_o
        return matrix.__class__(matrices_tot, metadata)


class SimpleImpedanceMatrix(AbstractImpedanceMatrix):
    "The simplest form of impedance matrix which has only a single member"

    @classmethod
    def build(cls, s, Z, basis_o, basis_s, operator, part_o, part_s,
              symmetric):
        matrices = {'Z': Z}
        metadata = {'basis_o': basis_o, 'basis_s': basis_s, 's': s,
                    'operator': operator, 'part_o': part_o, 'part_s': part_s,
                    'symmetric': symmetric}
        return cls(matrices, metadata)

    def __init__(self, matrices, metadata):
        if 'Z' not in matrices:
            raise ValueError("Simple impedance matrix must have matrix 'Z'")
        super(SimpleImpedanceMatrix, self).__init__(matrices, metadata)

    def __getitem__(self, index):
        """Evaluates all or part of the impedance matrix, and returns it as
        an array.
        """
        return self.matrices['Z'][index]


class EfieImpedanceMatrix(AbstractImpedanceMatrix):
    """Holds an impedance matrix from the electric field integral equation,
    which contains two separate parts corresponding to the vector and scalar
    potential.

    This is a single impedance matrix for the whole system. Note that elements
    of the matrix should not be modified after being added to this object.
    """

    @classmethod
    def build(cls, s, L, S, basis_o, basis_s, operator, part_o, part_s,
              symmetric):
        matrices = {'L': L, 'S': S}
        metadata = {'basis_o': basis_o, 'basis_s': basis_s, 's': s,
                    'operator': operator, 'part_o': part_o, 'part_s': part_s,
                    'symmetric': symmetric}
        return cls(matrices, metadata)

    def __init__(self, matrices, metadata):
        if 'L' not in matrices and 'S' in matrices:
            raise ValueError("EFIE impedance matrix must have matrices"
                             "'L' and 'S'")
        super(EfieImpedanceMatrix, self).__init__(matrices, metadata)

    def __getitem__(self, index):
        """Evaluates all or part of the impedance matrix, and returns it as
        an array.
        """
        return (self.md['s']*self.matrices['L'][index] +
                self.matrices['S'][index]/self.md['s'])

    def frequency_derivative(self, index=slice(None)):
        return (self.matrices['L'][index] +
                self.md['s']*self.matrices['dL_ds'][index] -
                self.matrices['S'][index]/self.md['s']**2 +
                self.matrices['dS_ds'][index]/self.md['s'])

    def frequency_derivative_P(self, index=slice(None)):
        return (2*self.md['s']*self.matrices['L'][index] +
                self.md['s']**2*self.matrices['dL_ds'][index] +
                self.matrices['dS_ds'][index])


class CfieImpedanceMatrix(AbstractImpedanceMatrix):
    """Holds an impedance matrix from the combined field integral equation,
    which contains separate parts corresponding to the vector and scalar
    potential of the EFIE, and another matrix for the MFIE .

    This is a single impedance matrix for the whole system. Note that elements
    of the matrix should not be modified after being added to this object.
    """

    @classmethod
    def build(cls, s, L, S, M, alpha, basis_o, basis_s, operator, part_o,
              part_s, symmetric):
        matrices = {'L': L, 'S': S, 'M': M}
        metadata = {'basis_o': basis_o, 'basis_s': basis_s, 's': s,
                    'operator': operator, 'part_o': part_o, 'part_s': part_s,
                    'symmetric': symmetric, 'alpha': alpha}
        return cls(matrices, metadata)

    def __init__(self, matrices, metadata):
        if not all(x in matrices for x in ('L', 'S', 'M')):
            raise ValueError("CFIE impedance matrix must have matrices"
                             "'L', 'S' and 'M'")
        if 'alpha' not in metadata:
            raise ValueError("CFIE alpha parameter missing")

        super(CfieImpedanceMatrix, self).__init__(matrices, metadata)

    def __getitem__(self, index):
        """Evaluates all or part of the impedance matrix, and returns it as
        an array.
        """
        return (self.md['alpha']*(self.md['s']*self.matrices['L'][index] +
                                  self.matrices['S'][index]/self.md['s']) +
                (1.0-self.md['alpha'])*self.matrices['M'])


class EfieImpedanceMatrixLoopStar(EfieImpedanceMatrix):
    """A specialised impedance matrix which contains the results calculated in
    a loop-star basis. It is able to report which regions of the impedance
    matrices correspond to the loops and stars.
    """

    @property
    def loop_range_o(self):
        return slice(0, self.md['basis_o'].num_loops)

    @property
    def loop_range_s(self):
        return slice(0, self.md['basis_s'].num_loops)

    @property
    def star_range_o(self):
        return slice(self.md['basis_o'].num_loops, self.shape[0])

    @property
    def star_range_s(self):
        return slice(self.md['basis_s'].num_loops, self.shape[1])

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

        basis = get_combined_basis(basis_list=[row[0].md['basis_o']
                                               for row in matrices])

        loop_range_o = slice(0, 0)
        star_range_o = slice(basis.num_loops, basis.num_loops)

        for col_count, row in enumerate(matrices):
            m = row[0]
            loop_range_o = inc_slice(loop_range_o, m.md['basis_o'].num_loops)
            star_range_o = inc_slice(star_range_o, m.md['basis_o'].num_stars)

            loop_range_s = slice(0, 0)
            star_range_s = slice(basis.num_loops, basis.num_loops)

            for row_count, m in enumerate(row):
                S = m.matrices['S']
                L = m.matrices['L']
                loop_range_s = inc_slice(loop_range_s, m.md['basis_s'].num_loops)
                star_range_s = inc_slice(star_range_s, m.md['basis_s'].num_stars)

                # S only has stars
                S_tot[star_range_o, star_range_s] = S[m.star_range_o, m.star_range_s]

                # Some of these arrays may have one dimension of size zero if
                # there are no loops, but this is handled automatically.
                L_tot[loop_range_o, loop_range_s] = L[m.loop_range_o, m.loop_range_s]
                L_tot[loop_range_o, star_range_s] = L[m.loop_range_o, m.star_range_s]
                L_tot[star_range_o, loop_range_s] = L[m.star_range_o, m.loop_range_s]
                L_tot[star_range_o, star_range_s] = L[m.star_range_o, m.star_range_s]

        # TODO: check symmetric
        return EfieImpedanceMatrixLoopStar.build(s, L_tot, S_tot, basis, basis,
                                                 m.md['operator'], part_o,
                                                 part_s, True)


class PenetrableImpedanceMatrix(AbstractImpedanceMatrix):
    """Holds an impedance matrix from a surface equivalent problem for
    a penetrable scatterer

    Notation is from Yla-Oijala, Radio Science 2005
    """

    # The matrices and metadata needed by this class
    matrix_names = ('L_i', 'L_o', 'S_i', 'S_o', 'K_i', 'K_o')
    metadata_names = ('basis_o', 'basis_s', 's', 'operator', 'part_o',
                      'part_s', 'symmetric', 'w_EFIE_i', 'w_MFIE_i',
                      'w_EFIE_o', 'w_MFIE_o', 'eta_i', 'eta_o')

    @classmethod
    def build(cls, s, L_i, L_o, S_i, S_o, K_i, K_o, eta_i, eta_o, w_EFIE_i,
              w_MFIE_i, w_EFIE_o, w_MFIE_o, basis_o, basis_s, operator,
              part_o, part_s, symmetric):
        """Build the matrices and metadata structures, and use it to create the
        object, which is returned"""
        loc = locals()
        return cls({m: loc[m] for m in cls.matrix_names},
                   {m: loc[m] for m in cls.metadata_names})

    def __init__(self, matrices, metadata):
        if not all(m in matrices for m in self.matrix_names):
            raise ValueError("Penetrable impedance matrix missing matrices")

        if not all(m in metadata for m in self.metadata_names):
            raise ValueError("Penetrable impedance matrix missing metadata")

        super(PenetrableImpedanceMatrix, self).__init__(matrices, metadata)

    def __getitem__(self, index):
        """Evaluates all or part of the impedance matrix, and returns it as
        an array.
        """
        s = self.md['s']
        D_i = s*self.matrices['L_i'][index]+self.matrices['S_i'][index]/s
        D_o = s*self.matrices['L_o'][index]+self.matrices['S_o'][index]/s
        K_i = self.matrices['K_i'][index]
        K_o = self.matrices['K_o'][index]
        eta_o = self.md['eta_o']
        eta_i = self.md['eta_i']
        w_EFIE_i = self.md['w_EFIE_i']
        w_EFIE_o = self.md['w_EFIE_o']
        w_MFIE_i = self.md['w_MFIE_i']
        w_MFIE_o = self.md['w_MFIE_o']
        return np.vstack((np.hstack((eta_o*D_o*w_EFIE_o + eta_i*D_i*w_EFIE_i,
                                     -K_o*w_EFIE_o-K_i*w_EFIE_i)),
                         (np.hstack((K_o*w_MFIE_o+K_i*w_MFIE_i,
                                     D_o/eta_o*w_MFIE_o + D_i/eta_i*w_MFIE_i)))))


class ImpedanceMatrixLA(object):
    """An impedance matrix based on LookupArray, which can hold matrices for
    Parts of arbitrary level"""

    matrix_names = ('Z',)

    def __init__(self, part_o, part_s, basis_container, sources, unknowns,
                 metadata=None, matrices=None, derivatives=None):
        self.md = metadata or dict()
        self.part_o = part_o
        self.part_s = part_s
        self.basis_container = basis_container
        self.sources = sources
        self.unknowns = unknowns

        if matrices is None:
            # create the empty matrices
            self.matrices = {name: LookupArray(((part_o, basis_container),
                                                (part_s, basis_container)),
                                               dtype=np.complex128)
                             for name in self.matrix_names}
        else:
            self.matrices = matrices

        # create the frequency derivatives of the matrices
        if derivatives is None:
            self.der = None
        elif derivatives is True:
            self.der = {name: LookupArray(((part_o, basis_container,),
                                           (part_s, basis_container)),
                                          dtype=np.complex128)
                        for name in self.matrix_names}
        else:
            self.der = derivatives

    def val(self):
        "The value of the impedance matrix"
        Z = LookupArray((self.sources, (self.part_o, self.basis_container), self.unknowns,
                         (self.part_s, self.basis_container)), dtype=np.complex128)
        Z.simple_view()[:] = self.matrices['Z']
        return Z

    def clear_cached(self):
        "Clear any cached data"
        if hasattr(self, "lu_factored"):
            del self.lu_factored

    def factored(self):
        "Caches the LU factorisation of the matrix"
        try:
            return self.lu_factored
        except AttributeError:
            self.lu_factored = la.lu_factor(self.val().simple_view())
            return self.lu_factored

    def solve(self, vec):
        """Solve the impedance matrix for a source vector. Caches the
        factorised matrix for efficiently solving multiple vectors"""
        if self.part_o != self.part_s:
            raise ValueError("Can only invert a self-impedance matrix")

        Z_lu = self.factored()
        if isinstance(vec, LookupArray):
            vec = vec.simple_view()

        lookup = (self.unknowns, (self.part_s, self.basis_container))

        if len(vec.shape) > 1:
            lookup = lookup+(vec.shape[1],)

        I = LookupArray(lookup, dtype=np.complex128)
        I_simp = I.simple_view()
        I_simp[:] = la.lu_solve(Z_lu, vec)
        return I

    def __getitem__(self, index):
        "Retrieve the matrix for a subset of parts"
        try:
            ind1, ind2 = index
        except:
            ind1 = index
            ind2 = self.part_s

        matrices = {key: val[ind1, ind2] for key, val in self.matrices.iteritems()}
        if self.der in (None, False):
            der = self.der
        else:
            der = {key: val[ind1, ind2] for key, val in self.der.iteritems()}

        return self.__class__(ind1, ind2, self.basis_container, self.sources,
                              self.unknowns, metadata=self.md,
                              matrices=matrices, derivatives=der)

    def __setitem__(self, index, other):
        "Set part of this matrix from another impedance matrix"
        if not isinstance(other, ImpedanceMatrixLA):
            raise ValueError("Can only set to another impedance matrix")
        for name in self.matrix_names:
            self.matrices[name][index] = other.matrices[name]
            if self.der:
                self.der[name][index] = other.der[name]

    @property
    def T(self):
        matrices = {key: val.T for key, val in self.matrices.iteritems()}

        if self.der in (None, False):
            der = self.der
        else:
            der = {key: val.T for key, val in self.der.iteritems()}

        return self.__class__(self.part_s, self.part_o, self.basis_container, 
                              self.sources, self.unknowns, metadata=self.md,
                              matrices=matrices, derivatives=der)

    def weight(self, vr, vl):
        "Weight the impedance matrix by right and left vectors"
        new_matrices = {'Z': vl.dot(self.val().dot(vr)).simple_view()}
        macro_container = vr.lookup[3][1]
        return ImpedanceMatrixLA(self.part_o, self.part_s, macro_container,
                                 ('modes',), ('modes',), self.md, new_matrices, self.der)


class EfieImpedanceMatrixLA(ImpedanceMatrixLA):
    "An impedance matrix for metallic objects solved via EFIE"

    matrix_names = ('L', 'S')

    def __init__(self, part_o, part_s, basis_container, sources, unknowns,
                 metadata=None, matrices=None, derivatives=True):
        super(EfieImpedanceMatrixLA, self).__init__(part_o, part_s,
                                                    basis_container,
                                                    sources, unknowns,
                                                    metadata, matrices, True)

    def val(self):
        "The value of the impedance matrix"
        s = self.md['s']
        Z = LookupArray((self.sources, (self.part_o, self.basis_container),
                         self.unknowns, (self.part_s, self.basis_container)),
                        dtype=np.complex128)
        Z.simple_view()[:] = self.matrices['S']/s + s*self.matrices['L']
        return Z

    def frequency_derivative(self):
        # TODO: return LookupArray
        return (self.matrices['L'] +
                self.md['s']*self.der['L'] -
                self.matrices['S']/self.md['s']**2 +
                self.der['S']/self.md['s'])

    def frequency_derivative_P(self):
        return (2*self.md['s']*self.matrices['L'] +
                self.md['s']**2*self.der['L'] +
                self.der['S'])


class CfieImpedanceMatrixLA(ImpedanceMatrixLA):
    "An impedance matrix for metallic objects solved via EFIE"

    matrix_names = ('L', 'S', 'M')

    def val(self):
        "The value of the impedance matrix"
        s = self.md['s']
        alpha = self.md['alpha']
        Z = LookupArray((self.sources, (self.part_o, self.basis_container),
                         self.unknowns, (self.part_s, self.basis_container)),
                        dtype=np.complex128)
        Z.simple_view()[:] = (alpha*(self.matrices['S']/s + s*self.matrices['L']) +
                              (1.0-alpha)*self.matrices['M'])
        return Z


class PenetrableImpedanceMatrixLA(ImpedanceMatrixLA):
    "An impedance matrix for penetrable objects"

    matrix_names = ('L_i', 'L_o', 'S_i', 'S_o', 'K_i', 'K_o')

    # TODO: D_i and K_i are stored inefficiently as full matrices, but only
    # self terms are actually needed

    def val(self):
        "The value of the impedance matrix"
        s = self.md['s']
        D_i = s*self.matrices['L_i']+self.matrices['S_i']/s
        D_o = s*self.matrices['L_o']+self.matrices['S_o']/s
        K_i = self.matrices['K_i']
        K_o = self.matrices['K_o']
        eta_o = self.md['eta_o']
        eta_i = self.md['eta_i']
        w_EFIE_i = self.md['w_EFIE_i']
        w_EFIE_o = self.md['w_EFIE_o']
        w_MFIE_i = self.md['w_MFIE_i']
        w_MFIE_o = self.md['w_MFIE_o']

        Z = LookupArray((("E", "H"), (self.part_o, self.basis_container),
                         ("J", "M"), (self.part_s, self.basis_container)),
                        dtype=np.complex128)

        # first calculate the external problem contributions
        Z["E", :, "J"] = eta_o*D_o*w_EFIE_o
        Z["E", :, "M"] = -K_o*w_EFIE_o
        Z["H", :, "J"] = K_o*w_MFIE_o
        Z["H", :, "M"] = D_o/eta_o*w_MFIE_o

        # The internal contributions are only for self-terms
        for part_o in self.part_o.iter_single():
            for part_s in self.part_s.iter_single():
                if part_o == part_s:
                    Z["E", :, "J"][part_o, part_s] += eta_i[part_s]*D_i[part_o, part_s]*w_EFIE_i[part_s]
                    Z["E", :, "M"][part_o, part_s] -= K_i[part_o, part_s]*w_EFIE_i[part_s]
                    Z["H", :, "J"][part_o, part_s] += K_i[part_o, part_s]*w_MFIE_i[part_s]
                    Z["H", :, "M"][part_o, part_s] += D_i[part_o, part_s]/eta_i[part_s]*w_MFIE_i[part_s]

        return Z


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

    def eigenmodes(self, part=None, **kwargs):
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
        return self[part, part].eigenmodes(**kwargs)
