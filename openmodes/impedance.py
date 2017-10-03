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

from openmodes.array import LookupArray


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

        # Note that the internal LookupArray format is different from the
        # final format as it excludes the quantity lookup.
        self.matrices = {name: LookupArray(((part_o, basis_container),
                                            (part_s, basis_container)),
                                           dtype=np.complex128)
                         for name in self.matrix_names}

        if matrices is not None:
            # fill out any matrices which are supplied
            for name, mat in matrices.items():
                self.matrices[name][:] = mat

        # create the frequency derivatives of the matrices
        if derivatives is None:
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

    def frequency_derivative(self):
        # TODO: return LookupArray?
        return self.der['Z']

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

        matrices = {key: val[ind1, ind2] for key, val in self.matrices.items()}
        if self.der in (None, False):
            der = self.der
        else:
            der = {key: val[ind1, ind2] for key, val in self.der.items()}

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
        matrices = {key: val.T for key, val in self.matrices.items()}

        if self.der in (None, False):
            der = self.der
        else:
            der = {key: val.T for key, val in self.der.items()}

        return self.__class__(self.part_s, self.part_o, self.basis_container,
                              self.sources, self.unknowns, metadata=self.md,
                              matrices=matrices, derivatives=der)

    def weight(self, vr, vl):
        "Weight the impedance matrix by right and left vectors"
        new_matrices = {name: np.dot(vl.simple_view(), np.dot(mat, vr.simple_view()))
                        for name, mat in self.matrices.items()}
        new_der = {name: np.dot(vl.simple_view(), np.dot(mat, vr.simple_view()))
                   for name, mat in self.der.items()}
        macro_container = vr.lookup[3][1]
        return self.__class__(self.part_o, self.part_s, macro_container,
                              ('modes',), ('modes',), self.md, new_matrices,
                              new_der)


class EfieImpedanceMatrixLA(ImpedanceMatrixLA):
    "An impedance matrix for metallic objects solved via EFIE"

    matrix_names = ('L', 'S')

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
