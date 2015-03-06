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


import numpy as np
import logging

from openmodes.constants import epsilon_0, mu_0, c
from openmodes.vector import VectorParts
from openmodes.eig import eig_linearised, eig_newton

from openmodes.impedance import ImpedanceParts


class FreeSpaceGreensFunction(object):
    "Green's function in a homogeneous isotropic medium such as free space"
    def __init__(self, epsilon_r=1, mu_r=1):
        """
        Parameters
        ---------
        epsilon_r, mu_r : real
            Relative permittivity and permeability of the background medium

        A lossy background medium is currently unsupported
        """
        self.epsilon = epsilon_r*epsilon_0
        self.mu = mu_r*mu_0
        self.c = c/np.sqrt(mu_r*epsilon_r)
        self.eta = np.sqrt(self.mu/self.epsilon)


class Operator(object):
    "A base class for operator equations"

    def impedance(self, s, parent_o, parent_s):
        """Evaluate the self and mutual impedances of all parts in the
        simulation. Return an `ImpedancePart` object which can calculate
        several derived impedance quantities

        Parameters
        ----------
        s : number
            Complex frequency at which to calculate impedance (in rad/s)
        parent : Part
            Only this part and its sub-parts will be calculated

        Returns
        -------
        impedance_matrices : ImpedanceParts
            The impedance matrix object which can represent the impedance of
            the object in several ways.
        """

        matrices = {}

        # TODO: cache individual part impedances to avoid repetition?
        # May not be worth it because mutual impedances cannot be cached
        # except in specific cases such as arrays, and self terms may be
        # invalidated by green's functions which depend on coordinates

        for part_o in parent_o.iter_single():
            for part_s in parent_s.iter_single():
                if self.reciprocal and (part_s, part_o) in matrices:
                    # use reciprocity to avoid repeated calculation
                    res = matrices[part_s, part_o].T
                else:
                    res = self.impedance_single_parts(s, part_o, part_s)
                matrices[part_o, part_s] = res

        return ImpedanceParts(s, parent_o, parent_s, matrices, type(res))

    def singularities(self, s_start, modes, part, use_gram=True,
                      rel_tol=1e-6, max_iter=200):
        """Find the singularities of a part or of the whole system

        Parameters
        ----------
        s_start : complex
            The complex frequency at which to perform the estimate. Should be
            within the band of interest
        num_modes : integer or list
            An integer specifying the number of modes to find, or a list of
            mode numbers to find
        part : Part
            The part to solve for
        use_gram : boolean, optional
            Use the Gram matrix to scale the eigenvectors, so that the
            eigenvalues will be independent of the basis functions.
        rel_tol : float, optional
            The relative tolerance on the search for singularities
        max_iter : integer, optional
            The maximum number of iterations to use when searching for
            singularities

        Returns
        -------
        mode_s : ndarray (num_modes)
            The location of the singularities
        mode_j : ndarray (num_basis, num_modes)
            The current distributions at the singularities
        """

        logging.info("Finding singularities for part %s" % str(part.id))

        try:
            # check if a list of mode numbers was passed
            num_modes = len(modes)
        except TypeError:
            # assume that an integer was given
            num_modes = modes
            modes = range(num_modes)

        # first get an estimate of the pole locations
        Z = self.impedance(s_start, part, part)[part, part]
        lin_s, lin_currents = eig_linearised(Z, modes)

        mode_s = np.empty(num_modes, np.complex128)
        mode_j = VectorParts(part, self.basis_container, dtype=np.complex128,
                             cols=num_modes)

        Z_func = lambda s: self.impedance(s, part, part)[part, part][:]

        if use_gram:
            G = Z.basis_s.gram_matrix

        # Note that mode refers to the position in the array modes, which
        # at this point need not correspond to the original mode numbering
        for mode in range(num_modes):
            res = eig_newton(Z_func, lin_s[mode], lin_currents[:, mode],
                             weight='max element', lambda_tol=rel_tol,
                             max_iter=max_iter)

            lin_hz = lin_s[mode]/2/np.pi
            nl_hz = res['eigval']/2/np.pi
            logging.info("Converged after %d iterations\n"
                         "%+.4e %+.4ej (linearised solution)\n"
                         "%+.4e %+.4ej (nonlinear solution)"
                         % (res['iter_count'], lin_hz.real, lin_hz.imag,
                            nl_hz.real, nl_hz.imag))

            mode_s[mode] = res['eigval']
            j_calc = res['eigvec']

            if use_gram:
                j_calc /= np.sqrt(j_calc.T.dot(G.dot(j_calc)))
            else:
                j_calc /= np.sqrt(np.sum(j_calc**2))

            mode_j[:, mode] = j_calc

        return mode_s, mode_j
