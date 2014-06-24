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
"Fit scalar models to numerically calculated impedance data"

import logging
import numpy as np
from scipy.optimize import nnls
import scipy.linalg as la


def delta_eig(s, j, Z_func, eps=None):
    """Find the derivative of the eigenimpedance at the resonant frequency

    See section 5.7 of numerical recipes for calculating the step size h

    Impedance derivative is based on
    C. E. Baum, Proceedings of the IEEE 64, 1598 (1976).
    """

    if eps is None:
        # find the machine precision (this should actually be the accuracy with
        # which Z is calculated)
        eps = np.finfo(s.dtype).eps

    # first determine the optimal value of h
    h = abs(s)*eps**(1.0/3.0)*(1.0 + 1.0j)

    # make h exactly representable in floating point
    temp = s + h
    h = (temp - s)

    delta_Z = (Z_func(s+h)[:] - Z_func(s-h)[:])/(2*h)

    return np.dot(j.T, np.dot(delta_Z, j))


def fit_four_term(s_0, z_der):
    """
    Fit a 4 term model to a resonant frequency and impedance derivative
    To get reasonable condition number, omega_0 should be scaled to be near
    unity, and z_der should be scaled by the inverse of this factor
    """
    M = np.zeros((4, 4), np.float64)
    rhs = np.zeros(4, np.float64)

    # order of coefficients is C, R, L, R2

    # fit impedance being zero at resonance
    eq1 = np.array([1/s_0, 1, s_0, -s_0**2])
    M[0, :] = eq1.real
    M[1, :] = eq1.imag

    # fit impedance derivative at resonance
    eq2 = np.array([-1/s_0**2, 0, 1, -2*s_0])
    M[2, :] = eq2.real
    M[3, :] = eq2.imag

    rhs[2] = z_der.real
    rhs[3] = z_der.imag

    logging.debug("Fitting 4 term polynomial\nM = %s\nrhs = %s" %
                  (str(M), str(rhs)))

    return nnls(M, rhs)[0]


class ScalarModel(object):
    """A scalar model of a mode of a structure, assuming that the eigencurrents
    are frequency independent. Fits a 4th order model to the eigenfrequency
    and the derivative of the eigenimpedancec at resonance, as well as the
    condition of open-circuit impedance at zero frequency."""

    def __init__(self, part, mode_s, mode_j, operator):
        "Construct the scalar model"
        self.mode_s = mode_s
        self.mode_j = mode_j

        self.scale_factor = abs(mode_s.imag)/10

        Z_func = lambda s: operator.impedance(s, part, part)[part, part]
        z_der = delta_eig(mode_s, mode_j, Z_func)
        self.coefficients = fit_four_term(mode_s/self.scale_factor,
                                          z_der*self.scale_factor)
        logging.info("Creating scalar model\ndlambda/ds = %+.4e %+.4e\n"
                     "Coefficients: %s" % (z_der.real, z_der.imag,
                                           str(self.coefficients)))

    def scalar_impedance(self, s):
        "The scalar impedance of this mode"
        s = s/self.scale_factor
        powers = np.array([1/s, 1, s, -s**2])
        return np.dot(self.coefficients, powers.T)

    def solve(self, s, V):
        "Solve the model for the current at arbitrary frequency"
        return self.mode_j*np.dot(self.mode_j, V[:])/self.scalar_impedance(s)


def fit_LS(s_0, L_0, S_0):
    """
    Fit a polynomial model to the values of the scalar impedance components
    at the resonant frequency.

    Parameters
    ----------
    s_0 : complex
        The resonant frequency
    L : complex
        The scalar inductance at resonance
    S : complex
        The scalar susceptance at resonance

    Potentially extendable to include the second derivative (first derivative
    gives no extra information?)
    """

    M = np.zeros((2, 2))
    eq = np.array([1.0, -s_0])
    M[:, 0] = eq.real
    M[:, 1] = eq.imag

    L_coeffs = nnls(M, np.array([L_0.real, L_0.imag]))[0]

    eq = np.array([1.0, s_0])
    M[:, 0] = eq.real
    M[:, 1] = eq.imag

    S_coeffs = nnls(M, np.array([S_0.real, S_0.imag]))[0]

    return L_coeffs, S_coeffs


class ScalarModelLS(object):
    """A scalar model of a mode of a structure, assuming that the eigencurrents
    are frequency independent. Fits to the diagonalised partial impedance
    matrices L and S at resonance."""

    def __init__(self, mode_s, mode_j, Z_func):
        "Construct the scalar model"
        self.mode_s = mode_s
        self.mode_j = mode_j
        Z = Z_func(mode_s)
        self.L_scale = 1e10
        self.S_scale = 1e-10
        L_0 = mode_j.dot(Z.L.dot(mode_j))
        S_0 = mode_j.dot(Z.S.dot(mode_j))
        self.scale_factor = abs(mode_s.imag)/10
        self.L, self.S = fit_LS(mode_s/self.scale_factor, L_0*self.L_scale,
                                S_0*self.S_scale)

        logging.info("Creating scalar model\nL(s_0) = %+.4e %+.4e\n"
                     "S(s_0) = %+.4e %+.4e\nL Coefficients: %s\n"
                     "S Coefficients: %s" %
                     (L_0.real, L_0.imag, S_0.real, S_0.imag, str(self.L),
                      str(self.S)))

    def scalar_impedance(self, s):
        "The scalar impedance of this mode"
        powers_L = np.array([1.0, -s/self.scale_factor])
        powers_S = np.array([1.0, s/self.scale_factor])
        return (s*np.dot(self.L, powers_L.T)/self.L_scale +
                np.dot(self.S, powers_S.T)/s/self.S_scale)

    def solve(self, s, V):
        "Solve the model for the current at arbitrary frequency"
        return self.mode_j*np.dot(self.mode_j, V)/self.scalar_impedance(s)


class MutualPolyModel(object):
    """A model for mutual impedance between parts with multiple modes
    """
    def __init__(self, part_o, current_o, part_s, current_s, operator,
                 poly_order, s_max):
        """
        Create the model for mutual terms

        Parameters
        ----------
        part_o, part_s: Part
            The observing and source parts
        current_o, current_s: ndarray(num_basis, num_modes)
            The current distribution of the modes
        operator: Operator
            The operator describing the equations
        poly_order: integer
            The order of polynomial to use for mutual terms
        s_max: complex
            The highest frequency to calculate expansion for
        """
        self.part_o = part_o
        self.current_o = current_o
        self.part_s = part_s
        self.current_s = current_s
        self.operator = operator

        self.scale_factor = s_max.imag/20
        self.poly_order = poly_order

        self.num_modes_o = current_o.shape[1]
        self.num_modes_s = current_s.shape[1]

        # number of points is order divided by two, rounded up
        num_s = sum(divmod(poly_order, 2))
        s_range = np.linspace(s_max, 0, num_s, endpoint=False)/self.scale_factor
        S_data = np.empty((num_s, self.num_modes_o, self.num_modes_s),
                          np.complex128)
        L_data = np.empty_like(S_data)

        # calculate the data at each frequency
        for s_count, s in enumerate(s_range):
            L_data[s_count], S_data[s_count] = self.LS_direct(s*self.scale_factor)

        logging.info("Fitting mutual term model at frequencies %s" %
                     str(s_range*self.scale_factor))

        orders = np.arange(self.poly_order)
        self.models = []
        # For each mode, perform the polynomial fit. Note that this is a
        # fairly ill-conditioned process, so only low-order should be used.
        for mode_o in xrange(self.num_modes_o):
            model_row = []
            for mode_s in xrange(self.num_modes_s):
                matrix = s_range[:, None]**orders[None, :]
                matrix = np.vstack((matrix.real, matrix.imag))

                # When solving, specify the number of rows, so that if there
                # is an addition row of imaginary data, it is tropped
                rhs_S = S_data[:, mode_o, mode_s]
                rhs_S = np.hstack((rhs_S.real, rhs_S.imag))
                S_coeffs = la.solve(matrix[:poly_order], rhs_S[:poly_order])

                rhs_L = L_data[:, mode_o, mode_s]
                rhs_L = np.hstack((rhs_L.real, rhs_L.imag))
                L_coeffs = la.solve(matrix[:poly_order], rhs_L[:poly_order])

                logging.debug("Matrix data:\n%s\nrhs for L %s\nrhs for S %s" %
                              (str(matrix), str(rhs_L), str(rhs_S)))
                logging.info("L coefficients: %s\n S coefficients: %s" %
                             (str(L_coeffs), str(S_coeffs)))

                model_row.append((L_coeffs, S_coeffs))
            self.models.append(model_row)

    def LS(self, s, mode_o, mode_s):
        s /= self.scale_factor
        orders = np.arange(self.poly_order)
        L_coeffs, S_coeffs = self.models[mode_o][mode_s]
        return np.dot(L_coeffs, s**orders), np.dot(S_coeffs, s**orders)

    def block_impedance(self, s):
        "Calculate the impedance block matrix at the specified frequency"
        matrix = np.empty((self.num_modes_o, self.num_modes_s), np.complex128)
        for mode_o in xrange(self.num_modes_o):
            for mode_s in xrange(self.num_modes_s):
                L, S = self.LS(s, mode_o, mode_s)
                matrix[mode_o, mode_s] = s*L + S/s
        return matrix

    def LS_direct(self, s):
        """Calculate the impedance block matrices L and S at the specified
        frequency, directly by calculating and weighting the impedance matrix
        """
        Z = self.operator.impedance(s, self.part_o, self.part_s)[self.part_o,
                                                                 self.part_s]
        return (self.current_o.T.dot(Z.L.dot(self.current_s)),
                self.current_o.T.dot(Z.S.dot(self.current_s)))


class SelfModel(object):
    "A model for the self-impedance of a part with multiple modes"
    def __init__(self, part, mode_s, mode_j, operator):
        self.num_modes = len(mode_s)
        self.models = []
        for mode_num, s in enumerate(mode_s):
            self.models.append(ScalarModel(part, s, mode_j[:, mode_num],
                               operator))

    def block_impedance(self, s):
        "Calculate the impedance block matrix at the specified frequency"
        Z = np.zeros((self.num_modes, self.num_modes), np.complex128)
        for mode_count, model in enumerate(self.models):
            Z[mode_count, mode_count] = model.scalar_impedance(s)
        return Z


class ModelPolyInteraction(object):
    """A model for a system of several parts.

    Self-impedance terms are modelled as 4-term scalar impedances. Mutual terms
    are modelled by weighting L and S with the modes, and fitting with a
    polynomial.
    """
    def __init__(self, operator, parts_modes, poly_order, s_max):
        """
        Construct a model for a set of parts

        Parameters
        ----------
        operator: Operator
            The operator for the system of equations
        parts_modes: list of tuple of (Part, eigenfreq, eigencurrent)
            These parts will form the basis of the model
        poly_order: integer
            The order of polynomial to use for mutual terms
        s_max: complex
            The highest frequency to calculate expansion for
        """

        self.operator = operator
        self.models = {}

        self.num_rows = 0
        self.parts = []

        row = 0
        for part, mode_s, mode_j in parts_modes:
            num_rows = len(mode_s)
            self.parts.append((part, slice(row, row+num_rows)))
            row += num_rows
        self.num_rows = row

        for count_o, (part_o, mode_s_o, current_o) in enumerate(parts_modes):
            for count_s, (part_s, mode_s_s, current_s) in enumerate(parts_modes):
                if count_o == count_s:
                    # within the same part
                    self.models[part_o, part_s] = SelfModel(part_o, mode_s_o,
                                                            current_o,
                                                            operator)
                elif (not operator.reciprocal) or (count_o < count_s):
                    # between different parts
                    self.models[part_o, part_s] = MutualPolyModel(part_o,
                                                                  current_o,
                                                                  part_s,
                                                                  current_s,
                                                                  operator,
                                                                  poly_order,
                                                                  s_max)

    def impedance(self, s):
        """Calculate the reduced impedance matrix at a given frequency

        Parameters
        ----------
        s: complex
            Frequency at which to calculate
        """
        matrix = np.empty((self.num_rows, self.num_rows), np.complex128)

        for count_o, (part_o, slice_o) in enumerate(self.parts):
            for count_s, (part_s, slice_s) in enumerate(self.parts):
                if self.operator.reciprocal and count_o > count_s:
                    matrix[slice_o, slice_s] = matrix[slice_s, slice_o].T
                else:
                    matrix[slice_o, slice_s] = self.models[part_o, part_s].block_impedance(s)
        return matrix

    def solve(self, s, V):
        "Solve the model for a particular incident field"
        Z = self.impedance(s)
        return la.solve(Z, V)
