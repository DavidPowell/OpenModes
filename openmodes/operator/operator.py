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

from openmodes.eig import eig_linearised, eig_newton, poles_cauchy
from openmodes.array import LookupArray, view_lookuparray


class Operator(object):
    "A base class for operator equations"

    def impedance(self, s, parent_o, parent_s, frequency_derivatives=False,
                  metadata={}):
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

        symmetric = self.reciprocal and (parent_o == parent_s)

        Z = self.impedance_class(parent_o, parent_s, self.basis_container,
                                 self.sources, self.unknowns,
                                 derivatives=frequency_derivatives)

        # set the common metadata
        Z.md['s'] = s
        Z.md['symmetric'] = symmetric
        Z.md['operator'] = self
        Z.md.update(metadata)

        for count_o, part_o in enumerate(parent_o.iter_single()):
            for count_s, part_s in enumerate(parent_s.iter_single()):
                if symmetric and count_s < count_o:
                    Z[part_o, part_s] = Z[part_s, part_o].T
                else:
                    self.impedance_single_parts(Z, s, part_o, part_s, frequency_derivatives)
        return Z


    def gram_matrix(self, part):
        """Create a Gram matrix as a LookupArray"""
        G = self.basis_container[part].gram_matrix
        Gp = LookupArray((self.unknowns, (part, self.basis_container),
                          self.sources, (part, self.basis_container)),
                         dtype=G.dtype)
        Gp[:] = 0.0
        for unknown, source in zip(self.unknowns, self.sources):
            Gp[unknown, :, source, :] = G
        return Gp

    def estimate_poles(self, s_min, s_max, part, threshold=1e-11,
                       previous_result=None, cauchy_integral=True, modes=None):
        """Estimate pole location for an operator by Cauchy integration or
        the simpler quasi-static method"""

        if not cauchy_integral:
            # use the simpler quasi-static method
            Z = self.impedance(s_min, part, part)
            estimate_s, estimate_vr = eig_linearised(Z, modes)
            estimate_vr = view_lookuparray(estimate_vr,
                                           (self.unknowns, (part, self.basis_container),
                                            len(estimate_s)))
            result = {'s': estimate_s, 'vr': estimate_vr}
        else:
            def Z_func(s):
                Z = self.impedance(s, part, part, frequency_derivatives=False)
                return Z.val().simple_view()

            result = poles_cauchy(Z_func, s_min, s_max, threshold,
                                  previous_result=previous_result)
            # reformat vectors in result into LookupArrays
            for key in ('vr', 'vl', 'vl_out', 'vr_out'):
                this_result = result[key]
                if modes is not None:
                    this_result = this_result[:, modes]
                num_cols = this_result.shape[1]
                result[key] = view_lookuparray(this_result,
                                               (self.unknowns, (part, self.basis_container), num_cols))
            if modes is not None:
                result['s'] = result['s'][modes]

        result['part'] = part
        return result

    def poles(self, s_start, modes, part, use_gram=True,
              rel_tol=1e-6, max_iter=200, estimate_s=None, estimate_vr=None):
        """Find the poles of the operator applied to a specified part

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
        estimate_s : array of complex, optional
            The estimated location of the poles
        estimate_vr : array, optional
            Columns are the estimated right eigenvalues

        Returns
        -------
        mode_s : ndarray (num_modes)
            The location of the poles
        mode_j : ndarray (num_basis, num_modes)
            The current distributions at the poles
        """

        logging.info("Finding poles for part %s" % str(part.id))

        try:
            # check if a list of mode numbers was passed
            num_modes = len(modes)
        except TypeError:
            # assume that an integer was given
            num_modes = modes
            modes = range(num_modes)

        # If required, create a quasi-static estimate of the pole locations,
        # which will not find all modes
        if estimate_s is None or estimate_vr is None:
            Z = self.impedance(s_start, part, part)[part, part]
            estimate_s, estimate_vr = eig_linearised(Z, modes)
            estimate_vr = view_lookuparray(estimate_vr,
                                           (self.unknowns, (part, self.basis_container), len(estimate_s)))

        mode_s = np.empty(num_modes, np.complex128)
        mode_j = LookupArray((self.unknowns, (part, self.basis_container), num_modes),
                             dtype=np.complex128)

        # Adaptively check if the operator provides frequency derivatives, and
        # if so use them in the Newton iteration to find the poles.
        # TODO: Fix this nasty kludge
        if self.frequency_derivatives:
            def Z_func(s):
                Z = self.impedance(s, part, part, frequency_derivatives=True)
                return s*Z.val().simple_view(), Z.frequency_derivative_P().simple_view()
        else:
            def Z_func(s):
                Z = self.impedance(s, part, part, frequency_derivatives=False)
                return s*Z.val().simple_view()

        # Note that mode refers to the position in the array modes, which
        # at this point need not correspond to the original mode numbering
        for mode in range(num_modes):
            res = eig_newton(Z_func, estimate_s[mode],
                             estimate_vr[:, :, mode].simple_view(),
                             weight='max element', lambda_tol=rel_tol,
                             max_iter=max_iter,
                             func_gives_der=self.frequency_derivatives)

            logging.info("Converged after %d iterations\n"
                         "%+.4e %+.4ej (linearised solution)\n"
                         "%+.4e %+.4ej (nonlinear solution)"
                         % (res['iter_count'], estimate_s[mode].real,
                            estimate_s[mode].imag,
                            res['eigval'].real, res['eigval'].imag))

            mode_s[mode] = res['eigval']
            j_calc = res['eigvec']

            mode_j[:, :, mode] = j_calc.reshape((len(self.unknowns), -1))

        return mode_s, mode_j

    def source_vector(self, source_field, s, parent, extinction_field=False):
        "Calculate the relevant source vector for this operator"

        if extinction_field:
            fields = self.extinction_fields
        else:
            fields = self.sources

        V = LookupArray((fields, (parent, self.basis_container)),
                        dtype=np.complex128)

        # define the functions to interpolate over the mesh
        def elec_func(r):
            return source_field.electric_field(s, r)

        def mag_func(r):
            return source_field.magnetic_field(s, r)

        for field in fields:
            if field in ("E", "nxE"):
                field_func = elec_func
                source_cross = field == "nxE"
            elif field in ("H", "nxH"):
                field_func = mag_func
                source_cross = field == "nxH"
            else:
                raise ValueError(field)

            for part in parent.iter_single():
                basis = self.basis_container[part]
                V[field, part] = basis.weight_function(field_func, self.integration_rule,
                                                       part.nodes, source_cross)

        return V
