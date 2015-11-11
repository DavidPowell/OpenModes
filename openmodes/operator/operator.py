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

from openmodes.eig import (eig_linearised, eig_newton, poles_cauchy,
                           ConvergenceError)
from openmodes.array import LookupArray


class Operator(object):
    "A base class for operator equations"

    def impedance(self, s, parent_o, parent_s,  metadata=None):
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

        metadata = metadata or dict()
        symmetric = self.reciprocal and (parent_o == parent_s)

        Z = self.impedance_class(parent_o, parent_s, self.basis_container,
                                 self.sources, self.unknowns)

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
                    self.impedance_single_parts(Z, s, part_o, part_s)
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

    def estimate_poles(self, contour, part, threshold=1e-11,
                       previous_result=None, cauchy_integral=True, modes=None,
                       **kwargs):
        """Estimate pole location for an operator by Cauchy integration or
        the simpler quasi-static method"""

        if not cauchy_integral:
            # Use the simpler quasi-static method (contour will actually
            # just be a starting frequency)
            Z = self.impedance(contour, part, part)
            estimate_s, estimate_vr = eig_linearised(Z, modes)
            result = {'s': estimate_s, 'vr': estimate_vr, 'vl': estimate_vr}
        else:
            def Z_func(s):
                Z = self.impedance(s, part, part)
                return Z.val().simple_view()

            result = poles_cauchy(Z_func, contour, threshold,
                                  previous_result=previous_result, **kwargs)

        return result

    def refine_poles(self, estimates, part, rel_tol, max_iter):
        """Find the poles of the operator applied to a specified part

        Parameters
        ----------
        estimates : dictionary
            The data for the estimated poles
        rel_tol : float
            The relative tolerance on the search for singularities
        max_iter : integer
            The maximum number of iterations to use when searching for
            singularities

        Returns
        -------
        refined : dictionary
            The refined poles
        """

        #part = estimates['part']
        logging.info("Finding poles for part %s" % str(part.id))

        num_modes = len(estimates['s'])

        refined = {'s': []}
        refined['vr'] = []
        refined['vl'] = []

        # Adaptively check if the operator provides frequency derivatives, and
        # if so use them in the Newton iteration to find the poles.
        if self.frequency_derivatives:
            logging.info("Using exact impedance derivatives")
            def Z_func(s):
                Z = self.impedance(s, part, part)
                return Z.val().simple_view(), Z.frequency_derivative().simple_view()
        else:
            logging.info("Using approximate impedance derivatives")
            def Z_func(s):
                Z = self.impedance(s, part, part)
                return Z.val().simple_view()

        symmetric = self.reciprocal

        # weight_type = 'max element'
        if symmetric:
            weight_type = 'rayleigh symmetric'
        else:
            weight_type = 'rayleigh asymmetric'

        # Note that mode refers to the position in the array modes, which
        # at this point need not correspond to the original mode numbering
        for mode in range(num_modes):
            logging.info("Searching for mode %d"%mode)
            try:
                res = eig_newton(Z_func, estimates['s'][mode],
                                 estimates['vr'][:, mode],
                                 weight=weight_type, lambda_tol=rel_tol,
                                 max_iter=max_iter,
                                 func_gives_der=self.frequency_derivatives,
                                 y_0=estimates['vl'][mode, :])
            except (ConvergenceError, ValueError):
                logging.info("Convergence failed, mode discarded")
                continue

            logging.info("Converged after %d iterations\n"
                         "%+.4e %+.4ej (linearised solution)\n"
                         "%+.4e %+.4ej (nonlinear solution)"
                         % (res['iter_count'], estimates['s'][mode].real,
                            estimates['s'][mode].imag,
                            res['eigval'].real, res['eigval'].imag))

            refined['s'].append(res['eigval'])
            refined['vr'].append(res['eigvec'])
            refined['vl'].append(res['eigvec_left'])

        # convert lists to arrays
        refined['s'] = np.array(refined['s'])
        refined['vr'] = np.array(refined['vr']).T
        refined['vl'] = np.array(refined['vl'])
        return refined

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
