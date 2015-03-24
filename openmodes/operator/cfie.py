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

from openmodes.operator.operator import Operator
from openmodes.operator.efie import EfieOperator
from openmodes.operator.mfie import MfieOperator
from openmodes.impedance import CfieImpedanceMatrix


class CfieOperator(Operator):
    "Combined field integral equation for PEC objects"
    reciprocal = False

    def __init__(self, integration_rule, basis_container, alpha=0.5,
                 num_singular_terms=2):
        """
        Parameters
        ----------
        integration_rule: object
            The integration rule over the standard triangle, to be used for all
            non-singular integrals
        basis_container: BasisContainer
            The object which retrieves basis functions for a Part
        alpha : real
            The relative weighting of EFIE vs MFIE, 0 < alpha < 1
        """

        self.basis_container = basis_container
        self.integration_rule = integration_rule
        self.num_singular_terms = num_singular_terms
        self.alpha = alpha

        self.efie = EfieOperator(integration_rule, basis_container,
                                 tangential_form=True,
                                 num_singular_terms=num_singular_terms)
        self.mfie = MfieOperator(integration_rule, basis_container,
                                 tangential_form=False,
                                 num_singular_terms=num_singular_terms)

    def source_single_part(self, source_field, s, part, extinction_field):
        basis = self.basis_container[part]
        E_field = lambda r: source_field.electric_field(s, r)
        V_E = basis.weight_function(E_field, self.integration_rule,
                                    part.nodes, False)
        if extinction_field:
            return V_E

        H_field = lambda r: source_field.magnetic_field(s, r)
        V_H = basis.weight_function(H_field, self.integration_rule,
                                    part.nodes, True)

        return self.alpha*V_E+(1-self.alpha)*V_H

    def impedance_single_parts(self, s, part_o, part_s=None):
        """Calculate a self or mutual impedance matrix at a given complex
        frequency

        Parameters
        ----------
        s : complex
            Complex frequency at which to calculate impedance
        part_o : SinglePart
            The observing part, which must be a single part, not a composite
        part_s : SinglePart, optional
            The source part, if not specified will default to observing part
        """

        # if source part is not given, default to observer part
        part_s = part_s or part_o

        basis_o = self.basis_container[part_o]
        basis_s = self.basis_container[part_s]

        if not (basis_o.mesh.closed_surface and basis_s.mesh.closed_surface):
            raise ValueError("MFIE can only be solved for closed objects")

        Z_m = self.mfie.impedance_single_parts(s, part_o, part_s)
        Z_e = self.efie.impedance_single_parts(s, part_o, part_s)

        return CfieImpedanceMatrix.build(s, Z_e.matrices['L'],
                                         Z_e.matrices['S'], Z_m.matrices['Z'],
                                         self.alpha, basis_o, basis_s, self,
                                         part_o, part_s, False)
