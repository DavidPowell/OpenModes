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


import logging

from openmodes.basis import LinearTriangleBasis, LoopStarBasis
from openmodes.impedance import (EfieImpedanceMatrix,
                                 EfieImpedanceMatrixLoopStar)

from openmodes.operator.operator import Operator
from openmodes.operator import rwg


class EfieOperator(Operator):
    """An operator for the electric field integral equation, discretised with
    respect to some set of basis functions. Assumes that Galerkin's method is
    used, such that the testing functions are the same as the basis functions.
    """

    def __init__(self, integration_rule, basis_container,
                 tangential_form=True, num_singular_terms=2,
                 singularity_accuracy=1e-5):
        self.basis_container = basis_container
        self.integration_rule = integration_rule
        self.num_singular_terms = num_singular_terms
        self.singularity_accuracy = singularity_accuracy

        self.tangential_form = tangential_form
        if tangential_form:
            self.reciprocal = True
            self.source_cross = False
        else:
            raise NotImplementedError("n x EFIE")

        logging.info("Creating EFIE operator, tangential form: %s"
                     % str(tangential_form))

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
        if part_s is None:
            part_s = part_o
            symmetric = self.reciprocal
        else:
            symmetric = False

        basis_o = self.basis_container[part_o]
        basis_s = self.basis_container[part_s]

        if isinstance(basis_o, LinearTriangleBasis):
            L, S = rwg.impedance_G(s, self.integration_rule, basis_o,
                                   part_o.nodes, basis_s, part_s.nodes,
                                   part_o == part_s, self.num_singular_terms,
                                   self.singularity_accuracy)
        else:
            raise NotImplementedError

        if issubclass(self.basis_container.basis_class, LoopStarBasis):
            return EfieImpedanceMatrixLoopStar.build(s, L, S, basis_o, basis_s,
                                                     self, part_o, part_s,
                                                     symmetric)
        else:
            return EfieImpedanceMatrix.build(s, L, S, basis_o, basis_s, self,
                                             part_o, part_s, symmetric)

    def source_single_part(self, source_field, s, part,
                           extinction_field):
        "Since the EFIE is symmetric, extinction_field is the same field"
        field = lambda r: source_field.electric_field(s, r)
        basis = self.basis_container[part]
        return basis.weight_function(field, self.integration_rule,
                                     part.nodes, self.source_cross)
