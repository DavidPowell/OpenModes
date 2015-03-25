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


from openmodes.basis import LinearTriangleBasis
from openmodes.impedance import SimpleImpedanceMatrix
import logging

from openmodes.operator.operator import Operator
from openmodes.operator import rwg


class MfieOperator(Operator):
    """An operator for the magnetic field integral equation, discretised with
    respect to some set of basis functions. Assumes that Galerkin's method is
    used, such that the testing functions are the same as the basis functions.
    """
    source_field = "magnetic_field"

    def __init__(self, integration_rule, basis_container,
                 tangential_form=False, num_singular_terms=2,
                 singularity_accuracy=1e-5):
        """
        Parameters
        ----------
        integration_rule: object
        The integration rule over the standard triangle, to be used for all
        non-singular integrals
        basis_container: BasisContainer
        The object which retrieves basis functions for a Part
        tangential_form: boolean
        If True, -n x n x K is solved, otherwise n x K form is used
        """
        self.basis_container = basis_container
        self.integration_rule = integration_rule
        self.num_singular_terms = num_singular_terms
        self.singularity_accuracy = singularity_accuracy

        self.tangential_form = tangential_form
        if tangential_form:
            self.reciprocal = False
            self.source_cross = False
        else:
            self.reciprocal = False
            self.source_cross = True

        logging.info("Creating MFIE operator, tangential form: %s"
                     % str(tangential_form))

    def source_single_part(self, source_field, s, part, extinction_field):
        if extinction_field:
            field = lambda r: source_field.electric_field(s, r)
            source_cross = False
        else:
            field = lambda r: source_field.magnetic_field(s, r)
            source_cross = self.source_cross
        basis = self.basis_container[part]
        return basis.weight_function(field, self.integration_rule,
                                     part.nodes, source_cross)

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

        normals = basis_o.mesh.surface_normals

        if isinstance(basis_o, LinearTriangleBasis):
            Z = rwg.impedance_curl_G(s, self.integration_rule, basis_o,
                                     part_o.nodes, basis_s, part_s.nodes,
                                     normals, part_o == part_s,
                                     self.num_singular_terms,
                                     self.singularity_accuracy,
                                     self.tangential_form)
        else:
            raise NotImplementedError

        return SimpleImpedanceMatrix.build(s, Z, basis_o, basis_s, self,
                                           part_o, part_s, symmetric=False)


class TMfieOperator(MfieOperator):
    def __init__(self, **kwargs):
        MfieOperator.__init__(self, tangential_form=True, **kwargs)
