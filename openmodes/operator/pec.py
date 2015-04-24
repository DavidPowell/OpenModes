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
"Operators for PEC scatterers"

import logging

from openmodes.basis import LinearTriangleBasis, LoopStarBasis
from openmodes.impedance import (EfieImpedanceMatrix,
                                 EfieImpedanceMatrixLoopStar,
                                 CfieImpedanceMatrix, SimpleImpedanceMatrix)

from openmodes.operator.operator import Operator
from openmodes.operator import rwg
from openmodes.constants import epsilon_0, mu_0


class EfieOperator(Operator):
    """An operator for the electric field integral equation, discretised with
    respect to some set of basis functions. Assumes that Galerkin's method is
    used, such that the testing functions are the same as the basis functions.
    """

    def __init__(self, integration_rule, basis_container, background_material,
                 tangential_form=True, num_singular_terms=2,
                 singularity_accuracy=1e-5):
        self.basis_container = basis_container
        self.background_material = background_material
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

    def impedance_single_parts(self, s, part_o, part_s=None,
                               frequency_derivatives=False):
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

        eps = self.background_material.epsilon_r(s)
        mu = self.background_material.epsilon_r(s)

        if isinstance(basis_o, LinearTriangleBasis):
            res = rwg.impedance_G(s, self.integration_rule, basis_o,
                                  part_o.nodes, basis_s, part_s.nodes,
                                  part_o == part_s, eps, mu,
                                  self.num_singular_terms,
                                  self.singularity_accuracy,
                                  frequency_derivatives)
        else:
            raise NotImplementedError

        if frequency_derivatives:
            L, S, dL_ds, dS_ds = res
        else:
            L, S = res

        L *= mu*mu_0
        S /= eps*epsilon_0

        matrices = {'L': L, 'S': S}

        if frequency_derivatives:
            dL_ds *= mu*mu_0
            dS_ds /= eps*epsilon_0

            matrices['dL_ds'] = dL_ds
            matrices['dS_ds'] = dS_ds

        metadata = {'basis_o': basis_o, 'basis_s': basis_s, 's': s,
                    'operator': self, 'part_o': part_o, 'part_s': part_s,
                    'symmetric': symmetric}

        if issubclass(self.basis_container.basis_class, LoopStarBasis):
            return EfieImpedanceMatrixLoopStar(matrices, metadata)
        else:
            return EfieImpedanceMatrix(matrices, metadata)

    def source_single_part(self, source_field, s, part,
                           extinction_field):
        "Since the EFIE is symmetric, extinction_field is the same field"
        field = lambda r: source_field.electric_field(s, r)
        basis = self.basis_container[part]
        return basis.weight_function(field, self.integration_rule,
                                     part.nodes, self.source_cross)


class MfieOperator(Operator):
    """An operator for the magnetic field integral equation, discretised with
    respect to some set of basis functions. Assumes that Galerkin's method is
    used, such that the testing functions are the same as the basis functions.
    """
    source_field = "magnetic_field"

    def __init__(self, integration_rule, basis_container, background_material,
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
        self.background_material = background_material
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

        eps = self.background_material.epsilon_r(s)
        mu = self.background_material.epsilon_r(s)

        if not (basis_o.mesh.closed_surface and basis_s.mesh.closed_surface):
            raise ValueError("MFIE can only be solved for closed objects")

        normals = basis_o.mesh.surface_normals

        if isinstance(basis_o, LinearTriangleBasis):
            Z = rwg.impedance_curl_G(s, self.integration_rule, basis_o,
                                     part_o.nodes, basis_s, part_s.nodes,
                                     normals, part_o == part_s, eps, mu,
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


class CfieOperator(Operator):
    "Combined field integral equation for PEC objects"
    reciprocal = False

    def __init__(self, integration_rule, basis_container, background_material,
                 alpha=0.5, num_singular_terms=2, singularity_accuracy=1e-5):
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
        self.background_material = background_material
        self.integration_rule = integration_rule
        self.num_singular_terms = num_singular_terms
        self.singularity_accuracy = singularity_accuracy
        self.alpha = alpha

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

        eps = self.background_material.epsilon_r(s)
        mu = self.background_material.epsilon_r(s)

        normals = basis_o.mesh.surface_normals

        if not (basis_o.mesh.closed_surface and basis_s.mesh.closed_surface):
            raise ValueError("MFIE can only be solved for closed objects")

        if isinstance(basis_o, LinearTriangleBasis):
            L, S = rwg.impedance_G(s, self.integration_rule, basis_o,
                                   part_o.nodes, basis_s, part_s.nodes,
                                   part_o == part_s, eps, mu,
                                   self.num_singular_terms,
                                   self.singularity_accuracy)
            L *= mu*mu_0
            S /= eps*epsilon_0

            M = rwg.impedance_curl_G(s, self.integration_rule, basis_o,
                                     part_o.nodes, basis_s, part_s.nodes,
                                     normals, part_o == part_s, eps, mu,
                                     self.num_singular_terms,
                                     self.singularity_accuracy,
                                     tangential_form=False)

        else:
            raise NotImplementedError

        return CfieImpedanceMatrix.build(s, L, S, M, self.alpha, basis_o,
                                         basis_s, self,  part_o, part_s, False)
