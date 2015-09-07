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
import numpy as np

from openmodes.basis import LinearTriangleBasis
from openmodes.impedance import (EfieImpedanceMatrixLA,
                                 CfieImpedanceMatrixLA, ImpedanceMatrixLA)

from openmodes.operator.operator import Operator
from openmodes.operator import rwg
from openmodes.constants import epsilon_0, mu_0
from openmodes.array import LookupArray


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
        self.impedance_class = EfieImpedanceMatrixLA

        self.tangential_form = tangential_form
        self.unknowns = ("J",)
        if tangential_form:
            self.sources = ("E",)
            self.reciprocal = True
            self.source_cross = False
        else:
            raise NotImplementedError("n x EFIE")

        self.extinction_fields = ("E",)
        self.frequency_derivatives = True

        logging.info("Creating EFIE operator, tangential form: %s"
                     % str(tangential_form))

    def impedance_single_parts(self, Z, s, part_o, part_s):
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

        basis_o = self.basis_container[part_o]
        basis_s = self.basis_container[part_s]

        eps = self.background_material.epsilon_r(s)
        mu = self.background_material.epsilon_r(s)

        if isinstance(basis_o, LinearTriangleBasis):
            res = rwg.impedance_G(s, self.integration_rule, basis_o,
                                  part_o.nodes, basis_s, part_s.nodes,
                                  part_o == part_s, eps, mu,
                                  self.num_singular_terms,
                                  self.singularity_accuracy, True)
        else:
            raise NotImplementedError

        L, S, dL_ds, dS_ds = res

        Z.matrices['L'][part_o, part_s] = L*(mu*mu_0)
        Z.matrices['S'][part_o, part_s] = S/(eps*epsilon_0)
        Z.der['L'][part_o, part_s] = dL_ds*(mu*mu_0)
        Z.der['S'][part_o, part_s] = dS_ds/(eps*epsilon_0)

    def source_vector(self, source_field, s, parent, extinction_field):
        "Calculate the relevant source vector for this operator"

        V = LookupArray((("E"), (parent, self.basis_container)),
                        dtype=np.complex128)

        for part in parent.iter_single():
            V["E", part] = self.source_single_part(source_field, s, part,
                                                   extinction_field)

        return V

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
        self.impedance_class = ImpedanceMatrixLA

        self.tangential_form = tangential_form

        self.unknowns = ("J",)
        if tangential_form:
            self.reciprocal = False
            self.source_cross = False
            self.sources = ("H",)
        else:
            self.reciprocal = False
            self.source_cross = True
            self.sources = ("nxH",)

        self.extinction_fields = ("E",)
        self.frequency_derivatives = False

        logging.info("Creating MFIE operator, tangential form: %s"
                     % str(tangential_form))

    def impedance_single_parts(self, Z, s, part_o, part_s=None):
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
            res = rwg.impedance_curl_G(s, self.integration_rule, basis_o,
                                       part_o.nodes, basis_s, part_s.nodes,
                                       normals, part_o == part_s, eps, mu,
                                       self.num_singular_terms,
                                       self.singularity_accuracy,
                                       self.tangential_form)
        else:
            raise NotImplementedError

        Z.matrices['Z'][part_o, part_s] = res[0]
        Z.der['Z'][part_o, part_s] = res[1]


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
        self.impedance_class = CfieImpedanceMatrixLA

        self.unknowns = ("J",)
        self.sources = ("E+nxH",)

        self.extinction_fields = ("E",)
        self.frequency_derivatives = False

    def source_vector(self, source_field, s, parent, extinction_field=False):
        "Calculate the relevant source vector for this operator"

        if extinction_field:
            return super(CfieOperator, self).source_vector(source_field, s, parent, True)

        fields = ("E", "nxH")

        V = LookupArray((fields, (parent, self.basis_container)),
                        dtype=np.complex128)

        # define the functions to interpolate over the mesh
        def elec_func(r):
            return source_field.electric_field(s, r)

        def mag_func(r):
            return source_field.magnetic_field(s, r)

        for field in fields:
            if field == "E":
                field_func = elec_func
                source_cross = False
            elif field == "nxH":
                field_func = mag_func
                source_cross = True

            for part in parent.iter_single():
                basis = self.basis_container[part]
                V[field, part] = basis.weight_function(field_func, self.integration_rule,
                                                       part.nodes, source_cross)

        V_final = LookupArray((self.sources, (parent, self.basis_container)),
                              dtype=np.complex128)
        V_final[:] = self.alpha*V["E"]+(1.0-self.alpha)*V["nxH"]

        return V_final

    def impedance(self, s, parent_o, parent_s):
        metadata = {'alpha': self.alpha}
        return super(CfieOperator, self).impedance(s, parent_o, parent_s, metadata)

    def impedance_single_parts(self, Z, s, part_o, part_s=None):
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
            raise ValueError("CFIE can only be solved for closed objects")

        if isinstance(basis_o, LinearTriangleBasis):
            L, S = rwg.impedance_G(s, self.integration_rule, basis_o,
                                   part_o.nodes, basis_s, part_s.nodes,
                                   part_o == part_s, eps, mu,
                                   self.num_singular_terms,
                                   self.singularity_accuracy)

            M, _ = rwg.impedance_curl_G(s, self.integration_rule, basis_o,
                                     part_o.nodes, basis_s, part_s.nodes,
                                     normals, part_o == part_s, eps, mu,
                                     self.num_singular_terms,
                                     self.singularity_accuracy,
                                     tangential_form=False)

        else:
            raise NotImplementedError

        Z.matrices['L'][part_o, part_s] = L*(mu*mu_0)
        Z.matrices['S'][part_o, part_s] = S/(eps*epsilon_0)
        Z.matrices['M'][part_o, part_s] = M
