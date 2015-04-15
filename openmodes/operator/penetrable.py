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
"""Operators for surface equivalent problems for penetrable scatteres"""

from __future__ import division

import numpy as np

from openmodes.basis import LinearTriangleBasis
from openmodes.impedance import PenetrableImpedanceMatrix
from openmodes.operator.operator import Operator
from openmodes.operator import rwg
from openmodes.constants import epsilon_0, mu_0, c


class TOperator(Operator):
    """General tangential-form operator for penetrable objects

    Note that this class is designed as an abstract base, so it should not
    be created directly.
    """
    reciprocal = False

    def __init__(self, integration_rule, basis_container,
                 background_material, num_singular_terms=2,
                 singularity_accuracy=1e-5):
        """
        Parameters
        ----------
        integration_rule: object
            The integration rule over the standard triangle, to be used for all
            non-singular integrals
        basis_container: BasisContainer
            The object which retrieves basis functions for a Part
        eps_i, mu_i, ep_o, mu_o : scalar or function
            The permittivity and permeability of the inner and outer regions.
            These may be constants, or analytic functions of frequency 's'
        """

        self.basis_container = basis_container
        self.integration_rule = integration_rule
        self.num_singular_terms = num_singular_terms
        self.background_material = background_material
        self.singularity_accuracy = singularity_accuracy

    def source_single_part(self, source_field, s, part, extinction_field):
        basis = self.basis_container[part]
        E_field = lambda r: source_field.electric_field(s, r)
        V_E = basis.weight_function(E_field, self.integration_rule,
                                    part.nodes, False)

        H_field = lambda r: source_field.magnetic_field(s, r)
        V_H = basis.weight_function(H_field, self.integration_rule,
                                    part.nodes, False)

        return V_E, V_H

    def impedance_single_parts(self, s, part_o, part_s=None):
        """Calculate a self or mutual impedance matrix at a given complex
        frequency. Note that this abstract function should be called by
        sub-classes, not by the user.

        Parameters
        ----------
        s : complex
            Complex frequency at which to calculate impedance
        part_o : SinglePart
            The observing part, which must be a single part, not a composite
        part_s : SinglePart, optional
            The source part, if not specified will default to observing part
        """

        # TODO: Handle the mutual impedance case

        # if source part is not given, default to observer part
        part_s = part_s or part_o

        basis_o = self.basis_container[part_o]
        basis_s = self.basis_container[part_s]

        normals = basis_o.mesh.surface_normals

        if not (basis_o.mesh.closed_surface and basis_s.mesh.closed_surface):
            raise ValueError("Penetrable objects must be closed")

        # TODO: fix this for mutual impedance terms
        eps_i = part_s.material.epsilon_r(s)
        eps_o = self.background_material.epsilon_r(s)
        mu_i = part_s.material.mu_r(s)
        mu_o = self.background_material.mu_r(s)
        c_i = c/np.sqrt(eps_i*mu_i)
        c_o = c/np.sqrt(eps_o*mu_o)
        eta_i = np.sqrt((mu_i*mu_0)/(eps_i*epsilon_0))
        eta_o = np.sqrt((mu_o*mu_0)/(eps_o*epsilon_0)),

        if isinstance(basis_o, LinearTriangleBasis):
            L_i, S_i = rwg.impedance_G(s, self.integration_rule, basis_o,
                                       part_o.nodes, basis_s, part_s.nodes,
                                       part_o == part_s, eps_i, mu_i,
                                       self.num_singular_terms,
                                       self.singularity_accuracy)

            L_o, S_o = rwg.impedance_G(s, self.integration_rule, basis_o,
                                       part_o.nodes, basis_s, part_s.nodes,
                                       part_o == part_s, eps_o, mu_o,
                                       self.num_singular_terms,
                                       self.singularity_accuracy)

            # This scaling ensures that this operator has the same definition
            # as cursive D defined by Yla-Oijala, Radio Science 2005.
            L_i /= c_i
            S_i *= c_i
            L_o /= c_o
            S_o *= c_o

            # note opposite sign of normals for interior problem
            K_i = rwg.impedance_curl_G(s, self.integration_rule, basis_o,
                                       part_o.nodes, basis_s, part_s.nodes,
                                       -normals, part_o == part_s, eps_i, mu_i,
                                       self.num_singular_terms,
                                       self.singularity_accuracy,
                                       tangential_form=True)

            K_o = rwg.impedance_curl_G(s, self.integration_rule, basis_o,
                                       part_o.nodes, basis_s, part_s.nodes,
                                       normals, part_o == part_s, eps_o, mu_o,
                                       self.num_singular_terms,
                                       self.singularity_accuracy,
                                       tangential_form=True)
        else:
            raise NotImplementedError

        # Build the matrices and metadata for creating the impedance matrix
        # object from the locally defined variables. This relies on them having
        # the correct name in this function. The parent class must set the
        # weights for the different parts of the equations
        loc = locals()
        matrices = {name: loc[name] for name in
                    PenetrableImpedanceMatrix.matrix_names}
        metadata = {name: loc[name] for name in
                    PenetrableImpedanceMatrix.metadata_names if name in loc}
        # TODO: some sub-matrices are symmetric but total isn't...
        metadata["symmetric"] = False
        metadata["operator"] = self

        return matrices, metadata


class PMCHWTOperator(TOperator):
    "Tangential PMCHWT operator for penetrable objects"
    reciprocal = False

    def __init__(self, integration_rule, basis_container,
                 background_material,
                 num_singular_terms=2, singularity_accuracy=1e-5):
        super(PMCHWTOperator, self).__init__(integration_rule, basis_container,
                                             background_material,
                                             num_singular_terms,
                                             singularity_accuracy)

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
        matrices, metadata = super(PMCHWTOperator, self).impedance_single_parts(s, part_o, part_s)
        # set the weights
        metadata['w_EFIE_i'] = 1.0
        metadata['w_EFIE_o'] = 1.0
        metadata['w_MFIE_i'] = 1.0
        metadata['w_MFIE_o'] = 1.0
        return PenetrableImpedanceMatrix(matrices, metadata)
