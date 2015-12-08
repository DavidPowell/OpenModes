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
from openmodes.impedance import PenetrableImpedanceMatrixLA
from openmodes.operator.operator import Operator
from openmodes.operator import rwg
from openmodes.constants import epsilon_0, mu_0, c, eta_0


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

        self.unknowns = ("J", "M")
        self.sources = ("E", "H")
        self.extinction_fields = ("E", "H")
        self.impedance_class = PenetrableImpedanceMatrixLA
        self.frequency_derivatives = False

    def impedance(self, s, parent_o, parent_s, metadata=None):

        metadata = metadata or dict()

        metadata['eta_o'] = self.background_material.eta_r(s)
        metadata['eta_i'] = {}
        metadata['w_EFIE_o'], metadata['w_MFIE_o'] = self.weights_o(s)
        metadata['w_EFIE_i'] = {}
        metadata['w_MFIE_i'] = {}

        for part in parent_s.iter_single():
            metadata['eta_i'][part] = part.material.eta_r(s)
            metadata['w_EFIE_i'][part], metadata['w_MFIE_i'][part] = self.weights_i(s, part)

        return super(TOperator, self).impedance(s, parent_o, parent_s, metadata)

    def impedance_single_parts(self, Z, s, part_o, part_s=None):
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

        is_self_term = part_o == part_s

        matrix_names = ('L_o', 'S_o', 'K_o')
        if isinstance(basis_o, LinearTriangleBasis):
            if is_self_term:
                res = rwg.impedance_G(s, self.integration_rule, basis_o,
                                      part_o.nodes, basis_s, part_s.nodes,
                                      is_self_term, eps_i, mu_i,
                                      self.num_singular_terms,
                                      self.singularity_accuracy)
                L_i = res[0]/c_i*eta_0
                S_i = res[1]*c_i*eta_0

                # note opposite sign of normals for interior problem
                res = rwg.impedance_curl_G(s, self.integration_rule, basis_o,
                                           part_o.nodes, basis_s, part_s.nodes,
                                           -normals, is_self_term, eps_i, mu_i,
                                           self.num_singular_terms,
                                           self.singularity_accuracy,
                                           tangential_form=True)
                K_i = res[0]*eta_0

                matrix_names += ('L_i', 'S_i', 'K_i')

            res = rwg.impedance_G(s, self.integration_rule, basis_o,
                                       part_o.nodes, basis_s, part_s.nodes,
                                       is_self_term, eps_o, mu_o,
                                       self.num_singular_terms,
                                       self.singularity_accuracy)

            # This scaling ensures that this operator has the same definition
            # as cursive D defined by Yla-Oijala, Radio Science 2005.
            L_o = res[0]/c_o*eta_0
            S_o = res[1]*c_o*eta_0

            res = rwg.impedance_curl_G(s, self.integration_rule, basis_o,
                                       part_o.nodes, basis_s, part_s.nodes,
                                       normals, is_self_term, eps_o, mu_o,
                                       self.num_singular_terms,
                                       self.singularity_accuracy,
                                       tangential_form=True)
            K_o = res[0]*eta_0
        else:
            raise NotImplementedError

        # Build the matrices and metadata for creating the impedance matrix
        # object from the locally defined variables. This relies on them having
        # the correct name in this function.
        loc = locals()
        for name in matrix_names:
            Z.matrices[name][part_o, part_s] = loc[name]

    def source_vector(self, source_field, s, parent, extinction_field=False):
        V = super(TOperator, self).source_vector(source_field, s, parent,
                                                 extinction_field)
        if extinction_field:
            V["H"] *= eta_0
        else:
            w_EFIE_o, w_MFIE_o = self.weights_o(s)
            V["E"] *= w_EFIE_o
            V["H"] *= eta_0*w_MFIE_o
        return V


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

    def weights_o(self, s):
        "Weights for outer EFIE and MFIE problems"
        return 1.0, 1.0

    def weights_i(self, s, part):
        "Weights for inner EFIE and MFIE problems, part specific"
        return 1.0, 1.0


class CTFOperator(TOperator):
    """Combined tangential form operator, a better conditioned alternative
    to PMCHWT. See Yla-Oijala, Radio Science 2005

    This operator is further scaled so that the quantities H' = eta_0*H and
    J' = eta_0*J are solved for. This improves the scaling of the eigenvalues,
    giving electric and magnetic modes similar eigenimpedances.
    """
    reciprocal = False

    def __init__(self, integration_rule, basis_container,
                 background_material,
                 num_singular_terms=2, singularity_accuracy=1e-5):
        super(CTFOperator, self).__init__(integration_rule, basis_container,
                                          background_material,
                                          num_singular_terms,
                                          singularity_accuracy)

    def weights_o(self, s):
        "Weights for outer EFIE and MFIE problems"
        eta_o = self.background_material.eta_r(s)
        return 1.0/eta_o, eta_o

    def weights_i(self, s, part):
        "Weights for inner EFIE and MFIE problems, part specific"
        eta_i = part.material.eta_r(s)
        return 1.0/eta_i, eta_i
