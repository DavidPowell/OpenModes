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


from __future__ import division

import logging
import time
import tempfile

# numpy and scipy
import numpy as np

from openmodes import integration
from openmodes.parts import Part
from openmodes.impedance import ImpedanceParts, ImpedancePartsLoopStar
from openmodes.basis import LoopStarBasis, get_basis_functions
from openmodes.operator import EfieOperator, FreeSpaceGreensFunction
from openmodes.eig import eig_linearised, eig_newton
from openmodes.visualise import plot_mayavi, write_vtk
from openmodes.model import ScalarModel, ScalarModelLS


class Simulation(object):
    """This object controls everything within the simluation. It contains all
    the parts which have been placed, and the operator equation which is
    used to solve the scattering problem.
    """

    def __init__(self, integration_rule=5, basis_class=LoopStarBasis,
                 operator_class=EfieOperator,
                 greens_function=FreeSpaceGreensFunction(),
                 name=None, enable_logging=True):
        """
        Parameters
        ----------
        integration_rule : integer
            the order of the integration rule on triangles
        basis_class : type
            The class representing the type of basis function which will be
            used
        operator_class : type
            The class representing the operator equation to be solved
        greens_function : object, optional
            The Green's function (currently unused)
        name : string, optional
            A name for this simulation, which will be used for logging
        enable_logging : bool, optional
            Enable logging of simulation information to a temporary file
        """

        if name is None:
            name = repr(self)

        if enable_logging:
            # create a unique logger for each simulation object
            self.logger = logging.getLogger(name)
            self.logfile = tempfile.NamedTemporaryFile(mode='wt',
                                    prefix=time.strftime("%Y-%m-%d--%H-%M-%S"),
                                    suffix=".log", delete=False)
            handler = logging.StreamHandler(self.logfile)
            handler.setFormatter(logging.Formatter(
                                 '%(asctime)s - %(levelname)s - %(message)s'))
            self.logger.addHandler(handler)

            #self.logger.setLevel(logging.CRITICAL)
            self.logger.setLevel(logging.INFO)
            self.logger.propagate = False
            #print "Logging info in %s" % self.logfile.name
        else:
            self.logger = None

        self.quadrature_rule = integration.get_dunavant_rule(integration_rule)

        self.parts = []

        self.basis_class = basis_class
        self.operator = operator_class(quadrature_rule=self.quadrature_rule,
                                       basis_class=basis_class,
                                       greens_function=greens_function,
                                       logger=self.logger)

    def place_part(self, mesh, location=None):
        """Add a part to the simulation domain

        Parameters
        ----------
        mesh : an appropriate mesh object
            The part to place
        location : array, optional
            If specified, place the part at a given location, otherwise it will
            be placed at the origin

        Returns
        -------
        part : Part
            The part placed in the simulation

        The part will be placed at the origin. It can be translated, rotated
        etc using the relevant methods of `Part`
        """

        sim_part = Part(mesh, location=location)
        self.parts.append(sim_part)

        #self.logger.info("Placed part %s" % repr(sim_part))

        return sim_part

    def iter_freqs(self, freqs):
        """Return an iterator over a range of frequencies

        Parameters
        ----------
        freqs : array or list
            All the frequencies over which to iterate, in Hz

        Returns
        -------
        freq_iter : generator
            An iterator, which yields the frequency count and the complex
            frequency `s` for each frequency in the range. It also logs the
            frequency sweep.
        """

        num_freqs = len(freqs)
        for freq_count, freq in enumerate(freqs):

            self.logger.info("Sweep frequency %d/%d" % (freq_count, num_freqs))
            yield freq_count, 2j*np.pi*freq

    def impedance(self, s, part_o=None, part_s=None):
        """Evaluate the self and mutual impedances of all parts in the
        simulation. Return an `ImpedancePart` object which can calculate
        several derived impedance quantities

        Parameters
        ----------
        s : number
            complex frequency at which to calculate impedance (in rad/s)
        part_o : Part, optional
            If not specified then all impedances will be calculated. Otherwise
            only the self-impedance for this part, or mutual impedance between
            this part and another
        part_s : Part, optional
            If `part_o` and `part_s` are both specified, then the mutual
            impedance between those two parts will be calculated

        Returns
        -------
        impedance_matrices : ImpedanceParts (if `part_o` not specified)
            The impedance matrix object which can represent the impedance of
            the object in several ways.
        impedance_matrices : ImpedanceMatrix (if `part_o` is specified)
            The impedance matrix of a single part.
        """

        # if only one impedance term is required
        if part_o is not None:
            if part_s is None:
                part_s = part_o
            return self.operator.impedance_matrix(s, part_o, part_s)

        matrices = []

        # TODO: cache individual part impedances to avoid repetition?
        # May not be worth it because mutual impedances cannot be cached
        # except in specific cases such as arrays

        for index_a, part_a in enumerate(self.parts):
            matrices.append([])
            for index_b, part_b in enumerate(self.parts):
                if (index_b < index_a) and self.operator.reciprocal:
                    # use reciprocity to avoid repeated calculation
                    res = matrices[index_b][index_a].T
                else:
                    res = self.operator.impedance_matrix(s, part_a, part_b)
                matrices[-1].append(res)

        if issubclass(self.basis_class, LoopStarBasis):
            ImpedancePartsClass = ImpedancePartsLoopStar
        else:
            ImpedancePartsClass = ImpedanceParts

        return ImpedancePartsClass(s, len(self.parts), matrices)

    def source_plane_wave(self, e_inc, jk_inc):
        """Evaluate the source vectors due to an incident plane wave, returning
        separate vectors for each part.

        Parameters
        ----------
        e_inc: ndarray
            incident field polarisation in free space
        jk_inc: ndarray
            incident wave vector in free space

        Returns
        -------
        V : list of ndarray
            the source vector for each part
        """
        return [self.operator.source_plane_wave(part, e_inc, jk_inc) for part
                in self.parts]

    def part_singularities(self, s_start, num_modes, use_gram=True):
        """Find the singularities of each part of the system in the complex
        frequency plane

        Parameters
        ----------
        s_start : complex
            The complex frequency at which to perform the estimate. Should be
            within the band of interest
        num_modes : integer
            The number of modes to find for each part
        use_gram : boolean, optional
            Use the Gram matrix to scale the eigenvectors, so that the
            eigenvalues will be independent of the basis functions.
            
        Returns
        -------
        mode_s : list of ndarray
            The location of the singularities
        mode_j : list of ndarray
            The current distributions at the singularities
        """

        all_s = []
        all_j = []

        solved_parts = {}

        for part in self.parts:
            # TODO: unique ID needs to be modified if different materials or
            # placement above a layer are possible

            unique_id = (part.mesh.id,)  # cache identical parts
            if unique_id in solved_parts:
                #print "got from cache"
                mode_s, mode_j = solved_parts[unique_id]
            else:
                if self.logger:
                    self.logger.info("Finding singularities for part %s"
                                     % str(unique_id))
                # first get an estimate of the pole locations
                basis = get_basis_functions(part.mesh, self.basis_class, self.logger)
                Z = self.operator.impedance_matrix(s_start, part)
                lin_s, lin_currents = eig_linearised(Z, num_modes)

                mode_s = np.empty(num_modes, np.complex128)
                mode_j = np.empty((len(basis), num_modes), np.complex128)

                Z_func = lambda s: self.operator.impedance_matrix(s, part)[:]

                if use_gram:
                    G = basis.gram_matrix

                for mode in xrange(num_modes):
                    res = eig_newton(Z_func, lin_s[mode], lin_currents[:, mode],
                                     weight='max element', lambda_tol=1e-8,
                                     max_iter=200)

                    lin_hz = lin_s[mode]/2/np.pi
                    nl_hz = res['eigval']/2/np.pi
                    if self.logger:
                        self.logger.info("Converged after %d iterations\n"
                                         "%+.4e %+.4ej (linearised solution)\n"
                                         "%+.4e %+.4ej (nonlinear solution)"
                                         % (res['iter_count'],
                                            lin_hz.real, lin_hz.imag,
                                            nl_hz.real, nl_hz.imag))

                    mode_s[mode] = res['eigval']
                    j_calc = res['eigvec']

                    if use_gram:
                        j_calc /= np.sqrt(j_calc.T.dot(G.dot(j_calc)))
                    else:
                        j_calc /= np.sqrt(np.sum(j_calc**2))

                    mode_j[:, mode] = j_calc

                # add to cache
                solved_parts[unique_id] = (mode_s, mode_j)

            all_s.append(mode_s)
            all_j.append(mode_j)

        return all_s, all_j

    def system_singularities(self, s_start, num_modes, use_gram=True):
        """Find the singularities of the whole system in the complex frequency
        plane

        Parameters
        ----------
        s_start : number
            The complex frequency at which to perform the estimate. Should be
            within the band of interest
        num_modes : integer
            The number of modes to find
        use_gram : boolean, optional
            Use the Gram matrix to scale the eigenvectors, so that the
            eigenvalues will be independent of the basis functions.

        Returns
        -------
        mode_s : 1D array
            The singularities coresponding to the resonant frequencies
        mode_j : 2D array
            The eigencurrents, the columns of which corresponding to the
            solutions of the system without excitation, at the frequencies
            given by `mode_s`
        """

        # first get an estimate of the pole locations
        Z = self.impedance(s_start).combine_parts()
        lin_s, lin_currents = eig_linearised(Z, num_modes)

        mode_s = np.empty_like(lin_s)
        mode_j = np.empty_like(lin_currents)

        Z_func = lambda s: self.impedance(s).combine_parts()[:]

        if self.logger:
            self.logger.info("Finding singularities for the whole system")

        if use_gram:
            G = Z.basis_o.gram_matrix

        for mode in xrange(num_modes):
            res = eig_newton(Z_func, lin_s[mode], lin_currents[:, mode],
                             weight='max element', lambda_tol=1e-8,
                             max_iter=200)

            lin_hz = lin_s[mode]/2/np.pi
            nl_hz = res['eigval']/2/np.pi
            if self.logger:
                self.logger.info("Converged after %d iterations\n"
                                 "%+.4e %+.4ej (linearised solution)\n"
                                 "%+.4e %+.4ej (nonlinear solution)"
                                 % (res['iter_count'],
                                    lin_hz.real, lin_hz.imag,
                                    nl_hz.real, nl_hz.imag))

            mode_s[mode] = res['eigval']
            j_calc = res['eigvec']

            if use_gram:
                j_calc /= np.sqrt(j_calc.T.dot(G.dot(j_calc)))
            else:
                j_calc /= np.sqrt(np.sum(j_calc**2))

            mode_j[:, mode] = j_calc

        return mode_s, mode_j

    def construct_model_system(self, mode_s, mode_j):
        """Construct a scalar model for the modes of the whole system

        Parameters
        ----------
        mode_s : ndarray
            The mode frequency of the whole system
        mode_j : list of ndarray
            The currents for the modes of the whole system

        Returns
        -------
        scalar_models : list
            The scalar models
        """

        scalar_models = []

        for s_n, j_n in zip(mode_s, mode_j.T):
            Z_func = lambda s: self.impedance(s).combine_parts()
            scalar_models.append(ScalarModel(s_n, j_n, Z_func,
                                             logger=self.logger))
        return scalar_models

    def construct_models(self, mode_s, mode_j):
        """Construct a scalar model for the modes of each part

        Parameters
        ----------
        mode_s : list of ndarray
            The mode frequency of each part
        mode_j : list of ndarray
            The currents for the modes of each part

        Returns
        -------
        scalar_models : list
            The scalar models
        """

        solved_parts = {}
        scalar_models = []

        for part_count, part in enumerate(self.parts):
            # TODO: unique ID needs to be modified if different materials or
            # placement above a layer are possible

            unique_id = (part.mesh.id,)  # cache identical parts
            if unique_id in solved_parts:
                scalar_models.append(solved_parts[unique_id])
            else:
                scalar_models.append([])
                for s_n, j_n in zip(mode_s[part_count], mode_j[part_count].T):
                    Z_func = lambda s: self.operator.impedance_matrix(s, part)
                    scalar_models[-1].append(ScalarModel(s_n, j_n, Z_func,
                                                         logger=self.logger))

                solved_parts[unique_id] = scalar_models[-1]

            return scalar_models

    def plot_solution(self, solution, output_format, filename=None,
                      compress_scalars=None, compress_separately=False):
        """Plot a solution on several parts

        Parameters
        ----------
        solution : array
            The solution to plot, typically a vector of current
        output_format : string
            The format of the output. Currently 'mayavi' or 'vtk'
        filename : string, optional
            If saving to a file, the name of the file to save to
        compress_scalars : real, optional
            Compression factor to change the dynamic range of the scalar
            solution, which will make the resulting plot easier to view, but
            less 'physically correct'
        compress_separately : boolean, optional
            If compressing dynamic range, do it separately for each part. This
            will conceal any difference in the relative strength of excitation
            between parts.
        """

        charges = []
        currents = []
        centres = []

        for part_num, part in enumerate(self.parts):
            I = solution[part_num]
            basis = get_basis_functions(part.mesh, self.basis_class, self.logger)

            centre, current, charge = basis.interpolate_function(I,
                                                            return_scalar=True,
                                                            nodes=part.nodes)
            charges.append(charge.real)
            currents.append(current.real)
            centres.append(centre)

        output_format = output_format.lower()
        if output_format == 'mayavi':
            plot_mayavi(self.parts, charges, currents, vector_points=centres,
                        compress_scalars=compress_scalars, filename=filename)

        elif output_format == 'vtk':
            write_vtk(self.parts, charges, currents, filename=filename,
                      compress_scalars=compress_scalars,
                      autoscale_vectors=True,
                      compress_separately=compress_separately)
        else:
            raise ValueError("Unknown output format")
