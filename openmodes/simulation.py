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

# numpy and scipy
import numpy as np

import os.path as osp

from openmodes import integration, gmsh
from openmodes.parts import SinglePart, CompositePart
from openmodes.impedance import ImpedanceParts
#from openmodes.vector import VectorParts
from openmodes.basis import LoopStarBasis, get_basis_functions
from openmodes.operator import EfieOperator, FreeSpaceGreensFunction
from openmodes.eig import eig_linearised, eig_newton
from openmodes.visualise import plot_mayavi, write_vtk
from openmodes.model import ScalarModel, ScalarModelLS
from openmodes.mesh import TriangularSurfaceMesh
from openmodes.helpers import Identified
#from openmodes.namedarray import NamedArray
from openmodes.vector import empty_vector_parts


class Simulation(Identified):
    """This object controls everything within the simluation. It contains all
    the parts which have been placed, and the operator equation which is
    used to solve the scattering problem.
    """

    def __init__(self, integration_rule=5, basis_class=LoopStarBasis,
                 operator_class=EfieOperator,
                 greens_function=FreeSpaceGreensFunction(),
                 name=None, enable_logging=True, log_display_level=None,
                 log_level="info"):
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
        log_display_level : integer, optional
            The level of logging messages which should be displayed to the
            user via the stderr stream. The default value prevents any logging
            messages from being displayed. Useful values are 20 (general info)
            and 10 (full debugging information)
        """

        super(Simulation, self).__init__()

        if name is None:
            name = self.id

        if enable_logging:
            # create a unique logger for each simulation object

            import logging
            import time
            import tempfile

            self.logger = logging.getLogger(str(self.id))
            formatter = logging.Formatter(name+' - %(asctime)s - %(message)s')

            self.logfile = tempfile.NamedTemporaryFile(mode='wt',
                                    prefix=time.strftime("%Y-%m-%d--%H-%M-%S"),
                                    suffix=".log", delete=False)
            handler = logging.StreamHandler(self.logfile)
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

            self.logger.setLevel(1)
            self.logger.propagate = False

            if log_display_level is not None:
                # dump logging info to the screen as well
                import sys
                handler = logging.StreamHandler(sys.stderr)
                handler.setFormatter(formatter)
                handler.setLevel(log_display_level)
                self.logger.addHandler(handler)
                
        else:
            self.logger = None

        self.quadrature_rule = integration.get_dunavant_rule(integration_rule)

        self.parts = CompositePart()

        self.basis_class = basis_class
        self.operator = operator_class(quadrature_rule=self.quadrature_rule,
                                       basis_class=basis_class,
                                       greens_function=greens_function,
                                       logger=self.logger)

        if self.logger:
            self.logger.info('Creating simulation\nQuadrature order %d\n'
                             'Basis function class %s\n'
                             'Log file %s' %  (integration_rule,
                                               basis_class,
                                               self.logfile.name))

    def place_part(self, mesh=None, parent=None, location=None):
        """Add a part to the simulation domain

        Parameters
        ----------
        mesh : an appropriate mesh object, optional
            If specified, an individual part will be placed, otherwise an
            empty composite part will be created
        parent : CompositePart, optional
            If specified, the part will be a child of some composite part
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

        if mesh is None:
            part = CompositePart(location)
        else:
            part = SinglePart(mesh, location=location)
        
        # if not parent specified, use the root part of the simulation
        parent = parent or self.parts
        if not isinstance(parent, CompositePart):
            raise ValueError("Can only add a part to a composite part")

        parent.add_part(part)

        return part

    def iter_freqs(self, freqs, log_skip=10):
        """Return an iterator over a range of frequencies

        Parameters
        ----------
        freqs : array or list
            All the frequencies over which to iterate, in Hz
        log_skip : integer, optional
            How many frequencies to skip between logging calls

        Returns
        -------
        freq_iter : generator
            An iterator, which yields the frequency count and the complex
            frequency `s` for each frequency in the range. It also logs the
            frequency sweep.
        """

        num_freqs = len(freqs)
        for freq_count, freq in enumerate(freqs):
            if freq_count % log_skip == 0:
                self.logger.info("Sweep frequency %d/%d" %
                                 (freq_count, num_freqs))
            yield freq_count, 2j*np.pi*freq

    def impedance_part(self, s, part_o, part_s=None):
        """Evaluate the self impedance of a part, or the mutual impedance
        between parts.

        Parameters
        ----------
        s : number
            complex frequency at which to calculate impedance (in rad/s)
        part_o : Part
            The observer part, which may also be a composite part
        part_s : Part, optional
            The source part, which may also be a composite part. If
            unspecified, then the source part will be used

        Returns
        -------
        impedance_matrix : ImpedanceMatrix 
            The single self or mutual impedance matrix
        """

        part_s = part_s or part_o

        # if only one impedance term is required
        return self.operator.impedance_matrix(s, part_o, part_s)


    def impedance(self, s, parent=None):
        """Evaluate the self and mutual impedances of all parts in the
        simulation. Return an `ImpedancePart` object which can calculate
        several derived impedance quantities

        Parameters
        ----------
        s : number
            Complex frequency at which to calculate impedance (in rad/s)
        parent : Part, optional
            If specified, then only this part and its sub-parts will be
            calculated

        Returns
        -------
        impedance_matrices : ImpedanceParts
            The impedance matrix object which can represent the impedance of
            the object in several ways.
        """

        matrices = {}
        parent = parent or self.parts

        # TODO: cache individual part impedances to avoid repetition?
        # May not be worth it because mutual impedances cannot be cached
        # except in specific cases such as arrays, and self terms may be
        # invalidated by green's functions which depend on coordinates

        for part_o in parent.iter_single():
            for part_s in parent.iter_single():
                if self.operator.reciprocal and (part_s, part_o) in matrices:
                    # use reciprocity to avoid repeated calculation
                    res = matrices[part_s, part_o].T
                else:
                    res = self.operator.impedance_matrix(s, part_o, part_s)
                matrices[part_o, part_s] = res

        return ImpedanceParts(s, parent, matrices, type(res))

    def source_plane_wave(self, e_inc, jk_inc, parent=None):
        """Evaluate the source vectors due to an incident plane wave, returning
        separate vectors for each part.

        Parameters
        ----------
        e_inc: ndarray
            incident field polarisation in free space
        jk_inc: ndarray.dot(
            incident wave vector in free space

        Returns
        -------
        V : list of ndarray
            the source vector for each part
        """

        parent = parent or self.parts
        vector = empty_vector_parts(parent, self.basis_class, self.operator, 
                                    self.logger, dtype=np.complex128)

        for part in parent.iter_single():
            vector[part] = self.operator.source_plane_wave(part, e_inc, jk_inc)

        return vector

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

        for part in self.parts.iter_single():
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
                Z = self.impedance_part(s_start, part)
                lin_s, lin_currents = eig_linearised(Z, num_modes)

                mode_s = np.empty(num_modes, np.complex128)
                mode_j = np.empty((len(basis), num_modes), np.complex128)

                Z_func = lambda s: self.impedance_part(s, part)[:]

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

        for part_count, part in enumerate(self.parts.iter_single()):
            # TODO: unique ID needs to be modified if different materials or
            # placement above a layer are possible

            unique_id = (part.mesh.id,)  # cache identical parts
            if unique_id in solved_parts:
                scalar_models.append(solved_parts[unique_id])
            else:
                scalar_models.append([])
                for s_n, j_n in zip(mode_s[part_count], mode_j[part_count].T):
                    Z_func = lambda s: self.impedance_part(s, part)
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

        for part_num, part in enumerate(self.parts.iter_single()):
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

    def load_mesh(self, filename, mesh_tol=None, force_tuple=False, scale=None,
                  parameters={}):
        """
        Open a geometry file and mesh it (or directly open a mesh file), then
        convert it into a mesh object. Note that the mesh is _not_ added to
        the simulation.
    
        Parameters
        ----------
        filename : string
            The name of the file to open. Can be a gmsh .msh file, or a gmsh
            geometry file, which will be meshed first
        mesh_tol : float, optional
            If opening a geometry file, it will be meshed with this tolerance
        force_tuple : boolean, optional
            Ensure that a tuple is always returned, even if only a single part
            is found in the file
        scale : real, optional
            A scaling factor to apply to all nodes, in case conversion between
            units is required. Note that `mesh_tol` is expressed in the original
            units of the geometry, before this scale factor is applied.
        parameters : dictionary, optional
            A dictionary containing geometric parameters to be overridden in
            a geometry file, before meshing
    
        Returns
        -------
        parts : tuple
            A tuple of `SimulationParts`, one for each separate geometric entity
            found in the gmsh file
    
        Currently only `TriangularSurfaceMesh` objects are created
        """
    
        if osp.splitext(osp.basename(filename))[1] == ".msh":
            # assume that this is a binary mesh already generate by gmsh
            meshed_name = filename
            if parameters != {}:
                raise ValueError("Cannot modify parameters of an existing mesh")
        else:
            # assume that this is a gmsh geometry file, so mesh it first
            if self.logger:
                self.logger.info("Meshing geometry %s with parameters %s"
                                 % (filename, str(parameters)))
            meshed_name = gmsh.mesh_geometry(filename, mesh_tol, parameters=parameters)

        if self.logger:
            self.logger.info("Loading mesh %s" % meshed_name)
    
        raw_mesh = gmsh.read_mesh(meshed_name)
    
        parts = tuple(TriangularSurfaceMesh(sub_mesh, scale=scale, logger=self.logger)
                      for sub_mesh in raw_mesh)
        if len(parts) == 1 and not force_tuple:
            return parts[0]
        else:
            return parts
