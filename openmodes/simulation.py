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
import logging

from openmodes import gmsh
from openmodes.integration import DunavantRule
from openmodes.parts import SinglePart, CompositePart
from openmodes.basis import LoopStarBasis, BasisContainer
from openmodes.operator import EfieOperator, FreeSpaceGreensFunction
from openmodes.visualise import plot_mayavi, write_vtk, preprocess
from openmodes.model import ScalarModelLeastSq
from openmodes.mesh import TriangularSurfaceMesh
from openmodes.helpers import Identified
from openmodes.vector import VectorParts


class Simulation(Identified):
    """This object controls everything within the simluation. It contains all
    the parts which have been placed, and the operator equation which is
    used to solve the scattering problem.
    """

    def __init__(self, integration_rule=DunavantRule(5),
                 basis_class=LoopStarBasis,
                 operator_class=EfieOperator,
                 greens_function=FreeSpaceGreensFunction(), name=None,
                 basis_args=tuple()):
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
            A name for this simulation
        """

        super(Simulation, self).__init__()

        if name is None:
            name = str(self.id)

        self.integration_rule = integration_rule

        self.parts = CompositePart()

        self.basis_class = basis_class
        self.basis_container = BasisContainer(basis_class, basis_args)
        self.operator = operator_class(integration_rule=integration_rule,
                                       basis_container=self.basis_container,
                                       greens_function=greens_function)

        logging.info('Creating simulation %s\nQuadrature order %d\n'
                     'Basis function class %s'
                     % (name, integration_rule.order, basis_class))

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

    def iter_freqs(self, freqs, log_skip=10, log_label="Sweep frequency"):
        """Return an iterator over a range of frequencies

        It is possible to nest multiple frequency iterators, e.g. to sweep
        both real and imaginary parts of frequency.

        Parameters
        ----------
        freqs : array or list
            All the frequencies over which to iterate, in Hz
        log_skip : integer, optional
            How many frequencies to skip between logging calls. Set this very
            large to avoid all logging.
        log_label : string, optional
            The logging output string, denoting this sweep

        Returns
        -------
        freq_iter : generator
            An iterator, which yields the frequency count and the complex
            frequency `s` for each frequency in the range. It also logs the
            frequency sweep.
        """

        num_freqs = len(freqs)
        for freq_count, freq in enumerate(freqs):
            if freq_count % log_skip == 0 and freq_count != 0:
                logging.info(log_label+" %d/%d" % (freq_count, num_freqs))
            yield freq_count, 2j*np.pi*freq

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

        parent = parent or self.parts
        return self.operator.impedance(s, parent, parent)

    def source_vector(self, source_field, s, parent=None):
        """Evaluate the source vectors due to an incident field, returning
        separate vectors for each part.

        Relevant objects describing incident fields can be constructed from
        the classes found in `openmodes.sources`

        Parameters
        ----------
        source_field: source object
            The object specifying the source field for arbitrary frequencies
        s: complex
            The frequency at which to evaluate the source
        parent : Part, optional
            If specified, then only this part and its sub-parts will be
            calculated

        Returns
        -------
        V : VectorParts
            The source vector, which can be indexed by `Part` objects to find
            the field on each part.
        """

        parent = parent or self.parts

        V = self.empty_vector(parent)

        field_function = getattr(source_field, self.operator.source_field)

        for part in parent.iter_single():
            field = lambda r: field_function(s, r)

            basis = self.basis_container[part]

            V[part] = basis.weight_function(field, self.integration_rule,
                                            part.nodes,
                                            self.operator.source_cross)

        return V

    def singularities(self, s_start, modes, part=None, use_gram=True):
        """Find the singularities of a part or of the whole system

        Parameters
        ----------
        s_start : complex
            The complex frequency at which to perform the estimate. Should be
            within the band of interest
        num_modes : integer or list
            An integer specifying the number of modes to find, or a list of
            mode numbers to find
        part : Part, optional
            The part to solve for. If not specified, the singularities of the
            full system will be solved for
        use_gram : boolean, optional
            Use the Gram matrix to scale the eigenvectors, so that the
            eigenvalues will be independent of the basis functions.

        Returns
        -------
        mode_s : ndarray (num_modes)
            The location of the singularities
        mode_j : ndarray (num_basis, num_modes)
            The current distributions at the singularities
        """

        part = part or self.parts

        return self.operator.singularities(s_start, modes, part, use_gram)

    def construct_models(self, mode_s, mode_j, part=None,
                         model_class=ScalarModelLeastSq):
        """Construct a scalar models from the modes of a part

        Parameters
        ----------
        mode_s : ndarray
            The mode frequency of the whole system
        mode_j : list of ndarray
            The currents for the modes of the whole system
        part : Part, optional
            The part for which to construct the model. If unspecified, a scalar
            model will be created for the full system of all parts
        model_class : class, optional
            The class describing the type of model to construct

        Returns
        -------
        scalar_models : list
            The scalar models
        """

        part = part or self.parts

        scalar_models = []

        for s_n, j_n in zip(mode_s, mode_j.T):
            scalar_models.append(model_class(part, s_n, j_n, self.operator))
        return scalar_models

    def empty_vector(self, part=None, cols=None):
        """
        Create a zero vector of the appropriate size to contain solutions for
        all of the parts, or a single part and its sub-parts

        Parameters
        ----------
        part : Part, optional
            The part for which to create the vector. If not specified, the full
            simulation
        cols : integer, optional
            If specified, this many columns will be included
        """

        part = part or self.parts
        return VectorParts(part, self.basis_container, dtype=np.complex128,
                           cols=cols)

    def plot_3d(self, solution=None, part=None, output_format='webgl',
                filename=None, compress_scalars=None,
                compress_separately=False, **kwargs):
        """Plot a solution on several parts

        Parameters
        ----------
        solution : array, optional
            The solution to plot, typically a vector of current. If not
            specified, then only the geometry will be plotted
        part : Part, optional
            If specified, then only a particular part will be plotted
        output_format : string, optional
            The format of the output. Currently 'mayavi', 'vtk' or 'webgl'
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

        part = part or self.parts

        if solution is None:
            # don't plot a solution, just plot a part
            parts_list = list(part.iter_single())
            charges = currents = centres = None
        else:
            parts_list, charges, currents, centres = preprocess(
                    part, solution, self.basis_container,
                    compress_scalars, compress_separately)

        output_format = output_format.lower()
        if output_format == 'mayavi':
            plot_mayavi(parts_list, charges, currents, vector_points=centres,
                        compress_scalars=compress_scalars, filename=filename)

        elif output_format == 'vtk':
            write_vtk(parts_list, charges, currents, filename=filename,
                      autoscale_vectors=True,
                      compress_separately=compress_separately,
                      scalar_name="charge", vector_name="current")

        elif output_format == 'webgl':
            from openmodes.ipython import plot_3d
            return plot_3d(parts_list, charges, currents, centres,
                           **kwargs)
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
            units is required. Note that `mesh_tol` is expressed in the
            original units of the geometry, before this scale factor is
            applied.
        parameters : dictionary, optional
            A dictionary containing geometric parameters to be overridden in
            a geometry file, before meshing

        Returns
        -------
        parts : tuple
            A tuple of mesh objects, one for each separate geometric
            entity found in the gmsh file

        Currently only `TriangularSurfaceMesh` objects are created
        """

        if osp.splitext(osp.basename(filename))[1] == ".msh":
            # assume that this is a binary mesh already generate by gmsh
            meshed_name = filename
            if parameters != {}:
                raise ValueError("Cannot modify parameters of existing mesh")
        else:
            # assume that this is a gmsh geometry file, so mesh it first
            logging.info("Meshing geometry %s with parameters %s"
                         % (filename, str(parameters)))
            meshed_name = gmsh.mesh_geometry(filename, mesh_tol,
                                             parameters=parameters)

        logging.info("Loading mesh %s" % meshed_name)
        raw_mesh = gmsh.read_mesh(meshed_name)

        parts = tuple(TriangularSurfaceMesh(sub_mesh, scale=scale)
                      for sub_mesh in raw_mesh)
        if len(parts) == 1 and not force_tuple:
            return parts[0]
        else:
            return parts
