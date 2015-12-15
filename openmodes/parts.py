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

from openmodes.helpers import Identified, PicklableRef
import numpy as np

from openmodes.material import PecMaterial


class Part(Identified):
    """A part which has been placed into the simulation, and which can be
    modified"""

    def __init__(self, location=None):
        super(Part, self).__init__()

        self.initial_location = location
        self.transformation_matrix = np.empty((4, 4))
        self.reset()
        self.parent_ref = None

    def reset(self):
        """Reset this part to the default values of the original `Mesh`
        from which this `Part` was created
        """
        self.transformation_matrix[:] = np.eye(4)
        if self.initial_location is not None:
            self.translate(self.initial_location)

    @property
    def complete_transformation(self):
        """The complete transformation matrix, which takes into account
        the transformation matrix of all parents"""
        if self.parent_ref is None:
            return self.transformation_matrix
        else:
            return self.parent_ref().complete_transformation.dot(
                                                    self.transformation_matrix)

    def translate(self, offset_vector):
        """Translate a part by an arbitrary offset vector

        Care needs to be take if this puts an object in a different layer
        """
        translation = np.eye(4)
        translation[:3, 3] = offset_vector

        self.transformation_matrix[:] = translation.dot(
                                                    self.transformation_matrix)

    def rotate(self, axis, angle):
        """
        Rotate about an arbitrary axis

        Parameters
        ----------
        axis : ndarray
            the vector about which to rotate
        angle : number
            angle of rotation in degrees

        Algorithm taken from
        http://en.wikipedia.org/wiki/Euler%E2%80%93Rodrigues_parameters
        """

        # TODO: enable rotation about arbitrary coordinates, and about the
        # centre of the object

        axis = np.array(axis, np.float64)
        axis /= np.sqrt(np.dot(axis, axis))

        angle = angle*np.pi/180.0

        a = np.cos(0.5*angle)
        b, c, d = axis*np.sin(0.5*angle)

        matrix = np.array([[a**2 + b**2 - c**2 - d**2,
                            2*(b*c - a*d), 2*(b*d + a*c), 0],
                           [2*(b*c + a*d), a**2 + c**2 - b**2 - d**2,
                            2*(c*d - a*b), 0],
                           [2*(b*d - a*c), 2*(c*d + a*b),
                            a**2 + d**2 - b**2 - c**2, 0],
                           [0, 0, 0, 1.0]])

        self.transformation_matrix[:] = matrix.dot(self.transformation_matrix)


class SinglePart(Part):
    """A single part, which corresponds exactly to one set of basis
    functions"""

    def __init__(self, mesh, material=PecMaterial, location=None):

        Part.__init__(self, location)
        self.mesh = mesh
        self.material = material

        self.initial_location = location
        self.reset()

        # A SinglePart is uniquely identified by its mesh and material. i.e.
        # all parts with the same unique_id would have the same modes etc. This
        # must be used with care since it does not account for the background
        # material (presumably the same for all parts), and it may be
        # invalidated by a Green's function for inhomogeneous media.
        self.unique_id = (mesh.id, material.id)

    @property
    def nodes(self):
        "The nodes of this part after all transformations have been applied"
        transform = self.complete_transformation
        return transform[:3, :3].dot(self.mesh.nodes.T).T + transform[:3, 3]

    def __contains__(self, key):
        """Although a single part is not a container, implementing
        `__contains__` allows recursive checking to be greatly simplified"""
        return self == key

    def iter_single(self):
        "Iterating over a single part just yields itself"
        yield self

    def iter_all(self, parent_first=None):
        "Iterating over a single part just yields itself"
        yield self

    def iter_lowest(self, lowest, parent_order=None):
        "Iterating over a single part just yields itself"
        if self not in lowest:
            raise ValueError("Reached unexpected part")
        yield self


class MultiPart(Part):
    """A collection of several parts, which should generally be treated
    separately
    """
    def __init__(self, location=None, children=None):
        """Note that location is essentially meaningless in this case"""
        super(MultiPart, self).__init__(location)
        self.children = children or []

    @property
    def unique_id(self):
        """The unique_id of a composite part is just the concatenation of that
        of its child parts"""
        return ("MultiPart",)+tuple(part.unique_id for part in self.children)

    def clear(self):
        "Clear all child parts from this part"
        self.children = []

    def add_part(self, part):
        "Add a part to this part, but do not become its parent"
        self.children.append(part)

    def iter_single(self):
        """Iterates over all single parts which are within this part

        Returns
        -------
        it : iterator
            An iterator object over all SingleParts
        """
        for part in self.children:
            for sub_part in part.iter_single():
                yield sub_part

    def iter_all(self, parent_first=False):
        """Iterates over all parts within this part

        Parameters
        ----------
        parent_first : boolean, optional
            If True, the a node is visited before all of its children,
            otherwise afterwards

        Returns
        -------
        it : iterator
            An iterator object over all parts
        """
        if parent_first:
            yield self
        for part in self.children:
            for sub_part in part.iter_all(parent_first=parent_first):
                yield sub_part
        if not parent_first:
            yield self

    def iter_lowest(self, lowest, parent_order=None):
        """Iterate through parts only down to the lowest level specified. An
        error occurs if a child part is reached which is not in `lowest`.

        Parameters
        ----------
        lowest: set
            The lowest level parts to be iterated down to
        parent_order : string, optional
            If 'before', the a node is visited before all of its children,
            if 'after' visited afterwards, otherwise not visited
        """
        if self in lowest:
            yield self
        else:
            if parent_order == 'before':
                yield self
            for part in self.children:
                for sub_part in part.iter_lowest(lowest, parent_order=parent_order):
                    yield sub_part
            if parent_order == 'after':
                yield self

    def __contains__(self, key):
        """Check if the given part is stored within this tree of parts"""
        return self == key or any(key in part for part in self.children)


class CompositePart(MultiPart):
    """A composite part containing sub-parts which can be treated as a
    whole. The sub-parts have this composite part as a parent.
    """
    def __init__(self, location=None):
        super(CompositePart, self).__init__(location)

    @property
    def unique_id(self):
        """The unique_id of a composite part is just the concatenation of that
        of its child parts"""
        return ("CompositePart",)+tuple(part.unique_id for part in self.children)

    def add_part(self, part):
        "Add a part to this part, and become its parent part"
        if part.parent_ref is not None:
            raise ValueError("Part already has a different parent")
        self.children.append(part)
        part.parent_ref = PicklableRef(self)
