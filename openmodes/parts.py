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

from openmodes.helpers import Identified
import numpy as np
import weakref

# a constant, indicating that this material is a perfect electric conductor
PecMaterial = "Perfect electric conductor"
    
class Part(Identified):
    """A part which has been placed into the simulation, and which can be
    modified"""

    def __init__(self, location = None):
        super(Part, self).__init__(self)

        self.initial_location = location
        self.transformation_matrix = np.empty((4, 4))
        self.reset()
        self.parent = None
        
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
        if self.parent is None:
            return self.transformation_matrix
        else:
            return self.parent.complete_transformation.dot(
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
        
        axis = np.array(axis)
        axis /= np.sqrt(np.dot(axis, axis))
        
        angle *= np.pi/180.0        
        
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

    def scale(self, scale_factor):
        raise NotImplementedError
        # non-affine transform, will cause problems

        # TODO: work out how scale factor affects pre-calculated 1/R terms
        # and scale them accordingly (or record them if possible for scaling
        # at some future point)

        # also, not clear what would happen to dipole moment

    def shear(self):
        raise NotImplementedError
        # non-affine transform, will cause MAJOR problems

class SinglePart(Part):
    """A single part, which corresponds exactly to one set of basis functions"""

    def __init__(self, mesh, material=PecMaterial, location=None):

        Part.__init__(self, location)                     
        self.mesh = mesh
        self.material = material

        self.initial_location = location
        self.reset()
        self.atom = True

    @property
    def nodes(self):
        "The nodes of this part after all transformations have been applied"
        transform = self.complete_transformation
        return transform[:3, :3].dot(self.mesh.nodes.T).T + transform[:3, 3]

    def __contains__(self, key):
        """Although a single part is not a container, implementing 
        `__contains__` allows recursive checking to be greatly simplified"""
        return self == key


class CompositePart(Part):
    """A composite part containing sub-parts which should be treated as a
    whole
    """
    def __init__(self, location = None, atom=False):

        Part.__init__(self, location)                     
        self.initial_location = location
        self.reset()
        self.parts = []
        self.atom = atom

    def add_part(self, part):
        self.parts.append(part)
        part.parent = weakref.proxy(self)

    def iter_atoms(self):
        """Returns a generator which iterates over all contained parts, but
        does not look within any part which is designated as an atom"""
        for part in self.parts:
            if part.atom:
                yield part
            else:
                for sub_part in part.iter_atoms():
                    yield sub_part

    def iter_single(self):
        """Returns a generator which iterates over all single parts, but
        does not look within any part which is designated as an atom"""
        for part in self.parts:
            if isinstance(part, SinglePart):
                yield part
            else:
                for sub_part in part.iter_single():
                    yield sub_part

    def iter_all(self):
        """Returns a generator which iterates over all parts, with parents
        being visited before children"""
        yield self
        for part in self.parts:
            if isinstance(part, SinglePart):
                yield part
            else:
                for sub_part in part.iter_all():
                    yield sub_part
        
    def __contains__(self, key):
        """Check if the given part is stored within this tree of parts"""
        return self == key or any(key in part for part in self.parts)

