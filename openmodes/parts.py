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

import uuid
import numpy as np

# a constant, indicating that this material is a perfect electric conductor
PecMaterial = "Perfect electric conductor"
    
class Part(object):
    """A part which has been placed into the simulation, and which can be
    modified"""

    def __init__(self, mesh, material = PecMaterial, location = None):
                     
        self.mesh = mesh
        self.material = material
        self.id = uuid.uuid4()

        self.initial_location = location
        self.transformation_matrix = np.empty((4, 4))
        self.reset()
        
    def reset(self):
        """Reset this part to the default values of the original `Mesh`
        from which this `Part` was created
        """        
        self.transformation_matrix[:] = np.eye(4)
        if self.initial_location is not None:
            self.translate(self.initial_location)

    @property
    def nodes(self):
        "The nodes of this part after all transformations have been applied"
        return np.dot(self.transformation_matrix[:3, :3], 
              self.mesh.nodes.T).T + self.transformation_matrix[:3, 3]
        
    def translate(self, offset_vector):
        """Translate a part by an arbitrary offset vector
        
        Care needs to be take if this puts an object in a different layer
        """
        translation = np.eye(4)
        translation[:3, 3] = offset_vector
         
        self.transformation_matrix[:] = np.dot(translation, 
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
        
        self.transformation_matrix[:] = np.dot(matrix, self.transformation_matrix)

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


class PartContainer(object):
    """A specialised list for containing parts. It allows arrays to be handled"""
    def __init__(self):
        self.individual_parts = []
        self.part_arrays = []
