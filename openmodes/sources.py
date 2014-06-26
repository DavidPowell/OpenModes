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

"Classes which represent possible distributions of the incident field"

import numpy as np
from openmodes.constants import eta_0, c


class PlaneWaveSource(object):
    def __init__(self, e_inc, k_hat, n=1, eta=eta_0):
        """Generate a plane wave with a given direction of propagation and
        magnitude and direction of the electric field

        Parameters
        ----------
        e_inc: ndarray[3]
            Incident field polarisation in free space
        k_hat: ndarray[3], real
            Incident wave vector in free space
        n: real, optional
            Refractive index of the background medium, defaults to free space
        eta: real, optional
            Characteristic impedance of background medium, defaults to free
            space
        """

        self.e_inc = np.asarray(e_inc)
        k_hat = np.array(k_hat)
        self.k_hat = k_hat/np.sqrt(np.sum(np.abs(k_hat)**2))
        self.eta = eta
        self.c = c/n
        self.h_inc = np.cross(k_hat, e_inc)/eta

    def electric_field(self, s, r):
        """Calculate the electric field distribution at a given frequency

        Parameters
        ----------
        s : complex
            Complex frequency at which to evaluate fields
        r : ndarray, real
            The locations at which to calculate the field. This array can have
            an arbitrary number of dimensions. The last dimension must of size
            3, corresponding to the three cartesian coordinates

        Returns
        -------
        E : ndarray, complex
            An array with the same dimensions as r, giving the field at each
            point
        """
        jk = self.k_hat*s/self.c

        # TODO: check sign of jk!!!
        # dimensions are expanded so that r can have an arbitrary number
        # of dimensions
        return self.e_inc*np.exp(np.dot(r, -jk))[..., None]
