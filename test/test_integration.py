# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 14:33:28 2013

@author: dap124
"""

from openmodes.integration import sphere_fibonacci
from mayavi import mlab
import numpy as np

#x, y, z = sphere_fibonacci(100)

theta, phi = sphere_fibonacci(20)

x = np.cos(phi) * np.sin(theta)
y = np.cos(phi) * np.cos(theta)
z = np.sin(phi)

mlab.figure()
mlab.points3d(x, y, z)
mlab.show()

