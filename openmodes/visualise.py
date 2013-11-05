# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 12:17:16 2013

@author: dap124
"""

import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
#import matplotlib.tri as mtri

def plot_parts(parts, figsize=(10, 4), view_angles = (20, 90)):
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    for part in parts:
        ax.scatter(part.nodes[:, 0], part.nodes[:, 1], part.nodes[:, 2], marker='x')
        #mtri.Triangulation(
     
    ax.view_init(*view_angles)
    plt.show()
 

from scipy.constants import mu_0, epsilon_0  
eta_0 = np.sqrt(mu_0/epsilon_0)

     
def write_vtk(self, filename, I=None, scale_coords = 1, 
              scale_current = eta_0, scale_charge = 1):
    """Write the mesh and solution data to a VTK file
    
    If the current vector is given, then currents and charges will be
    written to the VTK file. Otherwise only the mesh is written
    
    Parameters
    ----------
    filename : string
        file to save to (.vtk extension will be added)
    I : ndarray, optional
        current vector of MOM solution in RWG basis
    scale_coords : number, optional
        a scaling factor to apply to the coordinates
    """
    
    from pyvtk import PolyData, VtkData, Scalars, CellData, Vectors
    
    poly = [[int(j) for j in i] for i in self.tri.nodes]    
    struct = PolyData(points=self.nodes*scale_coords, polygons=poly)

    if I is not None:

        rmid, currents, charges = self.face_currents_and_charges(I)

        cell_data = CellData(
                    Vectors((currents.real*scale_current), name="Jr"), 
                    Vectors((currents.imag*scale_current), name="Ji"),
                    Scalars(charges.real*scale_charge, name="qr", 
                            lookup_table="default"),
                    Scalars(charges.imag*scale_charge, name="qi", 
                            lookup_table="default"))
    
        vtk = VtkData(struct, cell_data, "MOM solution")
    else:
        vtk = VtkData(struct, "MOM mesh")
        
    vtk.tofile(filename, "ascii")
    
        