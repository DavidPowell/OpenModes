# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 12:17:16 2013

@author: dap124
"""

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
#import matplotlib.tri as mtri

def plot_parts(parts, figsize=(10, 4), view_angles = (20, 90)):
    """Create a simple 3D plot to show the location of loaded parts in space
    
    Parameters
    ----------
    parts : list of LibraryPart or SimulationPart
        The parts to plot
    figsize : tuple, optional
        The figsize (in inches) which will be passed to matplotlib
    viewangles : tuple, optional
        The viewing angle of the 3D plot in degrees, will be passed to 
        matplotlib
    """
    
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    for part in parts:
        for edge in part.get_edges():
            nodes = part.nodes[edge]
            ax.plot(nodes[:, 0], nodes[:, 1], nodes[:, 2], 'k')
        #ax.scatter(part.nodes[:, 0], part.nodes[:, 1], part.nodes[:, 2], marker='x')
     
    ax.view_init(*view_angles)
    ax.autoscale()
    plt.show()
 

from openmodes.constants import eta_0 
     
def write_vtk(part, filename, I=None, scale_coords = 1, 
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
    
    poly = [[int(j) for j in i] for i in part.tri.nodes]    
    struct = PolyData(points=part.nodes*scale_coords, polygons=poly)

    if I is not None:

        rmid, currents, charges = part.face_currents_and_charges(I)

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
    
        