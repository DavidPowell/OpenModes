# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 12:17:16 2013

@author: dap124
"""

from openmodes.parts import Part

#import matplotlib.tri as mtri

def plot_parts(parts, figsize=(10, 4), view_angles = (20, 90)):
    """Create a simple 3D plot to show the location of loaded parts in space
    
    Parameters
    ----------
    parts : list of `Mesh` or `Part`
        The parts to plot
    figsize : tuple, optional
        The figsize (in inches) which will be passed to matplotlib
    viewangles : tuple, optional
        The viewing angle of the 3D plot in degrees, will be passed to 
        matplotlib
        
    Requires that matplotlib is installed
    """

    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    for part in parts:
        if isinstance(part, Part):
            mesh = part.mesh
        else:
            mesh = part
            
        for edge in mesh.get_edges():
            nodes = part.nodes[edge]
            ax.plot(nodes[:, 0], nodes[:, 1], nodes[:, 2], 'k')
        #ax.scatter(part.nodes[:, 0], part.nodes[:, 1], part.nodes[:, 2], marker='x')
     
    ax.view_init(*view_angles)
    ax.autoscale()
    plt.show()
 

#from openmodes.constants import eta_0 
     
def write_vtk(mesh, nodes, filename, vector_function=None, 
              scalar_function=None, vector_name="vector", 
              scalar_name="scalar"):
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
        
    Requires that pyvtk is installed
    """
    
    from pyvtk import PolyData, VtkData, Scalars, CellData, Vectors
    
    #poly = [[int(j) for j in i] for i in mesh.polygons]    
    polygons = mesh.polygons.tolist()
    struct = PolyData(points=nodes, polygons=polygons)

    data = []

    if vector_function is not None:
        data.append(Vectors(vector_function.real, name=vector_name+"_real"))
        data.append(Vectors(vector_function.imag, name=vector_name+"_imag"))
        
    if scalar_function is not None:
        data.append(Scalars(scalar_function.real, name=scalar_name+"_real", 
                            lookup_table="default"))
        data.append(Scalars(scalar_function.imag, name=scalar_name+"_imag", 
                            lookup_table="default"))
    
    cell_data = CellData(*data)

    vtk = VtkData(struct, cell_data, "OpenModes mesh and data")
    #else:
    #    vtk = VtkData(struct, "MOM mesh")


        
    vtk.tofile(filename, "ascii")
    
        