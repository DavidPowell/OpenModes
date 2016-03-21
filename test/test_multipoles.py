# -*- coding: utf-8 -*-
"""
Created on Fri Mar 18 16:59:21 2016

@author: dap124
"""

import numpy as np
import os.path as osp
import matplotlib.pyplot as plt

import openmodes
from openmodes.mesh import gmsh
from openmodes.constants import c
from openmodes.sources import PlaneWaveSource

import helpers

tests_filename = __file__
input_dir = helpers.get_input_dir(tests_filename)
meshfile = osp.join(input_dir, 'sphere.msh')


def generate_mesh():
    "Generate a fixed mesh file to ensure consistent results of tests"
    meshed_name = gmsh.mesh_geometry(osp.join(openmodes.geometry_dir, 'sphere.geo'),
                                     input_dir, parameters={'radius': 1, 'mesh_tol': 0.3})
    assert(meshed_name == meshfile)


def pec_sphere_multipoles(plot=False):
    "Multipole expansion of a PEC sphere"
    sim = openmodes.Simulation(name='pec_sphere_multipoles')
    mesh = sim.load_mesh(meshfile)
    sim.place_part(mesh)

    k0r = np.linspace(0.1, 3, 50)
    freqs = k0r*c/(2*np.pi)
    pw = PlaneWaveSource([1, 0, 0], [0, 0, 1], p_inc=1.0)

    multipole_order = 4
    extinction = np.empty(len(freqs), dtype=np.complex128)

    a_e = {}
    a_m = {}
    for l in range(multipole_order+1):
        for m in range(-l, l+1):
            a_e[l, m] = np.empty(len(freqs), dtype=np.complex128)
            a_m[l, m] = np.empty(len(freqs), dtype=np.complex128)

    for freq_count, s in sim.iter_freqs(freqs):
        Z = sim.impedance(s)
        V = sim.source_vector(pw, s)
        V_E = sim.source_vector(pw, s, extinction_field=True)
        I = Z.solve(V)
        extinction[freq_count] = np.vdot(V_E, I)

        a_en, a_mn = sim.multipole_decomposition(I, multipole_order, s)
        for l in range(multipole_order+1):
            for m in range(-l, l+1):
                a_e[l, m][freq_count] = a_en[l, m]
                a_m[l, m][freq_count] = a_mn[l, m]

    if plot:
        plt.figure()
        plt.plot(k0r, extinction.real)
        plt.xlabel('$k_{0}r$')
        plt.title("Total extinction")
        plt.show()

        plt.figure()
        for l in range(multipole_order+1):
            for m in range(-l, l+1):
                plt.plot(k0r, np.pi/k0r**2*(2*l+1)*np.abs(a_e[l, m])**2)
                plt.plot(k0r, np.pi/k0r**2*(2*l+1)*np.abs(a_m[l, m])**2, '--')
        plt.title("Multipole contributions to scattering")
        plt.xlabel('$k_{0}r$')
        plt.show()

    else:
        return {'name': 'pec_sphere_multipoles',
                'results': {'k0r': k0r, 'extinction': extinction,
                            'a_e': a_e, 'a_m': a_m}}


# The following boilerplate code is needed to generate an actual test from
# the function
def test_pec_sphere_multipoles():
    helpers.run_test(pec_sphere_multipoles, tests_filename)
test_pec_sphere_multipoles.__doc__ = pec_sphere_multipoles.__doc__

if __name__ == "__main__":
    # Uncomment the following lines to update reference solutions
    # generate_mesh()
    # helpers.create_reference(pec_sphere_multipoles, tests_filename)

    # Run the tested functions to produce plots, without any checks
    pec_sphere_multipoles(plot=True)
