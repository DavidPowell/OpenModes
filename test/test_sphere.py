# -*- coding: utf-8 -*-
"""
Created on Thu Feb 26 09:36:46 2015

@author: dap124
"""

from __future__ import print_function

import os.path as osp

import numpy as np
from numpy.testing import assert_allclose
import matplotlib.pyplot as plt
import scipy.linalg as la

import openmodes
from openmodes.basis import DivRwgBasis
from openmodes.sources import PlaneWaveSource
from openmodes.constants import c, eta_0
from openmodes.integration import triangle_centres
from openmodes.operator import MfieOperator, EfieOperator

tests_location = osp.split(__file__)[0]
mesh_dir = osp.join(tests_location, 'input', 'test_sphere')
reference_dir = osp.join(tests_location, 'reference', 'test_sphere')


def sphere_extinction_analytical(freqs, r):
    """Analytical expressions for a PEC sphere's extinction for plane wave
    with E = 1V/m

    Parameters
    ----------
    freqs : ndarray
        Frequencies at which to calculate
    r : real
        Radius of sphere
    """
    from scipy.special import sph_jnyn
    N = 40

    k0r = freqs*2*np.pi/c*r
    #scs = np.zeros(len(k0r))
    #scs_modal = np.zeros((len(k0r), N))
    ecs_kerker = np.zeros(len(k0r))

    for count, x in enumerate(k0r):
        jn, jnp, yn, ynp = sph_jnyn(N, x)
        h2n = jn - 1j*yn
        h2np = jnp - 1j*ynp
        a_n = ((x*jnp + jn)/(x*h2np + h2n))[1:]
        b_n = (jn/h2n)[1:]
        #scs[count] = 2*np.pi*sum((2*np.arange(1, N+1)+1)*(abs(a_n)**2 + abs(b_n)**2))/x**2 #
        #scs_modal[count] = 2*np.pi*(2*np.arange(1, N+1)+1)*(abs(a_n)**2 + abs(b_n)**2)/x**2 #
        ecs_kerker[count] = 2*np.pi*np.real(np.sum((2*np.arange(1, N+1)+1)*(a_n + b_n)))/x**2

    return ecs_kerker*r**2/eta_0


def test_extinction_mfie(plot_extinction=False, skip_asserts=False,
                         write_reference=False):
    "Extinction of a PEC sphere with MFIE"

    sim = openmodes.Simulation(name='horseshoe_extinction',
                               basis_class=DivRwgBasis,
                               operator_class=MfieOperator)

    radius = 5e-3
    sphere = sim.load_mesh(osp.join(mesh_dir, 'sphere.msh'))
    sim.place_part(sphere)

    num_freqs = 101
    freqs = np.linspace(1e8, 20e9, num_freqs)

    extinction = np.empty(num_freqs, np.complex128)

    e_inc = np.array([1, 0, 0], dtype=np.complex128)
    k_hat = np.array([0, 0, 1], dtype=np.complex128)
    pw = PlaneWaveSource(e_inc, k_hat)

    for freq_count, s in sim.iter_freqs(freqs):
        Z = sim.impedance(s)
        V = sim.source_vector(pw, s)
        V_E = sim.source_vector(pw, s, which_field="electric_field",
                                n_cross=False)
        extinction[freq_count] = np.vdot(V_E, Z.solve(V))

    extinction_filename = osp.join(reference_dir, 'extinction_mfie.npy')

    if write_reference:
        # generate the reference extinction solution
        np.save(extinction_filename, extinction)

    extinction_ref = np.load(extinction_filename)

    if not skip_asserts:
        assert_allclose(extinction, extinction_ref, rtol=1e-3)

    if plot_extinction:
        # to plot the generated and reference solutions

        # calculate analytically
        extinction_analytical = sphere_extinction_analytical(freqs, radius)
        plt.figure(figsize=(8, 6))
        plt.plot(freqs*1e-9, extinction.real)
        plt.plot(freqs*1e-9, extinction_ref.real, '--')
        plt.plot(freqs*1e-9, extinction_analytical, 'x')
        plt.plot(freqs*1e-9, extinction.imag)
        plt.plot(freqs*1e-9, extinction_ref.imag, '--')
        plt.xlabel('f (GHz)')
        plt.legend(('Calculated (Re)', 'Reference (Re)', 'Analytical (Re)',
                    'Calculated (Im)', 'Reference (Im)'), loc='right')
        plt.title('MFIE Extinction cross section of PEC sphere of radius %.2e'
                  % radius)
        plt.ylim(ymin=0)
        plt.show()


def test_extinction_efie(plot_extinction=False, skip_asserts=False,
                         write_reference=False):
    "Extinction of a PEC sphere with EFIE"

    sim = openmodes.Simulation(name='horseshoe_extinction',
                               basis_class=DivRwgBasis,
                               operator_class=EfieOperator)

    radius = 5e-3
    # this call is to generate and save the reference mesh
#    sphere = sim.load_mesh(osp.join(openmodes.geometry_dir, 'sphere.geo'),
#                           parameters={'radius': radius, 'mesh_tol': 2e-3},
#                           mesh_dir=mesh_dir)
    sphere = sim.load_mesh(osp.join(mesh_dir, 'sphere.msh'))
    sim.place_part(sphere)

    num_freqs = 101
    freqs = np.linspace(1e8, 20e9, num_freqs)

    extinction = np.empty(num_freqs, np.complex128)

    e_inc = np.array([1, 0, 0], dtype=np.complex128)
    k_hat = np.array([0, 0, 1], dtype=np.complex128)
    pw = PlaneWaveSource(e_inc, k_hat)

    for freq_count, s in sim.iter_freqs(freqs):
        Z = sim.impedance(s)
        V = sim.source_vector(pw, s)
        extinction[freq_count] = np.vdot(V, Z.solve(V))

    extinction_filename = osp.join(reference_dir, 'extinction_efie.npy')

    if write_reference:
        # generate the reference extinction solution
        np.save(extinction_filename, extinction)

    extinction_ref = np.load(extinction_filename)

    if not skip_asserts:
        assert_allclose(extinction, extinction_ref, rtol=1e-3)

    if plot_extinction:
        # to plot the generated and reference solutions

        # calculate analytically
        extinction_analytical = sphere_extinction_analytical(freqs, radius)
        plt.figure(figsize=(8, 6))
        plt.plot(freqs*1e-9, extinction.real)
        plt.plot(freqs*1e-9, extinction_ref.real, '--')
        plt.plot(freqs*1e-9, extinction_analytical, 'x')
        plt.plot(freqs*1e-9, extinction.imag)
        plt.plot(freqs*1e-9, extinction_ref.imag, '--')
        plt.xlabel('f (GHz)')
        plt.legend(('Calculated (Re)', 'Reference (Re)', 'Analytical (Re)',
                    'Calculated (Im)', 'Reference (Im)'), loc='right')
        plt.title('Extinction cross section of PEC sphere of radius %.2e'
                  % radius)
        plt.ylim(ymin=0)
        plt.show()


if __name__ == "__main__":
    test_extinction_mfie(plot_extinction=True, skip_asserts=True)
    test_extinction_efie(plot_extinction=True, skip_asserts=True)
