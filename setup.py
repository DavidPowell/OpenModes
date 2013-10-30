# -*- coding: utf-8 -*-
"""
Created on Mon May 14 19:13:54 2012

@author: dap124

Requires numpy 1.6.2+ for the extra_compile_args option

#TODO: add copyright
"""

from numpy.distutils.core import Extension, setup
import numpy.distutils.fcompiler

#from distutils.core import setup

fcompiler = numpy.distutils.fcompiler.get_default_fcompiler()

if fcompiler == "gnu95":
    print "Compiling with GNU Fortran"
    openmp_libraries=["gomp"]
    lapack_libraries = ["lapack"]
    extra_f90_compile_args=["-g", "-fimplicit-none",  "-fopenmp", "-O3"] #"-O0", "-fbounds-check", -fstack-arrays , "-frecursive" ,
elif fcompiler in ("intel", "intelem"):
    print "Compiling with Intel Fortran"
    if fcompiler == "intel":
        lapack_libraries = ["mkl_intel", "mkl_intel_thread", "mkl_core", "iomp5", "pthread"]
    else:
        lapack_libraries = ["mkl_intel_lp64", "mkl_intel_thread", "mkl_core", "iomp5", "pthread"]

    openmp_libraries=["iomp5"]
    extra_f90_compile_args=["-g"] # ifortran defaults to -O2 -xhost -openmp #"-ip", #, "-check", "all", "-traceback"    
else:
    raise ValueError("Unknown compiler %s" % fcompiler)

core_for = Extension(name = 'core_for',
                 sources = ['core_for.f90'], 
                 f2py_options=["only:","face_integrals_hanninen",
                               "impedance_core_hanninen", "z_efie_faces",
                               "arcioni_singular", "voltage_plane_wave", 
                               "set_threads", "get_threads", "face_to_rwg", ":"],
                               
                libraries=openmp_libraries, 
                extra_f90_compile_args=extra_f90_compile_args
                )

dunavant = Extension(name = 'dunavant', sources=['dunavant.f90']
                 #,extra_link_args=["-static"]
)


import os
os.environ['CXX'] = 'gcc'

from Cython.Build import cythonize

#numpy_include = r"C:\Programs-non-installed\numpy-1.7.1\numpy\core\include"
import numpy
#cython_modules = cythonize("core_cython.pyx", include_path = [numpy.get_include()]
#)

cython_modules = [
        Extension(name = "core_cython", 
                  sources = ["core_cython.pyx"],
                  include_dirs = [numpy.get_include()],
                  libraries=['m']
                  )]

setup(name = 'openmodes',
      description       = "Find the electromagnetic modes of open structures using the method of moments",
      author            = "David Powell",
      #ext_modules = [dunavant, core_for]
      ext_modules = cythonize(cython_modules)
      )
