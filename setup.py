# -*- coding: utf-8 -*-
"""
OpenModes - An eigenmode solver for open electromagnetic resonantors
Copyright (C) 2013 David Powell

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
#f2py -m openmodes_core -h signature.pyf src/rwg.f90 --overwrite-signature only: 
# set_threads, get_threads, face_integrals_hanninen, triangle_face_to_rwg, face_integrals_complex, scr_index,
#face_integrals_smooth_complex, impedance_core_hanninen, z_efie_faces_self, z_efie_faces_mutual,arcioni_singular,
#voltage_plane_wave, face_to_rwg:

import ez_setup
ez_setup.use_setuptools()

import setuptools

from os.path import join

try:
    import numpy
    #import scipy
except ImportError:
    print "Numpy must be installed"

if numpy.__version__ < '1.6.2':
    raise ValueError("Numpy 1.6.2 or greater required")


from numpy.distutils.core import Extension, setup
#from setuptools import setup

# Ideally would like to perform static linking under mingw32 to avoid
# packaging a whole bunch of dlls. However, static linking is not supported
# for the openmp libraries.
ccompiler_dependent_options = {
    'mingw32' : {
    #    'extra_link_args' : ['-static']
    }
}

# The following options are required to enable openmp to be used in the fortran
# code, which is entirely compiler dependent

fcompiler_dependent_options = {
    # gnu gfortran (including under mingw)
    'gnu95' : {
        'extra_f90_compile_args' : ["-g", "-fimplicit-none",  "-fopenmp", "-O3"],
        'libraries' : ["gomp"]
     },
        
    # intel x86 fortran
    'intel' : {
        'libraries' : ["iomp5"], 
    },
    
    # intel x86_64 fortran
    'intelem' : {
        'libraries' : ["iomp5"], 
    } 
}

core_for = Extension(name = 'openmodes_core',
                 sources = [join('src', 'openmodes_core.pyf'),
                            join('src', 'common.f90'),
                            join('src', 'rwg.f90')], 
#                 f2py_options=["only:",#"set_threads", "get_threads", 
#                                "face_integrals_hanninen",
#                               "triangle_face_to_rwg", 
#                               #"face_integrals_complex", "scr_index",
#                               #"face_integrals_smooth_complex",
#                               "impedance_core_hanninen", "z_efie_faces_self",
#                               #"z_efie_faces_mutual",
#                               "arcioni_singular", "voltage_plane_wave", 
#                               "face_to_rwg", ":",
#                               
#                                "skip:", "vectors", ":"                               
#                               ],
                )

dunavant = Extension(name = 'dunavant', sources=[join('src', 'dunavant.f90')])

from numpy.distutils.command.build_ext import build_ext

class compiler_dependent_build_ext( build_ext ):
    """A build extension which allows compiler-dependent options for
    compilation, linking etc. Options can depend on either the C or FORTRAN
    compiler which is actually used (as distinct from the default compilers,
    which are much easier to detect)
    
    Based on http://stackoverflow.com/a/5192738/482420
    """
    
#    user_options = build_ext.user_options+[
#                ('package-dlls', None, 'Include Mingw32 dlls in binary package')
#                ]  
#                
#    def initialize_options(self):
#        build_ext.initialize_options(self)
#        self.package_dlls=False
    
    def build_extensions(self):
        ccompiler = self.compiler.compiler_type
        fcompiler = self._f77_compiler.compiler_type            

        # add the compiler dependent options to each extension
        for extension in self.extensions:
            try:        
                modification = ccompiler_dependent_options[ccompiler]
                for key, val in modification.iteritems():
                    getattr(extension, key).extend(val)
            except KeyError:
                pass
            
            try:        
                modification = fcompiler_dependent_options[fcompiler]
                for key, val in modification.iteritems():
                    getattr(extension, key).extend(val)
            except KeyError:
                pass        
        
        build_ext.build_extensions(self)

with open('README.txt') as file:
    long_description = file.read()

setup(name = 'OpenModes',
    description = "An eigenmode solver for open electromagnetic resonantors using the method of moments",
    author = "David Powell",
    author_email = 'david.a.powell@anu.edu.au',
    license ='',
    url = '',
    packages = ['openmodes'],
    ext_modules = [dunavant, core_for],
    version = '0.1dev',
    #install_requires = ['numpy >= 1.6.2', 'scipy'],
    long_description=long_description,
    platforms = "Unix, Windows",
    classifiers=[
          'Development Status :: 4 - Beta',
          'Environment :: Console',
          'Environment :: Web Environment',
          'Intended Audience :: Science/Research'
          'Intended Audience :: Developers',
          'License :: OSI Approved :: Python Software Foundation License',
          'Operating System :: Microsoft :: Windows',
          'Operating System :: POSIX',
          'Programming Language :: Python :: 2.7',
          'Programming Language :: Fortran',
          'Topic :: Scientific/Engineering'
          ],
    cmdclass = {'build_ext': compiler_dependent_build_ext},

    # This is a horrible workaround, the dll files will be included for all
    # operating systems.
    data_files = [('', ['libgomp-1.dll', 'libgfortran-3.dll', 'libgcc_s_dw2-1.dll'])]
    )

