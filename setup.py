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

import ez_setup
ez_setup.use_setuptools()

import setuptools

from distutils.util import get_platform
import os.path as osp

try:
    import numpy
except ImportError:
    numpy_installed = False
else:
    numpy_installed = True

if not numpy_installed or (numpy.__version__ < '1.6.2'):
    raise ValueError("Numpy 1.6.2 or greater required")

from numpy.distutils.core import Extension, setup

# Ideally would like to perform static linking under mingw32 to avoid
# packaging a whole bunch of dlls. However, static linking is not supported
# for the openmp libraries.
ccompiler_dependent_options = {
    'mingw32': {
    #    'extra_link_args' : ['-static']
    }
}

# The following options are required to enable openmp to be used in the fortran
# code, which is entirely compiler dependent

fcompiler_dependent_options = {
    # gnu gfortran (including under mingw)
    'gnu95': {
        # -O3 is most desireable, but generate NaNs under mingw32
        'extra_f90_compile_args': ["-g", "-fimplicit-none", "-fopenmp", "-O1"],
        'libraries': ["gomp"]
     },

    'intel': {
              # -O3 also causes NaNs under intel fortran
              'extra_f90_compile_args': ['-openmp', '-O2'],
              #'extra_link_args' : ['-openmp']
              #'extra_f77_compile_args' : ['-openmp', '-O3'],
              #'extra_compile_args' : ['-openmp', '-O3', '-static'],
              #'extra_link_args' : ['-nodefaultlib:msvcrt']
    }
}

# Intel fortran compiler goes by several names depending on the version
# and target platform. Here the settings are all the same
fcompiler_dependent_options['intelem'] = fcompiler_dependent_options['intel']
fcompiler_dependent_options['intelvem'] = fcompiler_dependent_options['intel']

core = Extension(name='openmodes.core',
                 sources=[osp.join('src', 'core.pyf'),
                          osp.join('src', 'common.f90'),
                          osp.join('src', 'rwg.f90')],
                 )

dunavant = Extension(name='openmodes.dunavant',
                     sources=[osp.join('src', 'dunavant.pyf'),
                              osp.join('src', 'dunavant.f90')])

from numpy.distutils.command.build_ext import build_ext


class compiler_dependent_build_ext(build_ext):
    """A build extension which allows compiler-dependent options for
    compilation, linking etc. Options can depend on either the C or FORTRAN
    compiler which is actually used (as distinct from the default compilers,
    which are much easier to detect)

    Based on http://stackoverflow.com/a/5192738/482420
    """

    def build_extensions(self):
        ccompiler = self.compiler.compiler_type
        fcompiler = self._f77_compiler.compiler_type

        # add the compiler dependent options to each extension
        for extension in self.extensions:
            try:
                modification = ccompiler_dependent_options[ccompiler]
                for key, val in modification.iteritems():
                    getattr(extension, key).extend(val)
            except (KeyError, AttributeError):
                pass

            try:
                modification = fcompiler_dependent_options[fcompiler]
                for key, val in modification.iteritems():
                    getattr(extension, key).extend(val)
            except (KeyError, AttributeError):
                pass

        build_ext.build_extensions(self)

# Find library files which must be included, which should be placed in the
# appropriate subdirectory of the redist directory. This must be done manually,
# as this code cannot detect which compiler will be used.
redist_path = osp.join("redist", get_platform())
redist_data = []
if osp.exists(redist_path):
    redist_data.append(redist_path)

with open('README.rst') as description_file:
    long_description = description_file.read()

# run the script to find the version
execfile(osp.join("openmodes", "version.py"))

setup(name='OpenModes',
      description="An eigenmode solver for open electromagnetic resonantors",
      author="David Powell",
      author_email='david.a.powell -at- anu.edu.au',
      license='GPLv3+',
      url='https://github.com/DavidPowell/OpenModes',
      packages=setuptools.find_packages(),
      package_data={'openmodes': [osp.join("geometry", "*.geo")]},
      ext_modules=[dunavant, core],
      version=__version__,
      install_requires=['numpy >= 1.6.2', 'scipy', 'matplotlib'],
      long_description=long_description,
      platforms="Windows, Linux",
      use_2to3=True,
      classifiers=[
          'Development Status :: 4 - Beta',
          'Environment :: Console',
          'Environment :: Web Environment',
          'Intended Audience :: Science/Research',
          'Intended Audience :: Developers',
          'License :: OSI Approved :: Python Software Foundation License',
          'Operating System :: Microsoft :: Windows',
          'Operating System :: POSIX',
          'Programming Language :: Python :: 2.7',
          'Programming Language :: Fortran',
          'Topic :: Scientific/Engineering'
          ],
      cmdclass={'build_ext': compiler_dependent_build_ext},

      # Include any required library files
      data_files=[('openmodes', redist_data+["RELEASE-VERSION"])]
      )
