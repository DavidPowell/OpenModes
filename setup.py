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

import setuptools

from distutils.util import get_platform
import os.path as osp
import os

from pkg_resources import parse_version

try:
    import numpy
except ImportError:
    numpy_installed = False
else:
    numpy_installed = True

if not numpy_installed or (parse_version(numpy.__version__) < parse_version('1.10.0')):
    raise ValueError("Numpy 1.10.0 or greater required")

from numpy.distutils.core import Extension, setup

import platform

if platform.system() == 'Darwin':
    os.environ["CC"] = "gcc-7"
    os.environ["CXX"] = "gcc-7"

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
        'extra_f90_compile_args': ["-g", "-fimplicit-none", "-fopenmp", "-O3"],
        'libraries': ["gomp"]
     },

    'intel': {
              # Currently ifort gives NaNs in impedance matrix derivative 
              # on -O2, but not on -O3. To be investigated!
              #'extra_f90_compile_args': ['/debug', '-openmp', '-O3', '/fpe:0', '/fp:precise']#, '/traceback'],
              'extra_f90_compile_args': ['-openmp', '-O2', '/fpe:0', '/fp:fast=2']#, '/traceback'],
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
                for key, val in modification.items():
                    getattr(extension, key).extend(val)
            except (KeyError, AttributeError):
                pass

            try:
                modification = fcompiler_dependent_options[fcompiler]
                for key, val in modification.items():
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

try:
    import pypandoc
    long_description = pypandoc.convert('README.md', 'rst')
except(IOError, ImportError):
    long_description = open('README.md').read()

# run the script to find the version
exec(open(osp.join("openmodes", "version.py")).read())

setup(name='OpenModes',
      description="An eigenmode solver for open electromagnetic resonantors",
      author="David Powell",
      author_email='DavidAnthonyPowell@gmail.com',
      license='GPLv3+',
      url='http://davidpowell.github.io/OpenModes',
      packages=setuptools.find_packages(),
      package_data={'openmodes': [osp.join("geometry", "*.geo"),
                                  osp.join("external", "three.js", "*"),
                                  osp.join("templates", "*"),
                                  osp.join("static", "*")]},
      ext_modules=[dunavant, core],
      version=__version__,
      install_requires=['numpy >= 1.10.0', 'scipy >= 0.18.0', 'matplotlib', 'jinja2',
                        'six', 'ipywidgets', 'meshio', 'dill'],
      long_description=long_description,
      long_description_content_type="text/markdown",
      platforms="Windows, Linux",
      classifiers=[
          'Development Status :: 4 - Beta',
          'Environment :: Console',
          'Environment :: Web Environment',
          'Intended Audience :: Science/Research',
          'Intended Audience :: Developers',
          'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',
          'Operating System :: Microsoft :: Windows',
          'Operating System :: POSIX',
          'Programming Language :: Python :: 3.3',
          'Programming Language :: Python :: 3.4',
          'Programming Language :: Python :: 3.5',
          'Programming Language :: Python :: 3.6',
          'Programming Language :: Python :: 3.7',
          'Programming Language :: Fortran',
          'Topic :: Scientific/Engineering'
          ],
      cmdclass={'build_ext': compiler_dependent_build_ext},

      # Include any required library files
      data_files=[('openmodes', redist_data+["RELEASE-VERSION"])]
      )
