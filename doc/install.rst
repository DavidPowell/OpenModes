Installation Instructions
=========================

OpenModes is a package for the `Python <http://www.python.org/>`_ language, and the
instructions here include several methods to install it, starting from the easiest.

Windows Pre-compiled Package (Recommended Option)
-------------------------------------------------

1. Download the 64 bit version of the `Anaconda`_ 
   Python 3.7 distribution.

2. At the Anaconda command prompt, install gmsh

   ``conda install -c conda-forge gmsh``

3. From the Anaconda command prompt, OpenModes can be installed with the command 

   ``pip install openmodes``.

If installation was successful, the next step is to try the examples, see the :doc:`getting started <gettingstarted>` section.

Docker Image
------------

Docker is a container system (similar to a virtual machine), allowing you to install
OpenModes and all its dependencies in one package.

1. Download `Docker Toolbox <https://www.docker.com/products/docker-toolbox>`_ for Windows or Mac as appropriate

2. Run the installed 'Kitematic' application.

3. Click on `+New` and enter the image name `davidpowell/openmodes`

4. After the image has been downloaded and installed, click on the icon next
   to the web preview button to open the example notebooks in your web browser.
   
To get the best performance, you should change the docker settings to give access
to all CPUs and increase the allocated memory. By default the docker image will only be
accessible from your local machine. If you make it accessible over a network please
enable the `security features of the Jupyter notebook
<http://jupyter-notebook.readthedocs.io/en/latest/security.html>`_.


Upgrading to a Newer Version
----------------------------

If you are using windows and are using the recommended Anaconda distribution,
please upgrade your Anaconda distribution to python 3.6, if you haven't already.
If you need to upgrade your python distribution, then just perform a fresh install
of OpenModes as normal.

Otherwise, first update all your other Python packages with

    ``conda update --all``

You can upgrade your installed version of OpenModes from the command-line, using the command

    ``pip install --upgrade --no-deps openmodes``
   
Detailed Requirements
---------------------
The following software packages are the absolute minimum to run OpenModes:

- python version at least 3.3 (3.7 recommended)
- numpy (1.10.0 or later)
- scipy
- `gmsh`_ (3.x or later)
- matplotlib (or some other package to plot the results)
- jinja2 (for 3D plots in the notebook)
- six (used to write code suited to both python 2/3)

Strongly recommended packages

- Jupyter notebook (or Jupyter lab) is used for the examples, and it allows inline 3D plots
- dill (an alternative to pickle, required for saving many of the objects used by OpenModes)

Other packages which may be useful

- spyder (a GUI for editing python)
- `Mayavi`_ (can produce 3D plots in a GUI window)
- ViSit or `ParaView`_ (3D plotting software to view vtk files)

OpenModes contains some core routines which are optimised using fortran.
Therefore, on platforms where a binary package of OpenModes is not provided,
a fortran compiler is required.

For windows users, there are several choices of scientific python distribution
which allow easy installation of most of the required packages

- `Anaconda`_ (recommended)
- WinPython
- Enthought Python Distribution
- Enthought Canopy

Compiling yourself under windows is quite difficult, due to incompatibility
of the freely available fortran compilers with windows, particularly under 64 bit.
The pre-compiled versions were created with Microsoft C compilers and Intel
Fortran compiler.

Ubuntu Linux
------------

Most Linux distributions come with python and most of the required libraries. 
For ubuntu users, the appropriate packages can be installed using the following
command

``sudo apt-get install python python-numpy python-matplotlib gmsh gfortran
ipython python-dev python-mayavi python-pip``

OpenModes itself can then be installed using the command

``sudo pip install OpenModes``

If you don't have root access to your Linux machine, then use the command

``pip install --user OpenModes``

Alternatively, the `Anaconda`_ distribution can be used just as under Windows

Note that due to a bug in numpy, compilation may fail under Python 3.x. This is fixed
in numpy 1.10.0, but your Linux distribution may have an older release of numpy.
The fix is relatively simple to apply to your own local copy of numpy, see the 
`github pull request <https://github.com/numpy/numpy/pull/5638>`_.

Mac OSX
-------

The code compiles under Mac OSX, but it requires GCC to be installed instead of XCode.
Current versions of XCode include a version of Clang which lacks OpenMP support. According
to `user feedback <https://github.com/DavidPowell/OpenModes/issues/2>`_, it is possible to
compile via the following steps.

* Install python3 and required python libraries
* Install gmsh (Homebrew: brew install homebrew/science/gmsh)
* Manually install GCC(Homebrew: brew install gcc)
* prepend /usr/local/bin to PATH

setup.py has been modified to call gcc-7 under OSX.

Manual Install for Windows
--------------------------

Manual installation under windows is difficult, therefore it is recommended to use the
pre-compiled binaries. Compilation requires the use of 
mingw32 or mingw64 compilers, as these are the only free Fortran compilers available for
windows. Unfortunately the default setting on most systems will not successfully
compile the required libraries.

As of version 4.8 and possibly earlier, Mingw32/64 have a bug which causes the
fortran extensions to randomly generate NaNs in the returned arrays if
optimisation levels -O2 or -O3 are specified, therefore the default optimisation
level has been set to -O1.

Previously 32 bit windows binaries were successfully built under the mingw-64 
x32-4.8.1-posix-dwarf-rev5 compiler. For 64 bit windows, binaries were successfully
build using the x64-4.8.1-posix-seh-rev5 compiler. Both were installed using the
`mingw-builds <http://sourceforge.net/projects/mingwbuilds/>`_ installer.

In order for setup.py to find these compilers, they must be in the path. To be safe,
ensure that no other C or fortran compilers are in the path. Python's distutils
must be instructed to use the mingw32 compiler (for both 32 or 64 bit), using the
``--compiler flag``, or by editing the file ``Lib/site-packages/distutils/distutils.cfg``
under your python installation.

In addition, for 64 bit versions, it is necessary to replace the line
``raise NotImplementedError("Only MS compiler supported with gfortran on win64")`` with 
``pass`` in the file ``Lib/site-packages/numpy/distutils/fcompiler/gnu.py``.

Downloading the Source
----------------------
The source is available on `GitHub <https://github.com/DavidPowell/OpenModes>`_

Runnings Tests
--------------
In the ``test`` folder are several test files, designed to run with
the `pytest <http://pytest.org>`_ framework. After installing the ``pytest`` package, 
run ``py.test`` from this folder.

Building the Documentation
--------------------------

In order to build the documentation, the following packages are required

- Sphinx
- pandoc

At a system command prompt, enter the ``doc`` directory and type ``make html``.

.. _Anaconda: https://www.anaconda.com/distribution/
.. _gmsh: http://gmsh.info/
.. _mayavi: http://docs.enthought.com/mayavi/mayavi/
.. _Paraview: http://www.paraview.org/
