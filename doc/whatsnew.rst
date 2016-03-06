What's new in OpenModes
=======================

**Release 1.0.0**

March 2016

- Dielectric objects can now be handled via a surface equivalent problem
- New, more robust method for finding complex poles
- New classes to conveniently represent modes and use them for modelling
- Improved Python 3 support (Python 3.5 now recommended)
- Improvements to 3D plotting
- Working code for spherical multipole decomposition
- Impedance matrix objects are greatly simplified
- New pre-made geometries are included

Note that code for old versions of OpenModes will need to be modified to work
with the new version. All examples have been updated.

**Release 0.0.5**

March 2015

- In addition to the usual EFIE (electric field integral equation), it is now possible to solve
  the MFIE (magnetic field integral equation). This code is not yet as well tested as the EFIE,
  and does not yet support all the same functionality.
- Improved handling of singular integrals
- Automated tests added
- In addition to Python 2.7, OpenModes is now compatible with Python 3.3+.
- Fixed missing static files, so 3D plotting in the browser should now work correctly

**Release 0.0.4**

December, 2014

- New *3D Plotting in the web browser*, showing geometry and optionally charge
  and current distribution. There is an example file showing how this works
- Fixed a further bug with installing from source package.
- The examples have been moved to their own repository at https://github.com/DavidPowell/openmodes-examples.

**Release 0.0.3**

November, 2014

- Fixed bugs which prevented installing from source package. Installation under
  Linux should now work correctly.
- Documentation improvements

**Release 0.0.2**

August, 2014

Major changes which are visible to the user:

- The incident field weighting can now be performed in pure python code, instead
  of fortran subroutines. This makes it easier to model sources other than plane waves.
- The use of logging is simplified, to allow simulation progress to be monitored.
- Fixed several cases where data structures could not be pickled. This means that 
  structures can be cached or transmitted across the network for parallel computing.
- Additional pre-made geometries are included
- Planar structures with internal holes are now handled correctly

**Release 0.0.1**

May, 2014

The initial release, which was used to produce results for the arXiv paper
and manuscript submitted for review.
