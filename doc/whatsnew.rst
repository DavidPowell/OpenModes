What's new in OpenModes
=======================

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
