Features and Limitations
=========================

OpenModes implements the method of moments (MOM) numerical algorithm,
which is a general approach for solving many electromagnetic scattering
problems. However, the code has been optimised for a specific purpose, namely
extracting simple physical models from full numerical analysis of
structures which are approximately one wavelength in size or smaller.

Features
--------
* Highly flexible python scripting, allows a wide variety of problems
  to be solved and information to be extracted.
* Modelling essentially arbitrary 3D geometries, including
  2D thin layer structures
* Web browser based UI, including interactive 3D plots.
* Key routines are written in Fortran, making them very efficient
* Multi-core CPUs can be utilised effectively, thanks to the use of OpenMP
  parallel constructs.
* Object-oriented design allows different choices such as the type of basis
  function to be changed easily.
  
Limitations
-----------
* The software is currently only implemented for perfect electric conductors.
  Making it applicable to dielectric and plasmonic materials is a high priority.
* Acceleration techniques such as the Fast Multipole Method (FMM) are not used,
  since they are not well-suited for sub-wavelength structures.
* GPU computing is not currently supported.
  
