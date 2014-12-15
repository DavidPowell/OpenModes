Technical Details
=================

Performance
-----------
The emphasis of OpenModes is on creating relatively simple physical models from
fully numerical ones. Therefore, squeezing the maximum performance out of this
code has not been the highest priority. Despite this, considerable effort has
gone into optimising the core routines, which have been written in fortran, 
and have been parallelised with OpenMP, to take advantage of modern multi-core
computers.

As the emphasis of this code is on sub-wavelength elements, acceleration
techniques such as the Fast Multipole Method (FMM) are not used.


