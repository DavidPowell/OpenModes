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
* The software fully supports perfect electric conductors. Most features also
  work for dielectric materials as well, with support expected to improve. Plasmonic
  materials are currently untested.
* Acceleration techniques such as the Fast Multipole Method (FMM) are not used,
  since they are not well-suited for sub-wavelength structures.
* GPU computing is not currently supported.
  
Architecture
------------
There is a fairly complex set of objects which together make up OpenModes.
The user can largely avoid having to know about these, by creating a `Simulation`
object, which hides many implementation details and provides functions to perform
most common tasks. A few other key components are described here:

* Meshes are generated externally, mostly using the `gmsh` program. There is some
  limited support for using `freecad` to create geometries. The geometries shipped
  with openmodes are in parametric form, allowing dimensions to be easily modified
  before meshing. Calling these external meshing utilities, and reading the resultant
  mesh, is handled by `mesh.py`.

* When a mesh is added to the simulation, it is known as a Part, as defined in `part.py`.
  This stores the coordinates of the Part, and the corresponding mesh.  
  
* Basis functions are currently limited to RWG and loop-star functions. They are created
  over a mesh, using the routines in `basis.py`. They are created in a lazy fashion, using
  an object known as a `BasisContainer`.
  
* Operators implement the mathematics at the heart of the program, solving equations
  such as EFIE, MFIE, PMCHWT or CTF. These operators call fast Fortran routines to fill
  the impedance matrices.
 
* Impedance matrices are stored in special objects, defined in `impedance.py`. These
  objects can be indexed by parts, to conveniently find self and mutual impedance terms
  in a system of coupled objects.

* Sources are electric and/or magnetic fields which excite the structure, depending on
  which operator is used. They are defined in `sources.py`, and simple incident fields
  such as a plane wave are defined. It should be relatively easy for the user to add
  some arbitrary incident field form.

* Modes of a structure are found by looking for the poles of the Operator. They are
  stored in an object defined in `modes.py`, enabling their resonant frequency,
  current vector, and the Part they are associated with the be kept together.

  