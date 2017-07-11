Mathematical Details
====================

OpenModes is an electromagnetic solver based on the Method of Moments [also known as the boundary element method (BEM)].
It gives us a matrix equation relating a vector of incident fields :math:`\mathbf{V}(s)`

.. math::
    \mathbf{V}(s) = \mathbf{Z}(s)\cdot\mathbf{I}(s)

The matrix :math:`\mathbf{Z}(s)` is constructed from one of the operators described below.
The time convention used is :math:`\exp(st) = \exp\left[(j\omega + \Omega)t\right]`, corresponding to a Laplace
transform relationship with time-domain quantities.

In all cases a Green's function is required to construct the impedance matrix. To date, OpenModes only supports the free-space Green's
function given by

.. math::
    G(\mathbf{r}, \mathbf{r}') = \frac{\exp(-\gamma |\mathbf{r} - \mathbf{r}'|)}{4\pi|\mathbf{r} - \mathbf{r}'|}
    
Here :math:`\gamma=s/c` is the wavenumber in the medium. Other possible Green's functions include anisotropic, periodic, layered
media, and waveguide or closed cavity.
    
    
Perfect Electric Conductor (PEC) Structures
-------------------------------------------

The simplest case is for PEC structures, where electric fields are unable to penetrate into the structure.
In this case the current vector :math:`\mathbf{I}(s)` represents the electrical current on the surface.
There are 3 commonly used operators which are supported by OpenModes:

**Electric Field Integral Equation (EFIE).**

.. math::
  \bar{\bar{t}} \cdot \mathbf{E}_\mathrm{inc}(\mathbf{r}) = -\bar{\bar{t}} \cdot \eta \mathcal{D}_l\mathbf{J}(\mathbf{r})

.. math::
  \mathcal{D}\mathbf{F}(\mathbf{r}) = -\int\left(\gamma + \frac{1}{\gamma}
  \nabla'\nabla'\cdot\right) \mathbf{F}(\mathbf{r}') G(\mathbf{r}, \mathbf{r}') \mathrm{d}\mathbf{r}'
  
  
* The dyadic operator :math:`\bar{\bar{t}}` selects only the surface tangential components
* The excitation vector :math:`\mathbf{V}(s)` contains the tangential electric field.
* This is the only operator which can be
  used for infinitessimally thin metallic layers (as shown by the SRR :doc:`examples <examples>` included
  with OpenModes).
* It has the disadvantage that it can lead to many spurious eigenfrequencies
  clustered about the origin. This problem is minimised through the use of `Loop-Star
  basis functions <http://dx.doi.org/10.1109/8.761074>`_.
* For closed objects, it will also have eigenfrequencies on the :math:`j\omega`
  axis, corresponding to internal resonances of the structure. These are usually
  undesirable, and can be eliminated by excluding the :math:`j\omega` axis from the region
  when searching for modes.
* In OpenModes, this is the default operator, or it can be explicitly
  specified by passing the option :code:`operator_class = EfieOperator` when constructing
  the :code:`Simulation` object.
  
**Magnetic Field Integral Equation (MFIE)**

.. math::
  \hat{\mathbf{n}}\times \mathbf{H}_\mathrm{inc}(\mathbf{r}) = 
  \frac{\mathbf{J}(\mathbf{r})}{2} - \hat{\mathbf{n}}\times
  \mathcal{K}\left[\mathbf{F}(\mathbf{r})\right]
  
.. math::
  \mathcal{K}\left[\mathbf{F}(\mathbf{r})\right] = \int \mathbf{F}(\mathbf{r}') \times \nabla' G(\mathbf{r}, \mathbf{r}') \mathrm{d}\mathbf{r}'
  
  
* Here :math:`\hat{\mathbf{n}}` represents the surface normal
* The excitation vector :math:`\mathbf{V}(s)` contains the rotated tangential magnetic field.
* This only works for closed objects, i.e. it cannot be used to model infinitessimal metallic layers.
* It avoids the problem of
  spurious modes at the origin, however, it does have the problem of internal modes on
  the :math:`j\omega` axis.
* This operator is chosen by passing the option :code:`operator_class = MfieOperator`.
  
**The Combined Field Integral Equation (CFIE)**

* Is a linear superposition of the EFIE and MFIE operators.
* It avoids the problem of spurious modes at the origin, and it also eliminates the internal
  modes on the :math:`j\omega` axis.
* Can only be used for closed objects.
* It is selected by passing the option :code:`operator_class = CfieOperator`.

Dielectric and Plasmonic Structures
-----------------------------------

The chosen time convention means that all material parameters such as permittivity,
permeability, refractive index etc should have negative imaginary parts to satisfy passivity.
Penetrable objects such as dielectrics are described by a surface equivalence problem. Equivalent
surface electric current :math:`\mathbf{J}` and magnetic current :math:`\mathbf{M}` are defined on
the surface. An electric field integral equation can be defined for both the internal and external regions

.. math::
  \bar{\bar{t}} \cdot \mathbf{E}_\mathrm{inc}(\mathbf{r}) = -\bar{\bar{t}} \cdot \eta \mathcal{D}_l\mathbf{J}(\mathbf{r})
  +\bar{\bar{t}} \cdot \mathcal{K}\left[\mathbf{M}(\mathbf{r})\right] - \frac{1}{2} \hat{\mathbf{n}}\times \mathbf{M}(\mathbf{r})

In this equation :math:`\eta` and :math:`\gamma` are calculated from the permittivity and permeability of the internal and external
media respectively to yield the internal and external EFIE. Similarly, the MFIE can be related to both currents, noting that in
this case we consider the tangential form:

.. math::
  \bar{\bar{t}} \cdot \mathbf{H}_\mathrm{inc}(\mathbf{r}) = -\bar{\bar{t}} \cdot \eta \mathcal{D}_l\mathbf{M}(\mathbf{r})
  -\bar{\bar{t}} \cdot \mathcal{K}\left[\mathbf{J}(\mathbf{r})\right] + \frac{1}{2} \hat{\mathbf{n}}\times \mathbf{J}(\mathbf{r})


**Neither the surface equivalent EFIE or MFIE is stable for penetrable objects**. Therefore, they must be combined, to
produce a CFIE type formulation. In all cases, the current vector :math:`\mathbf{I}(j\omega)` contains equivalent surface
electric current :math:`\mathbf{J}` and magnetic current :math:`\mathbf{M}` defined on the surface. There are several different
combined field forms which can be found in the literature:

**Poggio-Miller-Chang-Harrington-Wu-Tsai (PMCHWT) Operator**

* A linear combination of internal and external EFIE and MFIE equations
* The excitation vector :math:`\mathbf{V}(s)` contains both tangential electric and magnetic fields
* Internal sources can also be included
* Has been shown to be positive-definite by `Rodriguez et al <http://dx.doi.org/10.1103/PhysRevB.88.054305>`_,
  which is essential for interference calculations.
* Is selected within OpenModes by passing the option :code:`operator_class = PMCHWTOperator`.

**Combined Tangential Form (CTF) Operator**

* A slight variation on the PMCHWT operator
* May converge faster than PMCHWT for certain material and geometry combinations
* Is not positive definite, therefore it is *not recommended* (particularly for interference calculations)
* Can be selected within OpenModes by passing the option :code:`operator_class = CTFOperator`.


**Volumetric EFIE**

* Instead of a surface equivalent problem, it is possible to solve for the polarisation density throughout the medium
* This approach is not currently supported by OpenModes (and may not ever be)
* This allows for complex internal inhomogeneity of objects
* Has been demonstrated for plasmonics by `Zheng et al <http://dx.doi.org/10.1109/JSTQE.2012.2227684>`_

References
----------

* P. Ylä-Oijala, M. Taskinen, and S. Järvenpää, ‘Surface integral equation formulations for solving electromagnetic scattering problems with iterative methods’,
  `Radio Sci., vol. 40, no. 6, p. RS6002, 2005. <http://dx.doi.org/10.1029/2004RS003169>`_
* C. Forestiere, G. Iadarola, G. Rubinacci, A. Tamburrino, L. Dal Negro, and G. Miano, 
  ‘Surface integral formulations for the design of plasmonic nanostructures’,
  `Journal of the Optical Society of America A, vol. 29, no. 11, pp. 2314–2314, Oct. 2012. <http://dx.doi.org/10.1364/JOSAA.29.002314>`_
* A. M. Kern and O. J. Martin, ‘Surface integral formulation for 3D simulations of plasmonic and high permittivity nanostructures’,
  `Journal of the Optical Society of America A, vol. 26, no. 4, pp. 732–732, Mar. 2009. <http://dx.doi.org/10.1364/JOSAA.26.000732>`_





