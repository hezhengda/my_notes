NequIP
======

In this section, we will introduce the `NequIP <https://github.com/mir-group/nequip>`_ package, which implements an equivariant neural network potential for crystal simulations. NequIP (Neural Equivariant Interatomic Potentials) is a powerful tool for developing accurate and efficient machine learning models for atomic-scale simulations, particularly in crystalline materials.

NequIP leverages the principles of equivariance to create potentials that respect the symmetries inherent in crystal structures. This approach ensures that the model predictions are invariant of scalar (e.g. energy) but equivariant of vector (e.g. forces) and tensor (e.g. stress) quantities under rotations, translations, and permutations of atoms, leading to more physically consistent and generalizable results.

We will explore both the theoretical foundations and practical applications of NequIP in the following subsections.

.. toctree::
   :maxdepth: 2
   :caption: NequIP:

   NequIP_theory
   NequIP_practice