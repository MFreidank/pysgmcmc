.. module:: sampling

=====================
Sampling
=====================

.. XXX: Needs to be changed, this now only has base-classes
This module contains implementations of various SGMCMC samplers which are 
well-suited for Bayesian Deep Learning. For instance, all samplers 
implemented in this module can be easily used to learn a 
BayesianNeuralNetwork.
.. XXX: Cross reference to BNN doku (and to samplers?)


.. toctree::
   :maxdepth: 2
   :caption: Samplers:

   sampling
   samplers/sghmc
   samplers/sgld
   samplers/relativistic_sghmc
