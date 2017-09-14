.. _intro:

************
Introduction
************


Purpose
=======
PySGMCMC is a Python package that enables users to fit Bayesian models 
using Markov chain Monte Charlo (MCMC) sampling methods in settings where only 
noisy gradient information is available. 

Due to the stochastic nature of the gradient, these methods are also called 
Stochastic Gradient Markov Chain Monte Carlo (SGMCMC) methods.

One particular target audience for our samplers are Bayesian Deep Learning 
practitioners. In Bayesian Deep Learning datasets quickly become large, 
which makes it intractable to compute the gradient of a model on the whole dataset.
A common remedy for this is to sub-sample the dataset into (mini-) batches. 

.. XXX Finish explanation above


.. PyMC3 is a probabilistic programming module for Python that allows users to fit Bayesian models using a variety of numerical methods, most notably Markov chain Monte Carlo (MCMC) and variational inference (VI). Its flexibility and extensibility make it applicable to a large suite of problems. Along with core model specification and fitting functionality, PyMC3 includes functionality for summarizing output and for model diagnostics.
.. XXX: Explain purpose of pysgmcmc

Features
========
.. PySGMCMC strives to allow all sampling methods to be used as black-boxes 
.. to allow painless and neat integration into arbitrary 

* Modern MCMC solutions applicable when fitting Bayesian models to 
  sub-sampled datasets.

* Tensorflow as the computational backend, which allows for efficient 
  numeric calculation, possibly on GPUs and automatic gradient calculation.

* Flexible: painless application of any of our samplers to your estimation
  problem.


.. A small example
.. ===============
.. XXX: How about this: show a small forward pass through a two-layer network
.. (represented as tf.Variables that are our target params)
.. working on some hpolib function and demonstrate how our sampling methods
.. handle that case

.. For a detailed overview of building models in PyMC3, please read the appropriate sections in the rest of the documentation. For a flavor of what PyMC3 models look like, here is a quick example.
