========
PYSGMCMC
========
|Build Status|
|Health_|

PYSGMCMC is a Python framework for Bayesian Deep Learning which focuses on 
Stochastic Gradient Markov Chain Monte Carlo methods. 

Features
========
* **Complex samplers as black boxes**, computing the next sample with corresponding costs of any MCMC sampler is as easy as:

.. code-block:: python

   sample, cost = next(sampler)

* Based on `tensorflow <https://www.tensorflow.org/>`_ that provides:
    * efficient numerical computation via data flow graphs
    * flexible computation environments (CPU/GPU support, desktop/server/mobile device support)
    * Linear algebra operations

Documentation
=============
Our documentation can be found at http://pysgmcmc.readthedocs.io/en/latest/.

.. |Build Status| image:: https://travis-ci.org/pymc-devs/pymc3.png?branch=master
   :target: https://travis-ci.org/pymc-devs/pymc3

.. |Health_| image:: https://travis-ci.org/pymc-devs/pymc3.png?branch=master
   :target: https://travis-ci.org/pymc-devs/pymc3

