========
PYSGMCMC
========
|Build Status|
|Docs_|
|Coverage_|
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

.. |Build Status| image:: https://travis-ci.org/MFreidank/pysgmcmc.png?branch_master
   :target: https://travis-ci.org/MFreidank/pysgmcmc

.. |Docs_| image:: https://readthedocs.org/projects/pysgmcmc/badge/?version=latest
   :target: http://pysgmcmc.readthedocs.io/en/latest/
   :alt: Docs

.. |Coverage_| image:: https://coveralls.io/repos/github/MFreidank/pysgmcmc/badge.svg
   :target: https://coveralls.io/github/MFreidank/pysgmcmc
   :alt: Coverage

.. |Health_| image:: https://landscape.io/github/MFreidank/pysgmcmc/master/landscape.svg?style=flat
   :target: https://landscape.io/github/MFreidank/pysgmcmc/master
   :alt: Code Health


Documentation
=============
Our documentation can be found at http://pysgmcmc.readthedocs.io/en/latest/.
