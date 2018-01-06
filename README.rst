========
PYSGMCMC
========
|Build Status|
|Docs_|
|Coverage_|
|Codacy_|

PYSGMCMC is a Python framework for Bayesian Deep Learning that focuses on 
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

.. |Build Status| image:: https://travis-ci.org/MFreidank/pysgmcmc.svg?branch_master
   :target: https://travis-ci.org/MFreidank/pysgmcmc

.. |Docs_| image:: https://readthedocs.org/projects/pysgmcmc/badge/?version=latest
   :target: http://pysgmcmc.readthedocs.io/en/latest/
   :alt: Docs

.. |Coverage_| image:: https://coveralls.io/repos/github/MFreidank/pysgmcmc/badge.svg
   :target: https://coveralls.io/github/MFreidank/pysgmcmc
   :alt: Coverage

.. |Codacy_| image:: https://api.codacy.com/project/badge/Grade/94a3778e36814055ad7b12875857d15e    
   :target: https://www.codacy.com/app/MFreidank/pysgmcmc?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=MFreidank/pysgmcmc&amp;utm_campaign=Badge_Grade
   :alt: Codacy

Install
=======

The quick way::

    pip3 install git+https://github.com/MFreidank/pysgmcmc

Try me
=======

Try our notebooks interactively directly in your browser, no installation 
required:

.. image:: https://mybinder.org/badge.svg 
   :target: https://mybinder.org/v2/gh/MFreidank/pysgmcmc/keras

Documentation
=============
Our documentation can be found at http://pysgmcmc.readthedocs.io/en/latest/.
