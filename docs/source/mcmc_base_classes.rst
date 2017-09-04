==========================================
Base Classes for MCMC Methods
==========================================

.. XXX Needs to be moved into "sampling"

This module provides abstract base classes for our SGMCMC sampling methods. 
All subclasses inheriting from any of these base classes automatically conform 
to the 
`iterator protocol <https://docs.python.org/3/library/stdtypes.html#iterator-types>`_.

This means that extracting the next sample with 
corresponding costs from *any of our samplers* is as simple as:

.. code-block:: python

   sample, cost = next(sampler)


.. module:: mcmc_base_classes

.. autoclass:: MCMCSampler
    :members:
    :special-members:
    :private-members:

For some applications (e.g. `Bayesian Optimization <https://en.wikipedia.org/wiki/Bayesian_optimization>`_), it is important that 
samplers come with as few design choices as possible. 
To reduce the number of such design choices, a recent contribution in the 
literature proposes an on-line *burn-in* procedure. 




.. XXX: Give reference to bohamiann paper and explain burn-in some more

.. autoclass:: BurnInMCMCSampler
    :members:
    :show-inheritance:
    :special-members:
    :private-members:
