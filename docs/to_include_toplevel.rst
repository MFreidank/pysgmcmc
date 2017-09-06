## Models
.. XXX: Move this to toplevel doku outside of api
.. This module contains implementations of machine learning models, which 
.. make use of our various SGMCMC samplers. 
.. Our models follow a standard *train* and *predict* interface that serves to 
.. hide unnecessary detail from the user. 


.. Finally, all our samplers can also be used stand-alone, 
.. they do not have a strong dependency on our models or our model interface. 
.. They merely expect a cost function that takes your model parameters as input 
.. and returns a (negative log likelihood) cost value when called.

.. Therefore, it is entirely possible to use them to build custom 
.. implementations of your favorite machine learning method. 


Samplers
==================
This module contains our actual samplers. 

All our samplers share the following common interface:

.. code-block:: python

   sample, cost = next(sampler)

This statement will provide the next sample with its corresponding cost
value for any of our sampling methods. 

.. XXX: Next below should be marked as python code, but inline
.. XXX: Link to python generator docu below [generators]
.. XXX: n should be rendered as math
One other thing that all our samplers share is that they are *infinite* 
generators. As such, they do not have an upper bound on how many times one may 
call *next* on a sampler. 
This renders our samplers applicable for various use-cases:
* sample for a fixed budget of $n$ iterations (simply call *next* $n$ times)
* sample until some condition is met
* run a sampler on-line (potentially forever)

.. XXX: More usecases?



.. XXX: Point out that they may be used in for-loops, but that this requires explicit breaking

Sampling
=====================

This module provides abstract base classes for our SGMCMC sampling methods. 
All subclasses inheriting from any of these base classes automatically conform 
to the 
`iterator protocol <https://docs.python.org/3/library/stdtypes.html#iterator-types>`_.

This means that extracting the next sample with 
corresponding costs from *any of our samplers* is as simple as:

.. code-block:: python

   sample, cost = next(sampler)

For some applications (e.g. `Bayesian Optimization <https://en.wikipedia.org/wiki/Bayesian_optimization>`_), it is important that 
samplers come with as few design choices as possible. 
To reduce the number of such design choices, a recent contribution in the 
literature proposes an on-line *burn-in* procedure. 


========================
Bayesian Neural Network
========================

An implementation of a **Bayesian Neural Network** that is trained 
using our SGMCMC sampling methods. 

..  XXX: Talk about architecture, cost function etc.


To discretize possible user choices for the  method used to train a 
Bayesian Neural Network, we maintain an Enum class called **SamplingMethod**. 

The Enum class also provides facilities to obtain a supported sampler directly. 
To obtain a sampler, it is enough to call 
`SamplingMethod.get_sampler(sampling_method, **sampler_args)`
with a supported `sampling_method` and corresponding keyword arguments in 
`sampler_args`.


This module contains our implementation of priors for the weights and log variance 
of our **Bayesian Neural Network**.
