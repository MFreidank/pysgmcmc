
========================
Bayesian Neural Network
========================

An implementation of a **Bayesian Neural Network** that is trained 
using our SGMCMC sampling methods. 

..  XXX: Talk about architecture, cost function etc.

.. module:: bayesian_neural_network

.. autoclass:: BayesianNeuralNetwork
    :members:
    :special-members:
    :private-members:

To discretize possible user choices for the  method used to train a 
Bayesian Neural Network, we maintain an Enum class called **SamplingMethod**. 

The Enum class also provides facilities to obtain a supported sampler directly. 
To obtain a sampler, it is enough to call 
`SamplingMethod.get_sampler(sampling_method, **sampler_args)`
with a supported `sampling_method` and corresponding keyword arguments in 
`sampler_args`.

.. autoclass:: SamplingMethod
    :members:

.. module:: bnn_priors

This module contains our implementation of priors for the weights and log variance 
of our **Bayesian Neural Network**.

.. autoclass:: LogVariancePrior
    :members:
    :special-members:
    :private-members:

.. autoclass:: WeightPrior
    :members:
    :special-members:
    :private-members:
