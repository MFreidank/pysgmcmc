import sys
from os.path import dirname, realpath, join as path_join
import logging

import numpy as np

from bayesian_optimizer import bayesian_optimization, scipy_maximizer, ei, random_sample_maximizer
sys.path.insert(0, path_join(dirname(realpath(__file__)), "..", ".."))
from pysgmcmc.models.bayesian_neural_network import BayesianNeuralNetwork


def branin(x):
    """
    Examples
    -------

    >>> import numpy as np
    >>> optima = [[-np.pi, 12.275], [np.pi, 2.275], [9.42478, 2.475]]
    >>> f_opt = 0.39788735773
    >>> np.allclose([branin(optimum) for optimum in optima], f_opt)
    True

    """
    y = (x[1] - (5.1 / (4 * np.pi ** 2)) * x[0] ** 2 + 5 * x[0] / np.pi - 6) ** 2
    y += 10 * (1 - 1 / (8 * np.pi)) * np.cos(x[0]) + 10
    return y


logging.basicConfig(level=logging.INFO)


# Defining the bounds and dimensions of the input space
lower = np.array([-5, 2])
upper = np.array([10, 20])

# Start Bayesian optimization to optimize the objective function
results = bayesian_optimization(branin, parameter_bounds=(lower, upper),
                                model_function=BayesianNeuralNetwork,
                                num_iterations=30, train_every=1,
                                acquisition_function=ei,
                                acquisition_maximizer=random_sample_maximizer,
                                num_initial_points=3, seed=None, learning_rate=1e-2)
