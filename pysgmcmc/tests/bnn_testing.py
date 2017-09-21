import tensorflow as tf
import numpy as np
from tests.utils import init_random_uniform
from pysgmcmc.sampling import Sampler
from pysgmcmc.models.bayesian_neural_network import BayesianNeuralNetwork

import numbers


# XXX Doku and proper testing
def data_for(function, n_train_points=100, dimensionality=2,
             function_domain=(0., 1.), seed=1):

    assert(hasattr(function_domain, "__len__"))
    assert(len(function_domain) == 2)

    lower, upper = function_domain

    if dimensionality == 1:
        # XXX Message
        # dimensionality of 1 implies that lower and upper bound of data
        # must be floats
        assert(isinstance(lower, numbers.Real))
        assert(isinstance(upper, numbers.Real))

    assert(isinstance(lower, numbers.Real) or hasattr(lower, "__iter__"))
    assert(isinstance(upper, numbers.Real) or hasattr(upper, "__iter__"))

    if isinstance(lower, numbers.Real):
        lower = np.ones(dimensionality) * lower
    else:
        lower = np.asarray(lower)

    if isinstance(upper, numbers.Real):
        upper = np.ones(dimensionality) * upper
    else:
        upper = np.asarray(lower)

    if seed is None:
        rng = np.random.RandomState(np.random.randint(0, 10000))
    else:
        rng = np.random.RandomState(seed)

    X_train = init_random_uniform(
        lower=lower, upper=upper, n_points=n_train_points, rng=rng
    )

    y_train = function(X_train)

    if dimensionality == 1:
        X_test = np.linspace(lower[0], upper[0], num=n_train_points * 10)
    else:
        X_test = np.meshgrid(
            *[np.linspace(lower[i], upper[i], n_train_points)
              for i in range(dimensionality)],
            sparse=True
        )
    y_test = function(X_test)

    return {"train": (X_train, y_train), "test": (X_test, y_test)}


def sampler_test(objective_function,
                 dimensionality,
                 passing_criterion,  # use default for this (r2 score)
                 function_domain=(0., 1.),
                 n_train_points=100,
                 seed=1,
                 sampling_method=Sampler.SGHMC, **sampler_args):

    data = data_for(
        objective_function, dimensionality=dimensionality,
        n_train_points=n_train_points,
        function_domain=function_domain, seed=seed,
    )

    with tf.Session() as session:
        bnn = BayesianNeuralNetwork(
            sampling_method=sampling_method, sampler_args=sampler_args,
            session=session
        )

        bnn.fit(*data["train"])

        X_test, y_test = data["test"]

        mean_prediction, variance_prediction = bnn.predict(X_test)

    passing_criterion(
        mean_prediction=mean_prediction,
        variance_prediction=variance_prediction,
        labels=y_test
    )
