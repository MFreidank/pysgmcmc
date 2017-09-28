import tensorflow as tf
import numpy as np
from pysgmcmc.tests.utils import init_random_uniform
from pysgmcmc.sampling import Sampler
from pysgmcmc.models.bayesian_neural_network import BayesianNeuralNetwork

import numbers


def data_for(function, n_train_points=100, dimensionality=2,
             function_domain=(0, 1), seed=1):
    """ Construct a training and test set for `dimensionality`-dimensional
        `function` with domain `function_domain`.
        Hereby, the training set is sampled uniformly at random from the given
        `function_domain`. This random sampling can be seeded by passing
        a `seed`.

    Parameters
    ----------
    function : callable
        XXX DOKU
    n_train_points : int, optional
        XXX Doku
    dimensionality : int, optional
        XXX Doku
    function_domain : Tuple(int, int), optional
        XXX Doku
    seed : int, optional
        XXX Doku

    Returns
    ----------
    XXX Doku

    Examples
    ----------
    XXX Doctest

    """

    assert hasattr(function_domain, "__len__")
    assert len(function_domain) == 2

    assert(dimensionality >= 1)

    lower, upper = function_domain

    assert(dimensionality > 1 or isinstance(lower, numbers.Real))
    assert(dimensionality > 1 or isinstance(upper, numbers.Real))

    assert isinstance(lower, numbers.Real) or hasattr(lower, "__iter__")
    assert isinstance(upper, numbers.Real) or hasattr(upper, "__iter__")

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
        X_test = np.linspace(lower[0], upper[0], num=n_train_points)[:, None]
    else:
        X_test = np.meshgrid(
            *[np.linspace(lower[i], upper[i], n_train_points)
              for i in range(dimensionality)],
            sparse=True
        )
    y_test = function(X_test)

    return {"train": (X_train, y_train), "test": (X_test, y_test)}


# XXX Docu
def sampler_test(objective_function,
                 dimensionality,
                 passing_criterion,  # use default for this (r2 score)
                 function_domain=(0., 1.),
                 n_train_points=100,
                 seed=1,
                 sampling_method=Sampler.SGHMC, sampler_args=None):

    if sampler_args is None:
        sampler_args = dict()

    data = data_for(
        objective_function, dimensionality=dimensionality,
        n_train_points=n_train_points,
        function_domain=function_domain, seed=seed,
    )

    graph = tf.Graph()
    with tf.Session(graph=graph) as session:
        bnn = BayesianNeuralNetwork(
            sampling_method=sampling_method,
            session=session,
            **sampler_args
        )

        bnn.train(*data["train"])

        X_test, y_test = data["test"]

        mean_prediction, variance_prediction = bnn.predict(X_test)

    passing_criterion(
        mean_prediction=mean_prediction,
        variance_prediction=variance_prediction,
        labels=y_test
    )
