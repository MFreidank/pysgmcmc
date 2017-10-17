import tensorflow as tf
import numpy as np
from numpy import allclose, asarray
from numpy.random import randint, RandomState
from sklearn.metrics import mean_squared_error
import pytest

from pysgmcmc.diagnostics.objective_functions import sinc
from pysgmcmc.models.bayesian_neural_network import BayesianNeuralNetwork


def test_train_predict_performance():
    """
    This test asserts that training
        pysgmcmc.models.bayesian_neural_network.BayesianNeuralNetwork
    on data from sinc and running predict on seperate validation data
    gives error close to 0.
    """

    rng, n_datapoints = RandomState(randint(0, 10000)), 100
    x_train = asarray([rng.uniform(0., 1., 1) for _ in range(n_datapoints)])
    y_train = sinc(x_train)

    X_test = np.linspace(0, 1, 100)[:, None]
    y_test = sinc(X_test)


    normalization_kwargs = (
        {"normalize_input": boolean, "normalize_output": boolean}
        for boolean in (True, False)
    )

    for normalization in normalization_kwargs:
        graph = tf.Graph()
        with tf.Session(graph=graph) as session:
            bnn = BayesianNeuralNetwork(
                session=session,
                dtype=tf.float64,
                burn_in_steps=1000,
                n_nets=10,
                **normalization,
            )
            bnn.train(x_train, y_train)
            assert bnn.is_trained

            prediction_mean, prediction_variance = bnn.predict(X_test)

        assert allclose(mean_squared_error(y_test, prediction_mean), 0.0, atol=1e-01)


def test_predict_before_train_error():
    """
    This test asserts that calling `predict` on a bnn that has not been trained
    raises a ValueError.
    """

    X_test = np.linspace(0, 1, 100)[:, None]

    graph = tf.Graph()

    with tf.Session(graph=graph) as session:
        bnn = BayesianNeuralNetwork(
            session=session,
            dtype=tf.float64,
            burn_in_steps=1000,
            n_nets=10
        )

        assert not bnn.is_trained

        with pytest.raises(ValueError):
            prediction_mean, prediction_variance = bnn.predict(X_test)
