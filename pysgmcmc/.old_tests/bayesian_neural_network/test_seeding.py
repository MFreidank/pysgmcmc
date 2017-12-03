import tensorflow as tf

from numpy import allclose, asarray
from numpy.random import randint, RandomState

from pysgmcmc.diagnostics.objective_functions import sinc
from pysgmcmc.models.bayesian_neural_network import BayesianNeuralNetwork
from pysgmcmc.sampling import Sampler
from random import choice

import pytest


def test_default_get_net_seed():
    """
    This test asserts that running
        pysgmcmc.models.bayesian_neural_network.get_default_net
    with the same seed multiple times results in the same network.
    """
    from pysgmcmc.models.bayesian_neural_network import get_default_net
    seed = randint(0, 2 ** 32 - 1)
    n_nets = randint(1, 5)

    rng, n_datapoints = RandomState(randint(0, 10000)), 10
    x_train = asarray([rng.uniform(0., 1., 1) for _ in range(n_datapoints)])

    _, n_inputs = x_train.shape

    def get_net():
        graph = tf.Graph()

        with tf.Session(graph=graph) as session:
            # set up placeholders for data minibatches
            X_Minibatch = tf.placeholder(shape=(None, n_inputs), dtype=tf.float64)
            net_output = get_default_net(inputs=X_Minibatch, seed=seed)
            session.run(tf.global_variables_initializer())
            session.run(net_output, feed_dict={X_Minibatch: x_train})
            return session.run(tf.trainable_variables())

    reference_net = get_net()

    nets = [get_net() for _ in range(n_nets)]

    for net in nets:
        for var1, var2 in zip(reference_net, net):
            assert allclose(var1, var2)
