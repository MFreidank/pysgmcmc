import tensorflow as tf

from numpy import allclose, asarray
from numpy.random import randint, RandomState

from pysgmcmc.diagnostics.objective_functions import sinc
from pysgmcmc.models.bayesian_neural_network import BayesianNeuralNetwork
# from pysgmcmc.sampling import Sampler
# from random import choice

import pytest


def test_default_get_net_seed():
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


@pytest.mark.xfail(
    reason="Seeding for bnns does not work yet.. causes are to be found!",
    raises=AssertionError
)
def test_bnn_seed():
    n_nets = randint(2, 6)
    seed = randint(0, 2 ** 32 - 1)
    burn_in_steps = randint(0, 10)

    # Draw a random supported sampler
    """
    all_samplers = list(Sampler)
    sampler = choice(all_samplers)
    while not Sampler.is_supported(sampler):
        sampler = choice(all_samplers)
    """

    rng, n_datapoints = RandomState(randint(0, 10000)), 100
    x_train = asarray([rng.uniform(0., 1., 1) for _ in range(n_datapoints)])
    y_train = sinc(x_train)

    def bnn_chain():
        graph = tf.Graph()

        with tf.Session(graph=graph) as session:
            bnn = BayesianNeuralNetwork(
                n_nets=n_nets,
                session=session,
                seed=seed,
                burn_in_steps=burn_in_steps
            )
            bnn.train(x_train, y_train)
            assert bnn.is_trained

        return bnn.samples

    chain1, chain2 = bnn_chain(), bnn_chain()
    for sample1, sample2 in zip(chain1, chain2):
        for dimension1, dimension2 in zip(sample1, sample2):
            assert allclose(dimension1, dimension2, atol=1e-01)
