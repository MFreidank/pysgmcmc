import tensorflow as tf

from numpy import allclose, asarray
from numpy.random import randint, RandomState

from pysgmcmc.diagnostics.objective_functions import sinc
from pysgmcmc.models.bayesian_neural_network import BayesianNeuralNetwork
import pysgmcmc
from pysgmcmc.sampling import Sampler
from random import choice

pysgmcmc.models.bayesian_neural_network.BayesianNeuralNetwork


def test_bnn_seed():
    n_nets = randint(2, 6)
    seed = randint(0, 2 ** 32 - 1)
    burn_in_steps = randint(0, 10)

    # Draw a random supported sampler
    all_samplers = list(Sampler)
    sampler = choice(all_samplers)
    while not Sampler.is_supported(sampler):
        sampler = choice(all_samplers)

    print("SAMPLER:", sampler)

    rng, n_datapoints = RandomState(randint(0, 10000)), 100
    x_train = asarray([rng.uniform(0., 1., 1) for _ in range(n_datapoints)])
    y_train = sinc(x_train)

    def bnn_chain():
        tf.reset_default_graph()
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
