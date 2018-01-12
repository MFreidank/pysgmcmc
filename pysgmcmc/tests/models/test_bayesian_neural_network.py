from random import choice
from random import randint

import numpy as np

from pysgmcmc.tests.models.objectives import OBJECTIVES
from pysgmcmc.models.bayesian_neural_network import BayesianNeuralNetwork


def test_sinc():
    n_datapoints = randint(20, 100)
    batch_size = randint(10, 100)
    seed = randint(0, 2**32 - 1)

    sinc, data_generator = OBJECTIVES["sinc"]

    x_train = data_generator(n_datapoints)
    y_train = sinc(x_train)
    x_test = np.linspace(0, 1, 100)[:, None]
    y_test = sinc(x_test)

    bayesian_neural_network = BayesianNeuralNetwork(
        batch_size=batch_size,
        seed=seed,
        learning_rate=0.01,
    )
    bayesian_neural_network.train(x_train, y_train)
    bayesian_neural_network.predict(x_test)

    # XXX: Compute score and assert it here
    assert True


def test_multiple_train():
    n_repetitions = randint(2, 10)
    seed = randint(0, 2 ** 32 - 1)
    objective = choice(tuple(OBJECTIVES.keys()))

    objective_function, data_generator = OBJECTIVES[objective]

    n_datapoints = 10

    x_train = data_generator(n_datapoints)
    y_train = objective_function(x_train)

    # Train each net only briefly, we simply want to make sure training
    # multiple times works without error
    bayesian_neural_network = BayesianNeuralNetwork(
        n_nets=3,
        burn_in_steps=10,
        seed=seed,
        learning_rate=0.01,
    )
    for _ in range(n_repetitions):
        # run a full train loop
        bayesian_neural_network.train(x_train, y_train)
