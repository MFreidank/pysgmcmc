#!/usr/bin/python3
# -*- coding: iso-8859-15 -*-
import logging

import matplotlib.pyplot as plt
import numpy as np
from torch.optim import Adam

from pysgmcmc.optimizers.sghmc import SGHMC
from pysgmcmc.models.bayesian_neural_network import BayesianNeuralNetwork as BNN
from pysgmcmc.diagnostics.objective_functions import sinc
from pysgmcmc.tests.utils import init_random_uniform

logging.basicConfig(level=logging.INFO)


def main():
    num_datapoints = 200
    batch_size = 20

    X = init_random_uniform(
        lower=np.zeros(1), upper=np.ones(1), n_points=num_datapoints
    )

    y = sinc(X)

    x_test = np.linspace(0, 1, 100)[:, None]
    y_test = sinc(x_test)

    optimizer = Adam
    # optimizer = SGHMC

    bnn = BNN(
        batch_size=batch_size,
        optimizer=optimizer,  # scale_grad=num_datapoints, lr=1e-2,
    )

    bnn.train(x_train=X, y_train=y)

    mean_prediction, variance_prediction = bnn.predict(x_test=x_test)

    prediction_std = np.sqrt(variance_prediction)

    plt.grid()

    plt.plot(x_test[:, 0], y_test, label="true", color="black")
    plt.plot(X[:, 0], y, "ro")

    plt.plot(x_test[:, 0], mean_prediction, label=optimizer.__name__, color="blue")
    plt.fill_between(x_test[:, 0], mean_prediction + prediction_std, mean_prediction - prediction_std, alpha=0.2, color="blue")

    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
