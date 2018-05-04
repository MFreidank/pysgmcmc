#!/usr/bin/python3
# -*- coding: iso-8859-15 -*-
import numpy as np
import matplotlib.pyplot as plt
from torch.optim import Adam

from pysgmcmc.models.bayesian_neural_network import BayesianNeuralNetwork
from pysgmcmc.optimizers.sgld import SGLD
from pysgmcmc.optimizers.sghmc import SGHMC


def main():

    input_dimensionality, num_datapoints = 1, 100
    x_train = np.array([
        np.random.uniform(np.zeros(1), np.ones(1), input_dimensionality)
        for _ in range(num_datapoints)
    ])
    y_train = np.sinc(x_train * 10 - 5).sum(axis=1)

    x_test = np.linspace(0, 1, 100)[:, None]
    y_test = np.sinc(x_test * 10 - 5).sum(axis=1)

    optimizer = SGLD
    import logging
    bnn = BayesianNeuralNetwork(optimizer=optimizer, logging_configuration={"level": logging.INFO})
    prediction, variance_prediction = bnn.train(x_train, y_train).predict(x_test)

    prediction_std = np.sqrt(variance_prediction)

    plt.grid()

    plt.plot(x_test[:, 0], y_test, label="true", color="black")
    plt.plot(x_train[:, 0], y_train, "ro")

    plt.plot(x_test[:, 0], prediction, label=optimizer.__name__, color="blue")
    plt.fill_between(x_test[:, 0], prediction + prediction_std, prediction - prediction_std, alpha=0.2, color="indianred")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
