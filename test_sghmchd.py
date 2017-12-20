#!/usr/bin/python3
# -*- coding: iso-8859-15 -*-
import numpy as np
import matplotlib.pyplot as plt
from keras.callbacks import TensorBoard

from pysgmcmc.diagnostics.objective_functions import sinc
from pysgmcmc.optimizers.sghmchd import SGHMCHD
from pysgmcmc.models.bayesian_neural_network import (
    BayesianNeuralNetwork
)
from pysgmcmc.keras_utils import optimizer_name


def init_random_uniform(lower, upper, n_points, rng=None):
    if rng is None:
        rng = np.random.RandomState(np.random.randint(0, 10000))

    n_dims = lower.shape[0]

    return np.array(
        [rng.uniform(lower, upper, n_dims) for _ in range(n_points)]
    )


def main():
    import logging
    # from tensorflow.python import debug as tf_debug
    from keras import backend as K
    logging.basicConfig(level=logging.DEBUG)

    sess = K.get_session()
    # sess = tf_debug.LocalCLIDebugWrapperSession(sess)
    # sess.add_tensor_filter('has_nan', has_nan)
    K.set_session(sess)

    # optimizer = SGHMC
    optimizer = SGHMCHD

    sghmc_arguments = {"learning_rate": 0.000000001}

    n_datapoints = 100
    x_train = init_random_uniform(
        lower=np.zeros(1), upper=np.ones(1), n_points=n_datapoints
    )

    y_train = sinc(x_train)

    x_train = x_train[:1000]
    y_train = y_train[:1000]

    x_test = np.linspace(0, 1, 100)[:, None]
    y_test = sinc(x_test)

    cb = TensorBoard(
        log_dir="./logs", histogram_freq=0, write_graph=True, write_grads=True,
    )

    burn_in_steps = 3000
    model = BayesianNeuralNetwork(
        burn_in_steps=burn_in_steps,
        n_steps=burn_in_steps + 50000,
        train_callbacks=[cb],
        optimizer=optimizer, **sghmc_arguments
    )
    model.train(x_train, y_train)

    prediction_mean, prediction_variance = model.predict(x_test)

    prediction_std = np.sqrt(prediction_variance)

    plt.grid()

    plt.plot(x_test[:, 0], y_test, label="true", color="black")
    plt.plot(x_train[:, 0], y_train, "ro")

    plt.plot(x_test[:, 0], prediction_mean, label=optimizer_name(optimizer), color="blue")
    plt.fill_between(x_test[:, 0], prediction_mean + prediction_std, prediction_mean - prediction_std, alpha=0.2, color="blue")

    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
