import tensorflow as tf
import numpy as np

from pysgmcmc.models.bayesian_neural_network import (
    log_variance_prior_log_like, weight_prior_log_like
)

from os.path import dirname, join as path_join, realpath

DATA_PATH = path_join(dirname(realpath(__file__)), "..", "data")
PRIORS_PATH = path_join(DATA_PATH, "bayesian_neural_network_priors")


ground_truth = {
    "log_variance": path_join(PRIORS_PATH, "log_variance.npy"),
    "weights": path_join(PRIORS_PATH, "weights.npy"),
}


def test_log_variance_prior_log_likelihood():
    mean, var = 1e-6, 0.01

    # intermediate f_log_var value stolen from a BNN run
    f_log_var = [[-11.25474104],
                 [-11.25474104],
                 [-11.25474104],
                 [-11.25474104],
                 [-11.25474104],
                 [-11.25474104],
                 [-11.25474104],
                 [-11.25474104],
                 [-11.25474104],
                 [-11.25474104],
                 [-11.25474104],
                 [-11.25474104],
                 [-11.25474104],
                 [-11.25474104],
                 [-11.25474104],
                 [-11.25474104],
                 [-11.25474104],
                 [-11.25474104],
                 [-11.25474104],
                 [-11.25474104]]

    # compute result
    with tf.Session() as session:
        result = np.array(
            session.run(
                log_variance_prior_log_like(
                    tf.constant(f_log_var, dtype=tf.float64),
                    mean=mean, var=var, dtype=tf.float64
                )
            )
        )

    # load precomputed ground truth
    expected_value = np.load(ground_truth["log_variance"])

    assert np.array_equal(result, expected_value)


def test_weight_prior_log_likelihood():
    # load inputs
    weight_inputs = np.load(path_join(PRIORS_PATH, "weights_inputs.npy"))

    inputs = [
        tf.convert_to_tensor(
            np.ndarray.astype(p, dtype=np.float64), dtype=tf.float64
        )
        for p in weight_inputs
    ]

    with tf.Session() as session:
        init = tf.global_variables_initializer()
        session.run(init)
        result = np.array(session.run(weight_prior_log_like(inputs)))

    # load precomputed ground truth
    expected_value = np.load(ground_truth["weights"])

    assert np.array_equal(result, expected_value)
