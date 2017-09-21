import tensorflow as tf
import numpy as np

from pysgmcmc.models.bayesian_neural_network import LogVariancePrior, WeightPrior

from os.path import dirname, join as path_join, realpath

data_path = path_join(dirname(realpath(__file__)), "..", "data")
priors_path = path_join(data_path, "bayesian_neural_network_priors")


ground_truth = {
    "log_variance": path_join(priors_path, "log_variance.npy"),
    "weights": path_join(priors_path, "weights.npy"),
}


def test_log_variance_prior_log_likelihood():
    mean, var = 1e-6, 0.01

    prior = LogVariancePrior(mean=mean, var=var)

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
                prior.log_like(tf.constant(f_log_var, dtype=tf.float64))
            )
        )

    # load precomputed ground truth
    expected_value = np.load(ground_truth["log_variance"])

    assert(np.array_equal(result, expected_value))


def test_weight_prior_log_likelihood():
    prior = WeightPrior()

    # load inputs
    weight_inputs = np.load(path_join(priors_path, "weights_inputs.npy"))

    inputs = [
        tf.convert_to_tensor(
            np.ndarray.astype(p, dtype=np.float64), dtype=tf.float64
        )
        for p in weight_inputs
    ]

    with tf.Session() as session:
        init = tf.global_variables_initializer()
        session.run(init)
        result = np.array(session.run(prior.log_like(inputs)))

    # load precomputed ground truth
    expected_value = np.load(ground_truth["weights"])

    assert(np.array_equal(result, expected_value))
