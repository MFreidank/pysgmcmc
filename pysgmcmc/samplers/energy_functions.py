from scipy.misc import logsumexp
import tensorflow as tf
import numpy as np
import functools


def to_negative_log_likelihood(log_likelihood_function):
    """ Decorator that converts a log likelihood into a negative log likelihood.
        Callable `log_likelihood_function` represents the log likelihood
        and a callable `negative_log_likelihood_function` with the same
        signature is returned.

    Parameters
    ----------
    log_likelihood_function : callable
        Callable that represents a log likelihood function.

    Returns
    -------
    negative_log_likelihood_function : callable
        Callable that returns negative log likelihood.

    Examples
    -------
    Wrapping a dummy log likelihood:

    >>> import numpy as np
    >>> log_likelihood = lambda a, b: np.log(a + b)
    >>> negative_log_likelihood = to_negative_log_likelihood(log_likelihood)
    >>> input_a, input_b = 4, 5
    >>> ll = log_likelihood(input_a, input_b)
    >>> nll = negative_log_likelihood(input_a, input_b)
    >>> np.allclose(-ll, nll)
    True

    The name attribute of the wrapped function remains unchanged:

    >>> log_likelihood.__name__ == negative_log_likelihood.__name__
    True

    """
    @functools.wraps(log_likelihood_function)
    def negative_log_likelihood(*args, **kwargs):
        return -log_likelihood_function(*args, **kwargs)
    return negative_log_likelihood


# XXX Give references: Relativistic Monte Carlo
def banana_log_likelihood(x):
    """
    Examples
    ----------

    >>> optimum, f_opt = (0, 10), 0.
    >>> np.allclose(banana_log_likelihood(optimum), f_opt)
    True

    """
    return -0.5 * (0.01 * x[0] ** 2 + (x[1] + 0.1 * x[0] ** 2 - 10) ** 2)


def gaussian_mixture_model_log_likelihood(x, mu=(-5, 0, 5), var=(1., 1., 1.),
                                          weights=(1. / 3., 1. / 3., 1. / 3.)):
    assert len(mu) == len(var) == len(weights)

    if hasattr(x, "__iter__"):
        assert(len(x) == 1)
        x = x[0]

    if isinstance(x, tf.Variable):
        def normldf_tf(x, mu, var):
            pi = tf.constant(np.pi)
            return -0.5 * tf.log(2.0 * pi * var) - 0.5 * ((x - mu) ** 2) / var

        return tf.reduce_logsumexp(
            [tf.log(weights[i]) + normldf_tf(x, mu[i], var[i]) for i in range(len(mu))]
        )
    else:
        def normldf(x, mu, var):
            return -0.5 * np.log(2.0 * np.pi * var) - 0.5 * ((x - mu) ** 2) / var

        return logsumexp([
            np.log(weights[i]) + normldf(x, mu[i], var[i])
            for i in range(len(mu))
        ])


# XXX: Give references for gmm functions (relativistic monte carlo paper)
def gmm1_log_likelihood(x):
    return gaussian_mixture_model_log_likelihood(x)


def gmm2_log_likelihood(x):
    return gaussian_mixture_model_log_likelihood(x, var=[1. / 0.5, 0.5, 1. / 0.5])


def gmm3_log_likelihood(x):
    return gaussian_mixture_model_log_likelihood(x, var=[1. / 0.3, 0.3, 1. / 0.3])
