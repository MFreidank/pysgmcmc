from scipy.misc import logsumexp
import tensorflow as tf
import numpy as np
import functools
import matplotlib.pyplot as plt


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


class Banana(object):
    def __call__(self, x):
        """
        Examples
        ----------

        >>> optimum, f_opt = (0, 10), 0.
        >>> np.allclose(Banana()(optimum), f_opt)
        True

        """
        return -0.5 * (0.01 * x[0] ** 2 + (x[1] + 0.1 * x[0] ** 2 - 10) ** 2)

    def plot(self):
        x = np.arange(-25, 25, 0.05)
        y = np.arange(-50, 20, 0.05)
        xx, yy = np.meshgrid(x, y, sparse=True)
        densities = np.asarray([np.exp(self.__call__((x, y))) for x in xx for y in yy])
        f, ax = plt.subplots(1)
        ax.contour(x, y, densities, 1, label="Banana")
        ax.plot([], [], label="Banana")
        ax.legend()
        ax.grid()

        ax.set_ylim(ymin=-60, ymax=20)
        ax.set_xlim(xmin=-30, xmax=30)
        return ax


class GaussianMixture(object):
    def __init__(self, mu=(-5., 0., 5.), var=(1., 1., 1.), weights=(1. / 3., 1. / 3., 1. / 3.)):
        assert len(mu) == len(var) == len(weights)
        self.mu, self.var, self.weights = mu, var, weights

    def __call__(self, x):
        if hasattr(x, "__iter__"):
            assert(len(x) == 1)
            x = x[0]

        if isinstance(x, tf.Variable):
            def normldf_tf(x, mu, var):
                pi = tf.constant(np.pi)
                return -0.5 * tf.log(2.0 * pi * var) - 0.5 * ((x - mu) ** 2) / var

            return tf.reduce_logsumexp([
                tf.log(self.weights[i]) + normldf_tf(x, self.mu[i], self.var[i])
                for i in range(len(self.mu))
            ])
        else:
            def normldf(x, mu, var):
                return -0.5 * np.log(2.0 * np.pi * var) - 0.5 * ((x - mu) ** 2) / var

            return logsumexp([
                np.log(self.weights[i]) + normldf(x, self.mu[i], self.var[i])
                for i in range(len(self.mu))
            ])

    def plot(self, ax=None):
        x = np.linspace(-10, 10, num=1000)
        y = np.asarray([self.__call__(x_) for x_ in x])
        plt.scatter(x, y)


class Gmm1(GaussianMixture):
    pass


class Gmm2(GaussianMixture):
    def __init__(self):
        super().__init__(var=[1. / 0.5, 0.5, 1. / 0.5])


class Gmm3(GaussianMixture):
    def __init__(self):
        super().__init__(var=[1. / 0.3, 0.3, 1. / 0.3])
