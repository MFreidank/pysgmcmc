# vim:foldmethod=marker
import abc
from scipy.misc import logsumexp
import tensorflow as tf
import numpy as np
from keras import backend as K
import functools
import matplotlib.pyplot as plt
from matplotlib.mlab import bivariate_normal


def apply_on_grid(logdensity_function, x, y):
    xx, yy = np.meshgrid(x, y, sparse=True)
    densities = np.asarray([np.exp(logdensity_function((x, y))) for x in xx for y in yy])
    return densities


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


class TwoDimensionalDistribution(object, metaclass=abc.ABCMeta):
    # XXX: Make label show in same color!
    def plot(self,
             grid=(np.arange(-4, 4, 0.05), np.arange(-4, 4, 0.05)),
             ax=None,
             title=None):
        x, y = grid
        densities = apply_on_grid(self.__call__, x, y)
        if ax is None:
            f, ax = plt.subplots(1)
        ax.contour(x, y, densities)
        if title is not None:
            ax.set_title(title)
            ax.legend()
        return ax


#  Distributions from "Relativistic Monte Carlo" paper; https://arxiv.org/abs/1609.04388 {{{ #
class Banana(TwoDimensionalDistribution):
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
        ax = super().plot(
            grid=(np.arange(-25, 25, 0.05), np.arange(-50, 20, 0.05)),
            title="Banana"
        )

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
                pi = tf.constant(np.pi, dtype=K.floatx())
                return -0.5 * tf.log(2.0 * pi * var) - 0.5 * ((x - mu) ** 2) / var

            return tf.reduce_logsumexp([
                tf.log(K.constant(self.weights[i], dtype=K.floatx())) + normldf_tf(x, K.constant(self.mu[i], dtype=K.floatx()), K.constant(self.var[i], dtype=K.floatx()))
                for i in range(len(self.mu))
            ])
        else:
            def normldf(x, mu, var):
                return -0.5 * np.log(2.0 * np.pi * var) - 0.5 * ((x - mu) ** 2) / var

            return logsumexp([
                np.log(self.weights[i]) + normldf(x, self.mu[i], self.var[i])
                for i in range(len(self.mu))
            ])

    def plot(self, ax=None, x=np.linspace(-10, 10, num=1000)):
        if ax is None:
            f, ax = plt.subplots()
        ax.plot(x, np.asarray([np.exp(self.__call__(x_)) for x_ in x]))
        return ax


class Gmm1(GaussianMixture):
    def plot(self, ax=None):
        ax = super().plot()
        ax.set_title("GMM1")
        return ax


class Gmm2(GaussianMixture):
    def __init__(self):
        super().__init__(var=[1. / 0.5, 0.5, 1. / 0.5])

    def plot(self, ax=None):
        ax = super().plot()
        ax.set_title("GMM2")
        return ax


class Gmm3(GaussianMixture):
    def __init__(self):
        super().__init__(var=[1. / 0.3, 0.3, 1. / 0.3])

    def plot(self, ax=None):
        ax = super().plot()
        ax.set_title("GMM3")
        return ax
#  }}} Distributions from "Relativistic Monte Carlo" paper; https://arxiv.org/abs/1609.04388 #

# XXX: Add more from that paper here.
#  Ported from L2HMC paper; https://arxiv.org/abs/1711.09268 {{{ #


class MoGL2HMC(GaussianMixture):
    def __init__(self):
        super().__init__(var=(0.1, 0.1), mu=(0., 4.), weights=(1. / 2., 1. / 2.))

    def plot(self, ax=None):
        ax = super().plot(ax=ax, x=np.linspace(-2., 6., num=1000))
        ax.set_title("MoG_L2HMC")

#  }}} Ported from L2HMC paper; https://arxiv.org/abs/1711.09268 #

#  Ported from mcmc-demo; https://github.com/chi-feng/mcmc-demo/blob/master/main/MCMC.js  {{{ #


class BivariateNormal(TwoDimensionalDistribution):
    def __init__(self, mu, cov):
        self.mu, self.cov = np.asarray(mu), np.asarray(cov)
        assert self.cov[0, 1] == self.cov[1, 0]
        assert len(self.mu) == 2
        assert self.cov.shape == (2, 2)

    def __call__(self, x):
        return bivariate_normal(
            *x,
            mux=self.mu[0], muy=self.mu[1],
            sigmax=self.cov[0, 0], sigmay=self.cov[1, 1],
            sigmaxy=self.cov[0, 1]
        )
        # logdensity
        from scipy.stats import multivariate_normal
        return np.asarray([
            multivariate_normal.logpdf(x_, mean=self.mu, cov=self.cov)
            for x_ in zip(*x)
        ])
        return multivariate_normal.logpdf(x, mean=self.mu, cov=self.cov)


class StandardNormal(BivariateNormal):
    def __init__(self):
        super().__init__([0, 0], [[1, 0], [0, 1]])

    def plot(self):
        return super().plot(title="StandardNormal")


class MultiModalBivariateNormal(TwoDimensionalDistribution):
    def __init__(self,
                 means=[[-1.5, -1.5], [1.5, 1.5], [-2., 2.]],
                 covariances=[[[0.8, 0.], [0., 0.8]],
                              [[0.8, 0.], [0., 0.8]],
                              [[0.5, 0.], [0., 0.5]]]):
        assert len(means) == len(covariances), (len(means), len(covariances))

        self.mixture_components = [
            BivariateNormal(mu=mu, cov=cov)
            for mu, cov in zip(means, covariances)
        ]

    def __call__(self, x):
        return np.log(
            sum(np.exp(component(x)) for component in self.mixture_components)
        )

    def plot(self):
        f, ax = plt.subplots()
        for component in self.mixture_components:
            ax = component.plot(
                ax=ax, grid=(np.arange(-4, 4, 0.05), np.arange(-4, 4, 0.05))
            )
            ax.set_title("MultiModalNormal")
        return ax


class Donut(TwoDimensionalDistribution):
    def __init__(self, radius=2.6, sigma2=0.033):
        self.radius, self.sigma2 = radius, sigma2

    def __call__(self, x):
        # log density
        # FIXME Add tensorflow implementation.
        r = np.linalg.norm(x)
        v = -((r - self.radius) ** 2) / self.sigma2
        return v

    def plot(self,):
        return super().plot(
            grid=(np.arange(-4, 4, 0.05), np.arange(-4, 4, 0.05)),
            title="Donut",
        )


class Squiggle(BivariateNormal):
    def __init__(self):
        super().__init__(mu=[[0], [0]], cov=[[2, 0.25], [0.25, 0.5]])

    def __call__(self, x):
        y = np.asarray([x[0], x[1] + np.sin(5 * x[0])])
        return super().__call__(y)

    def plot(self, ax=None):
        return super().plot(
            grid=(np.arange(-2, 2, 0.05), np.arange(-2, 2, 0.05)),
            title="Squiggle"
        )
#  }}} Ported from mcmc-demo; https://github.com/chi-feng/mcmc-demo/blob/master/main/MCMC.js  #
