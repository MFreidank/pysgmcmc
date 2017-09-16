from scipy.misc import logsumexp
import numpy as np


# XXX: Give references for banana function (this is the one they use in their notebook!)
def banana_log_likelihood(x):
    return -1.0 / 20.0 * (100 * (x[1] - x[0]**2)**2 + (1 - x[0]) ** 2)


def gaussian_mixture_model_log_likelihood(x, mu=(-5, 0, 5), var=(1., 1., 1.),
                                          weights=(1. / 3., 1. / 3., 1. / 3.)):
    assert(len(mu) == len(var) == len(weights))

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


def sinc(x):
    return np.sinc(x * 10 - 5).sum(axis=1)
