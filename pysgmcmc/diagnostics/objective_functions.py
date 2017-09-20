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


# ## HPOLIB SYNTHETIC FUNCTIONS ## #

# XXX Merge/write doku

def bohachevski(x):
    y = 0.7 + x[0] ** 2 + 2.0 * x[1] ** 2
    y -= 0.3 * np.cos(3.0 * np.pi * x[0])
    y -= 0.4 * np.cos(4.0 * np.pi * x[1])
    return y


def branin(x):
        y = (x[1] - (5.1 / (4 * np.pi ** 2)) * x[0] ** 2 + 5 * x[0] / np.pi - 6) ** 2
        y += 10 * (1 - 1 / (8 * np.pi)) * np.cos(x[0]) + 10
        return x


def camelback(x):
    y = (4 - 2.1 * (x[0] ** 2) + ((x[0] ** 4) / 3)) * \
        (x[0] ** 2) + x[0] * x[1] + (-4 + 4 * (x[1] ** 2)) * (x[1] ** 2)
    return y


def goldstein_price(x):
        y = (1 + (x[0] + x[1] + 1) ** 2 * (19 - 14 * x[0] + 3 * x[0] ** 2 - 14 * x[1] + 6 * x[0] * x[1] + 3 * x[1] ** 2))\
            * (30 + (2 * x[0] - 3 * x[1]) ** 2 * (18 - 32 * x[0] + 12 * x[0] ** 2 + 48 * x[1] - 36 * x[0] * x[1] + 27 * x[1] ** 2))
        return y


def hartmann3(x):
    alpha = [1.0, 1.2, 3.0, 3.2]
    A = np.array([[3.0, 10.0, 30.0],
                  [0.1, 10.0, 35.0],
                  [3.0, 10.0, 30.0],
                  [0.1, 10.0, 35.0]])
    P = 0.0001 * np.array([[3689, 1170, 2673],
                           [4699, 4387, 7470],
                           [1090, 8732, 5547],
                           [381, 5743, 8828]])
    external_sum = 0
    for i in range(4):
        internal_sum = 0
        for j in range(3):
            internal_sum += A[i, j] * (x[j] - P[i, j]) ** 2
        external_sum += alpha[i] * np.exp(-internal_sum)
    return -external_sum


def hartmann6(x):
    alpha = [1.00, 1.20, 3.00, 3.20]
    A = np.array([[10.00, 3.00, 17.00, 3.50, 1.70, 8.00],
                  [0.05, 10.00, 17.00, 0.10, 8.00, 14.00],
                  [3.00, 3.50, 1.70, 10.00, 17.00, 8.00],
                  [17.00, 8.00, 0.05, 10.00, 0.10, 14.00]])
    P = 0.0001 * np.array([[1312, 1696, 5569, 124, 8283, 5886],
                           [2329, 4135, 8307, 3736, 1004, 9991],
                           [2348, 1451, 3522, 2883, 3047, 6650],
                           [4047, 8828, 8732, 5743, 1091, 381]])

    external_sum = 0
    for i in range(4):
        internal_sum = 0
        for j in range(6):
            internal_sum += A[i, j] * (x[j] - P[i, j]) ** 2
        external_sum += alpha[i] * np.exp(-internal_sum)
    return -external_sum


def levy(x):
    z = 1 + ((x[0] - 1.) / 4.)
    s = np.power((np.sin(np.pi * z)), 2)
    y = (s + ((z - 1) ** 2) * (1 + np.power((np.sin(2 * np.pi * z)), 2)))
    return y


def rosenbrock(x):
    y = 0
    d = 2
    for i in range(d - 1):
        y += 100 * (x[i + 1] - x[i] ** 2) ** 2
        y += (x[i] - 1) ** 2

    return y


def sin_one(x):
    y = 0.5 * np.sin(13 * x[0]) * np.sin(27 * x[0]) + 0.5
    return y


def sin_two(x):
    y = (0.5 * np.sin(13 * x[0]) * np.sin(27 * x[0]) + 0.5) * (0.5 * np.sin(13 * x[1]) * np.sin(27 * x[1]) + 0.5)
    return y
