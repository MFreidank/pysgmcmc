import numpy as np

# XXX: Give references for banana function (this is the one they use in their notebook!)
def banana_log_likelihood(x):
    return -1.0 / 20.0 * (100 * (x[1] - x[0]**2)**2 + (1 - x[0]) ** 2)


def sinc(x):
    return np.sinc(x * 10 - 5).sum(axis=1)
