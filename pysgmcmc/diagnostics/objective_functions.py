import numpy as np


def sinc(x):
    return np.sinc(x * 10 - 5).sum(axis=1)
