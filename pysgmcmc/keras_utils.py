import numpy as np
from keras import backend as K


def vectorize(tensor):
    def vectorized_shape(tensor):
        # Compute vectorized shape
        n_elements = K.prod(tensor.shape)
        return (n_elements, 1)

    return K.reshape(tensor, shape=vectorized_shape(tensor))


def safe_division(x, y, small_constant=0.):
    return x / (y + small_constant)


def optimizer_name(optimizer):
    if isinstance(optimizer, str):
        return optimizer
    return optimizer.__name__
