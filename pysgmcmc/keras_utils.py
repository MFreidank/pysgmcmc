from contextlib import contextmanager
import tensorflow as tf
from keras import backend as K


def vectorize(tensor):
    def vectorized_shape(tensor):
        # Compute vectorized shape
        n_elements = K.prod(tensor.shape)
        return (n_elements, 1)

    return K.reshape(tensor, shape=vectorized_shape(tensor))


def safe_division(x, y, small_constant=1e-16):
    c = K.constant(small_constant)
    return x / (y + (2. * K.cast(K.sign(y), c.dtype) * c + c))


def safe_sqrt(x, min_value=0., max_value=float("inf")):
    return K.sqrt(
        K.clip(x, min_value=min_value, max_value=max_value)
    )


def optimizer_name(optimizer):
    if isinstance(optimizer, str):
        return optimizer
    return optimizer.__name__


@contextmanager
def keras_control_dependencies(control_inputs):
    if K.backend() == "tensorflow":
        with tf.control_dependencies(control_inputs):
            yield
    else:
        yield
