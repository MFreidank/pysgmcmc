from contextlib import contextmanager
import tensorflow as tf
from keras import backend as K


def vectorized_shape(original_shape):
    n_elements = K.prod(original_shape)
    return (n_elements, 1)


def vectorize(tensor):
    new_shape = vectorized_shape(tensor.shape)
    return K.reshape(tensor, shape=new_shape)


def to_vector(tensors):
    backend = K.backend()
    if backend == "tensorflow":
        return tf.concat([K.reshape(tensor, [-1]) for tensor in tensors], axis=0)
    else:
        raise NotImplementedError(
            "to_vector does not yet support backend '{}'".format(backend)
        )


def tensor_size(tensor):
    return n_dimensions([tensor])


def keras_split(tensor, num_or_size_splits, axis=0, num=None, name='split'):
    backend = K.backend()
    if backend == "tensorflow":
        print("SPLITTING", num_or_size_splits)
        return tf.split(
            tensor, num_or_size_splits, axis=axis, num=num, name=name
        )
    raise NotImplementedError(
        "tensor_size does not yet support backend '{}'".format(backend)
    )


def updates_for(params, update_tensor):
    param_sizes = tuple(tensor_size(param) for param in params)

    param_updates = keras_split(update_tensor, param_sizes, axis=0)
    # XXX: This methods needs to be finished
    raise NotImplementedError("Finish updates for method like in sghdhd towards the bottom")


def n_dimensions(tensors):
    from numpy import prod
    return sum(prod(K.int_shape(tensor)) for tensor in tensors)


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
