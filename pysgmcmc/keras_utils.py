import typing
from contextlib import contextmanager
from numpy import prod
import tensorflow as tf
from keras import backend as K
from pysgmcmc.typing import KerasVariable, KerasTensor


def n_dimensions(tensors: typing.List[KerasTensor]) -> int:
    dimensions = sum(prod(K.int_shape(tensor)) for tensor in tensors)
    is_integer = dimensions % 1 == 0
    assert is_integer
    return int(dimensions)


def tensor_size(tensor: KerasTensor) -> int:
    return n_dimensions([tensor])


def to_vector(tensors: typing.List[KerasTensor]) -> tf.Tensor:
    backend = K.backend()
    if backend == "tensorflow":
        return tf.concat([K.reshape(tensor, [-1]) for tensor in tensors], axis=0)
    else:
        raise NotImplementedError(
            "to_vector does not yet support backend '{}'".format(backend)
        )


def keras_split(tensor: KerasTensor,
                num_or_size_splits: typing.Union[int, typing.Iterable[int]],
                axis: int=0,
                num=None,
                name: str='split') -> typing.List[tf.Tensor]:

    backend = K.backend()

    if backend == "tensorflow":
        return tf.split(
            tensor, num_or_size_splits, axis=axis, num=num, name=name
        )
    raise NotImplementedError(
        "tensor_size does not yet support backend '{}'".format(backend)
    )


def updates_for(params: typing.List[KerasVariable],
                update_tensor: KerasVariable) -> typing.List[KerasVariable]:
    param_sizes = tuple(tensor_size(param) for param in params)
    return keras_split(update_tensor, param_sizes, axis=0)


def safe_division(x: KerasTensor, y: KerasTensor, small_constant: float=1e-16):
    c = K.constant(small_constant)
    return x / (y + (2. * K.cast(K.sign(y), c.dtype) * c + c))


def safe_sqrt(x: KerasTensor, min_value: float=0., max_value: float=float("inf")):
    return K.sqrt(
        K.clip(x, min_value=min_value, max_value=max_value)
    )


def optimizer_name(optimizer: typing.Union[str, type]) -> str:
    if isinstance(optimizer, str):
        return optimizer
    return optimizer.__name__


@contextmanager
def keras_control_dependencies(control_inputs: typing.List[KerasTensor]):
    if K.backend() == "tensorflow":
        with tf.control_dependencies(control_inputs):
            yield
    else:
        yield
