from functools import wraps
import typing
import re
from contextlib import contextmanager
import sympy
from numpy import prod
import tensorflow as tf
from keras import backend as K
from pysgmcmc.custom_typing import KerasVariable, KerasTensor

FLOAT_DTYPE = K.floatx()
PRECISION = re.search("[0-9]+", FLOAT_DTYPE).group()
INTEGER_DTYPE = "int{}".format(PRECISION)


class UnsupportedBackendError(NotImplementedError):
    """ Raised if a given callable does not support the current keras backend. """
    def __init__(self, callable_name, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data = {"callable": callable_name, "backend": K.backend()}

    def __str__(self):
        return "{callable} does not yet support backend '{backend}'".format(**self.data)


def supports_backends(supported_backends: typing.Iterable[str]):
    """ Decorator that ensures that only `supported_backends`
        can be used with a decorated function.
        Raises `pysgmcmc.keras_utils.UnsupportedBackendError`
        if a decorated function is called with an unsupported keras backend.

    Parameters
    ----------
    supported_backends: typing.Iterable[str]
        List of strings specifying keras backends that a given function supports.

    Returns
    ----------
    TODO DOKU FOR DECORATOR?

    """
    def backend_decorator(function):
        @wraps(function)
        def wrapped(*args, **kwargs):
            if K.backend() not in supported_backends:
                raise UnsupportedBackendError(function.__name__)
            return function(*args, **kwargs)
        return wrapped
    return backend_decorator


def n_dimensions(tensors: typing.List[KerasTensor]) -> int:
    """ Compute total number of dimensions of all given tensors.
        Total dimensions is sum of the number of elements of all tensors.

    Parameters
    ----------
    tensors: typing.List[KerasTensor]
        Tensors of any supported backend for keras.

    Returns
    ----------
    dimensions: int
        Total number of individual elements/dimensions of all tensors.
        Corresponds to the sum of the product of all individual tensor shapes.

    Examples
    ----------
    TODO

    """
    dimensions = sum(prod(K.int_shape(tensor)) for tensor in tensors)

    is_integer = dimensions % 1 == 0
    assert is_integer

    return int(dimensions)


def tensor_size(tensor: KerasTensor) -> int:
    """ Size of a given tensor, corresponds to its number of individual elements.
        Shorthand for `n_dimensions([tensor])`.

    Parameters
    ----------
    tensor: KerasTensor
        A tensor of any supported backend for keras.

    Returns
    ----------
    size: int
        Total number of individual elements/dimensions of this tensor.

    Examples
    ----------
    TODO

    """
    return n_dimensions([tensor])


@supports_backends({"tensorflow"})
def to_vector(tensors: typing.List[KerasTensor]) -> tf.Tensor:
    """ Assemble all given tensors into one large vector by concatenating their elements.

    Parameters
    ----------
    tensors: typing.List[KerasTensor]
        Tensors of any supported backend for keras.

    Returns
    ----------
    tensors_vector: KerasTensor
        Tensor of shape (N,) where N is the number of dimensions of `tensors`.
        Results from reshaping all tensors to shape (D,) and then concatenating
        those reshaped vectors.

    Examples
    ----------
    TODO

    """
    return tf.concat([K.reshape(tensor, [-1]) for tensor in tensors], axis=0)


@supports_backends({"tensorflow"})
def keras_split(tensor: KerasTensor,
                num_or_size_splits: typing.Union[int, typing.Iterable[int]],
                axis: int=0,
                name: str="split") -> typing.List[tf.Tensor]:
    """ TODO DOKU FROM TF

    Parameters
    ----------
    tensor: KerasTensor
        Tensor to split into smaller subtensors.

    num_or_size_splits: typing.Union[int : TODO
        TODO: Copy doku from tensorflow

    axis: int, optional
        TODO: Copy doku from tensorflow
        Defaults to `0`.

    name: str, optional
        TODO: Copy doku from tensorflow
        Defaults to `"split"`.

    Returns
    ----------
    subtensors: typing.List[KerasTensor]
        TODO DOKU FROM TF

    Examples
    ----------
    TODO

    """
    return tf.split(
        tensor, num_or_size_splits, axis=axis, name=name
    )


@supports_backends({"tensorflow"})
def updates_for(parameters: typing.List[KerasVariable],
                update_tensor: KerasTensor) -> typing.List[KerasTensor]:
    """ Split appropriately shaped sub-tensors from `update_tensor` to assign to all `parameters`.

    Parameters
    ----------
    params: typing.List[KerasVariable]
        List of variables for which we would like to compute updates

    update_tensor: KerasTensor
        Tensor from which we split out updates for each parameter in params.

    Returns
    ----------
    update_values: typing.List[KerasTensor]
        Values that have an appropriate shape to assign to their corresponding
        (by index) variable. These do not yet stem from an update operation
        and need to be assigned to `parameters` to have an effect.

    Examples
    ----------
    TODO

    """
    param_sizes = tuple(tensor_size(parameter) for parameter in parameters)
    return keras_split(update_tensor, param_sizes, axis=0)


def safe_division(x: KerasTensor, y: KerasTensor, small_constant: float=1e-16):
    """ Computes `x / y` after adding a small appropriately signed constant to `y`.
        Adding a small constant avoids division-by-zero artefacts that may
        occur due to precision errors.

    Parameters
    ----------
    x: KerasTensor
        Left-side operand of division.
    y: KerasTensor
        Right-side operand of division.
    small_constant: float, optional
        Small constant to add to/subtract from `y` before computing `x / y`.
        Defaults to `1e-16`.

    Returns
    ----------
    division_result : KerasTensor
        Result of `x / y` after adding a small appropriately signed constant
        to `y` to avoid division by zero.

    Examples
    ----------

    Will safely avoid divisions-by-zero under normal circumstances:

    >>> from keras import backend as K
    >>> import numpy as np
    >>> x = K.constant(1.0)
    >>> inf_tensor = x / 0.0  # will produce "inf" due to division-by-zero
    >>> np.isinf(K.get_value(inf_tensor))
    True
    >>> z = safe_division(x, 0., small_constant=1e-16)  # will avoid division-by-zero
    >>> np.isinf(K.get_value(z))
    False

    To see that simply adding a positive constant may fail, consider the
    following example. Note that this function handles such corner cases correctly:

    >>> from keras import backend as K
    >>> import numpy as np
    >>> x, y = K.constant(1.0), K.constant(-1e-16)
    >>> small_constant = 1e-16
    >>> inf_tensor = x / (y + small_constant)  # simply adding constant exhibits division-by-zero
    >>> np.isinf(K.get_value(inf_tensor))
    True
    >>> z = safe_division(x, y, small_constant=1e-16)  # will avoid division-by-zero
    >>> np.isinf(K.get_value(z))
    False

    """
    c = K.constant(small_constant, dtype=FLOAT_DTYPE)
    return x / (y + (2. * K.cast(K.sign(y), FLOAT_DTYPE) * c + c))


def optimizer_name(optimizer: typing.Union[str, type]) -> str:
    if isinstance(optimizer, str):
        return optimizer
    return optimizer.__name__


@contextmanager
def keras_control_dependencies(control_inputs: typing.List[KerasTensor]):
    """ Allows specifying control dependencies for keras tensors.
        Equivalent to `tf.control_dependencies`, and a no-op for
        other backends.
        NOTE: This implies that this operation fixes control order only for
        tensorflow, for other backends it does not do anything.

    Parameters
    ----------
    control_dependencies: typing.List[KerasTensor]
        TODO: DOKU FROM TENSORFLOW

    Examples
    ----------
    TODO

    """
    if K.backend() == "tensorflow":
        with tf.control_dependencies(control_inputs):
            yield
    else:
        yield


@supports_backends(("tensorflow",))
def sympy_to_keras(sympy_expression: sympy.expr.Expr,
                   sympy_tensors: typing.Tuple[sympy.Symbol, ...],
                   keras_tensors: typing.Tuple[KerasTensor, ...]) -> KerasTensor:
    """ Compute a given sympy expression using keras.
        Replace `sympy_tensors` with their corresponding `keras_tensors`
        and produce a keras tensor representing the result of the computation.

    Parameters
    ----------
    sympy_expression: sympy.expr.Exp
        Expression in a sympy graph that we want to compute.

    sympy_tensors: typing.Tuple[sympy.Symbol, ...]
        Tuple of (all) sympy tensors that `sympy_expression` depends on - explicitly or implictly.

    keras_tensors: typing.Tuple[KerasTensor, ...]
        Tuple of keras tensors to use in place of the corresponding (by index)
        sympy tensor in `sympy_tensors`.

    Returns
    ----------
    keras_result: KerasTensor
        Result of evaluating `sympy_expression` using `keras_tensors` as values for `sympy_tensors`.

    Examples
    ----------

    Using this function, it is straightforward to compute simple sympy
    expressions using keras/tensorflow:

    >>> from keras import backend as K
    >>> import sympy
    >>> (a, b), (a_tensor, b_tensor) = sympy.symbols("a b"), (K.constant(1.0), K.constant(2.0))
    >>> sympy_expression = a + b
    >>> K.get_value(sympy_to_keras(sympy_expression, (a, b), (a_tensor, b_tensor)))
    3.0

    One use-case where this comes in handy is when trying to access a complicated
    derivative in keras. Keras backends do not provide access to actual
    full symbolic gradients but only to a sum of gradients.
    In some cases, this information is not enough and using sympy to obtain
    a full symbolic derivative can be useful.

    To this end, we can construct a sympy graph for the symbolic derivative:

    >>> import sympy
    >>> a, b = sympy.symbols("a b")
    >>> c = a * (a ** 2 + b) - a ** 2 * b ** 2   # complex term whose full derivative we are interested in
    >>> derivative_sympy = sympy.diff(c, a)
    >>> derivative_sympy
    3*a**2 - 2*a*b**2 + b

    Next, we can query this full symbolic derivative from keras for
    any keras tensors we want. Note that it can be handy to keep track of
    tensor correspondences in a `collections.OrderedDict`:

    >>> from keras import backend as K
    >>> from collections import OrderedDict
    >>> a_tensor, b_tensor = K.constant([3.0, 5.0]), K.constant([4.0, 12.0])
    >>> tensor_correspondences = OrderedDict(((a, a_tensor), (b, b_tensor)))
    >>> derivative_keras = sympy_to_keras(derivative_sympy, tensor_correspondences.keys(), tensor_correspondences.values())
    >>> K.get_value(derivative_keras)
    array([  -65., -1353.], dtype=float32)

    """
    assert len(sympy_tensors) == len(keras_tensors)

    lambdified_function = sympy.lambdify(
        args=sympy_tensors, expr=sympy_expression, modules=K.backend()
    )
    return lambdified_function(*keras_tensors)


@supports_backends(("tensorflow",))
def while_loop(condition, body, loop_variables, shape_invariants=None, parallel_iterations=10):
    if K.backend() == "tensorflow":
        return tf.while_loop(
            cond=condition,
            body=body,
            loop_vars=loop_variables,
            shape_invariants=shape_invariants,
            parallel_iterations=parallel_iterations
        )

    raise UnsupportedBackendError(while_loop.__name__)


def logical_and(x: KerasTensor, y: KerasTensor) -> KerasTensor:
    """ Returns the truth value of x AND y element-wise.

    Parameters
    ----------
    x: KerasTensor
        A tensor of type `bool`.
    y: KerasTensor
        A tensor of type `bool`.

    Returns
    ----------
    and_tensor: KerasTensor
        Tensor of type `bool`.


    Examples
    ----------

    For two conditions that are truthy, `logical_and` returns `True`:

    >>> from keras import backend as K
    >>> conditions_true = (K.greater_equal(1.0, 0.0), K.less_equal(-1.0, 3.0))
    >>> K.get_value(logical_and(*conditions_true))
    True

    In all other cases, `logical_and` returns `False`:

    >>> from keras import backend as K
    >>> import itertools as it
    >>> conditions_false = (K.greater_equal(0.0, 5.0), K.less_equal(0.0, -3.0))
    >>> K.get_value(logical_and(*conditions_false))
    False
    >>> conditions = it.product(conditions_true, conditions_false)
    >>> any(K.get_value(logical_and(condition1, condition2)) for condition1, condition2 in conditions)
    False

    """

    if K.backend() == "tensorflow":
        return tf.logical_and(x, y)

    return K.all((x, y))


def indicator(condition: KerasTensor):
    """ Indicator function.
        Returns a scalar tensor with value `1` if `condition` is `True`, else
        a scalar `0` tensor.

    Parameters
    ----------
    condition: KerasTensor
        Tensor of type `bool`.

    Returns
    ----------
    indicator_tensor: KerasTensor
        Scalar integer tensor with value `1` if the given condition is `True`
        and value `0` otherwise.

    Examples
    ----------

    For truthy conditions, this function returns a scalar integer tensor
    with value `1`:

    >>> from keras import backend as K
    >>> conditions_true = (K.greater_equal(1.0, 0.0), K.less_equal(-1.0, 3.0))
    >>> all(K.get_value(indicator(condition)) == 1 for condition in conditions_true)
    True

    Otherwise, `indicator` returns a scalar integer tensor with value `0`:

    >>> from keras import backend as K
    >>> conditions_false = (K.greater_equal(0.0, 5.0), K.less_equal(0.0, -3.0))
    >>> all(K.get_value(indicator(condition)) == 0 for condition in conditions_false)
    True

    """
    return K.cast(condition, dtype=INTEGER_DTYPE)
