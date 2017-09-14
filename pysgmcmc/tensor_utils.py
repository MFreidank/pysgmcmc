"""
This module contains util functions to facilitate working
with `tensorflow.Tensor` objects.
"""

import numpy as np
import tensorflow as tf

__all__ = (
    "vectorize", "unvectorize", "median",
    "safe_divide", "safe_sqrt",
    "pdist", "squareform",
    "uninitialized_params",
)


def vectorize(tensor):
    """ Turn any matrix into a long vector by expanding it.
        Tranforms `[[a, b], [c, d]]` into `[a, b, c, d]`.

        For vector inputs, this simply returns a copy of the vector.

        For reference see also *vec*-operator in:
            https://hec.unil.ch/docs/files/23/100/handout1.pdf#page=2

    Parameters
    ----------

    tensor : tensorflow.Variable object or tensorflow.Tensor object
        Input tensor to vectorize.

    Returns
    ----------

    tensor_vectorized: tensorflow.Variable object or tensorflow.Tensor object
        Vectorized result for input `tensor`.

    Examples
    ----------

    A tensorflow.Variable can be vectorized:
    (NOTE: the returned vectorized variable must be initialized before using
    it in `tensorflow` computations.)

    >>> import tensorflow as tf
    >>> v1 = tf.Variable([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    >>> v1_vectorized = vectorize(v1)
    >>> session = tf.Session()
    >>> session.run(tf.global_variables_initializer())
    >>> session.run(v1_vectorized)
    array([[ 1.],
           [ 2.],
           [ 3.],
           [ 4.],
           [ 5.],
           [ 6.]], dtype=float32)


    A normal `tensorflow.Tensor` can be vectorized:

    >>> import tensorflow as tf
    >>> t1 = tf.constant([[12.0, 14.0, -3.0], [4.0, 3.0, 1.0], [9.0, 2.0, 4.0]])
    >>> t1_vectorized = vectorize(t1)
    >>> session = tf.Session()
    >>> session.run(t1_vectorized)
    array([[ 12.],
           [ 14.],
           [ -3.],
           [  4.],
           [  3.],
           [  1.],
           [  9.],
           [  2.],
           [  4.]], dtype=float32)

    """

    # Compute vectorized shape
    n_elements = np.prod(np.asarray(tensor.shape, dtype=np.int))
    vectorized_shape = (n_elements, 1)

    if type(tensor) == tf.Variable:
        return tf.Variable(
            tf.reshape(tensor.initialized_value(), shape=vectorized_shape)
        )

    elif isinstance(tensor, tf.Tensor):
        return tf.reshape(tensor, shape=vectorized_shape)

    else:
        raise ValueError(
            "Unsupported input to tensor_utils.vectorize: "
            "{value} is not a tensorflow.Tensor subclass".format(value=tensor)
        )


def unvectorize(tensor, original_shape):
    """ Reshape previously vectorized `tensor` back to its `original_shape`.
        Essentially the inverse transformation as the one performed by
        `tensor_utils.vectorize`.

    Parameters
    ----------
    tensor : tensorflow.Variable object or tensorflow.Tensor object
        Input tensor to unvectorize.

    original_shape : tensorflow.Shape
        Original shape of `tensor` prior to its vectorization.

    Returns
    ----------
    tensor_unvectorized : tensorflow.Tensor
        Tensor with the same values as `tensor` but reshaped back to
        shape `original_shape`.

    Examples
    ----------
    Function `unvectorize` undoes the work done by `vectorize`:

    >>> import tensorflow as tf
    >>> import numpy as np
    >>> t1 = tf.constant([[12.0, 14.0, -3.0], [4.0, 3.0, 1.0], [9.0, 2.0, 4.0]])
    >>> t2 = unvectorize(vectorize(t1), original_shape=t1.shape)
    >>> session = tf.Session()
    >>> t1_array, t2_array = session.run([t1, t2])
    >>> np.allclose(t1_array, t2_array)
    True

    It will also work for `tensorflow.Variable` objects, but will return
    `tensorflow.Tensor` as unvectorized output.

    >>> import tensorflow as tf
    >>> import numpy as np
    >>> v = tf.Variable([[0.0, 1.0], [2.0, 0.0]])
    >>> session = tf.Session()
    >>> session.run(tf.global_variables_initializer())
    >>> t = unvectorize(vectorize(v.initialized_value()), original_shape=v.shape)
    >>> v_array, t_array = session.run([v, t])
    >>> np.allclose(t_array, v_array)
    True

    """
    return tf.reshape(tensor, shape=original_shape)


def _is_vector(tensor):
    return len(tensor.shape.as_list()) == 1


def median(tensor):
    """
    Return the median (middle value) of data in `tensor`.

    When the number of data points is odd, return the middle data point.
    When the number of data points is even, the median is interpolated by
    taking the average of the two middle values.

    Parameters
    ----------
    tensor : tensorflow.Tensor
        Input tensor (may be multidimensional) for which the median
        should be computed.

    Returns
    ----------
    median_tensor : tensorflow.Tensor
        Scalar tensor whose value is the median of the input tensor.

    Examples
    ----------
    XXX Use simple examples from python.statistics doku

    """
    tensor_reshaped = tf.reshape(tensor, [-1])

    n_elements, *_ = tensor_reshaped.get_shape()

    sorted_tensor = tf.nn.top_k(tensor_reshaped, n_elements, sorted=True)

    mid_index = n_elements // 2

    if n_elements % 2 == 1:
        return sorted_tensor.values[mid_index]

    return (sorted_tensor.values[mid_index - 1] + sorted_tensor.values[mid_index]) / 2


def safe_divide(x, y, small_constant=1e-16, name=None):
    """ `tf.divide(x, y)` after adding a small appropriate constant to `y`
        in a smart way so that we can avoid division-by-zero artefacts.

    Parameters
    ----------
    x : tensorflow.Tensor
        Left-side operand of `tensorflow.divide`

    y : tensorflow.Tensor
        Right-side operand of `tensorflow.divide`

    small_constant : tensorflow.Tensor
        Small constant tensor to add to/subtract from `y` before computing
        `x / y` to avoid division-by-zero.

    name : string or NoneType, optional
        Name of the resulting node in a `tensorflow.Graph`.
        Defaults to `None`.

    Returns
    ----------
    division_result : tensorflow.Tensor
        Result of division `tf.divide(x, y)` after applying clipping to `y`.

    Examples
    ----------

    Will safely avoid divisions-by-zero under normal circumstances:

    >>> import tensorflow as tf
    >>> import numpy as np
    >>> session = tf.Session()
    >>> x = tf.constant(1.0)
    >>> nan_tensor = tf.divide(x, 0.0)  # will produce "inf" due to division-by-zero
    >>> np.isinf(nan_tensor.eval(session=session))
    True
    >>> z = safe_divide(x, 0., small_constant=1e-16)  # will avoid "inf" due to division-by-zero by clipping
    >>> np.isinf(z.eval(session=session))
    False

    To see that simply adding a constant may fail, but this implementation
    handles those corner cases correctly, consider this example:

    >>> import tensorflow as tf
    >>> import numpy as np
    >>> x, y = tf.constant(1.0), tf.constant(-1e-16)
    >>> small_constant = tf.constant(1e-16)
    >>> v1 = x / (y + small_constant)  # without sign
    >>> v2 = safe_divide(x, y, small_constant=small_constant) # with sign
    >>> val1, val2 = session.run([v1, v2])
    >>> np.isinf(val1) # simply adding without considering the sign can still yield "inf"
    True
    >>> np.isinf(val2)  # our version behaves appropriately
    False


    """
    return tf.divide(x, y + (2. * tf.sign(y) * small_constant + small_constant), name=name)


# XXX: Ensure that we only clip values that are close to zero already, and raise an error
# otherwise?
def safe_sqrt(x, clip_value_min=0., clip_value_max=float("inf"), name=None):
    """ Computes `tf.sqrt(x)` after clipping tensor `x` using
        `tf.clip_by_value(x, clip_value_min, clip_value_max)` to avoid
        square root (e.g. of negative values) artefacts.

    Parameters
    ----------
    x : tensorflow.Tensor or tensorflow.SparseTensor
        Operand of `tensorflow.sqrt`.

    clip_value_min : 0-D (scalar) tensorflow.Tensor, optional
        The minimum value to clip by.
        Defaults to `0`

    clip_value_max : 0-D (scalar) tensorflow.Tensor, optional
        The maximum value to clip by.
        Defaults to `float("inf")`

    name : string or NoneType, optional
        Name of the resulting node in a tensorflow.Graph.
        Defaults to `None`.

    Returns
    ----------
    sqrt_result: `tensorflow.Tensor`
        Result of square root `tf.sqrt(x)` after applying clipping to `x`.

    Examples
    ----------

    Will safely avoid square root of negative values:

    >>> import tensorflow as tf
    >>> import numpy as np
    >>> x = tf.constant(-1e-16)
    >>> z = tf.sqrt(x)  # fails, results in 'nan'
    >>> z_safe = safe_sqrt(x)  # works, results in '0'
    >>> session = tf.Session()
    >>> z_val, z_safe_val = session.run([z, z_safe])
    >>> np.isnan(z_val)  # ordinary tensorflow computation gives 'nan'
    True
    >>> np.isnan(z_safe_val) # `safe_sqrt` produces '0'.
    False
    >>> z_safe_val
    0.0

    """
    return tf.sqrt(
        tf.clip_by_value(
            x, clip_value_min=clip_value_min, clip_value_max=clip_value_max
        ), name=name
    )


def pdist(tensor, metric="euclidean"):
    """
    Pairwise distances between observations in n-dimensional space.
    Ported from `scipy.spatial.distance.pdist`
    @2f5aa264724099c03772ed784e7a947d2bea8398
    for cherry-picked distance metrics.

    Parameters
    ----------
    tensor : tensorflow.Tensor

    metric : string, optional
        Pairwise metric to apply.
        Defaults to "euclidean".

    Returns
    ----------
    Y : tensorflow.Tensor
        Returns a condensed distance matrix `Y` as `tensorflow.Tensor`.
        For each :math:`i` and :math:`j` (where :math:`i<j<m`),
        where m is the number of original observations.
        The metric ``dist(u=X[i], v=X[j])`` is computed and stored in
        entry ``j`` of subtensor ``Y[j]``.

    Examples
    ----------
    Gives equivalent results to `scipy.spatial.distance.pdist` but uses
    tensorflow.Tensor objects:

    >>> import tensorflow as tf
    >>> import numpy as np
    >>> from scipy.spatial.distance import pdist as pdist_scipy
    >>> input_scipy = np.array([[ 0.77228064,  0.09543156], [ 0.3918973 ,  0.96806584], [ 0.66008144,  0.22163063]])
    >>> result_scipy = pdist_scipy(input_scipy, metric="euclidean")
    >>> session = tf.Session()
    >>> input_tensorflow = tf.constant(input_scipy)
    >>> result_tensorflow = session.run(pdist(input_tensorflow, metric="euclidean"))
    >>> np.allclose(result_scipy, result_tensorflow)
    True

    Will raise a `NotImplementedError` for unsupported metric choices:

    >>> import tensorflow as tf
    >>> import numpy as np
    >>> input_scipy = np.array([[ 0.77228064,  0.09543156], [ 0.3918973 ,  0.96806584], [ 0.66008144,  0.22163063]])
    >>> session = tf.Session()
    >>> input_tensorflow = tf.constant(input_scipy)
    >>> session.run(pdist(input_tensorflow, metric="lengthy_metric"))
    Traceback (most recent call last):
     ...
    NotImplementedError: tensor_utils.pdist: Metric 'lengthy_metric' currently not supported!

    Like `scipy.spatial.distance.pdist`, we fail for input that is not 2-d:

    >>> import tensorflow as tf
    >>> import numpy as np
    >>> input_scipy = np.random.rand(2, 2, 1)
    >>> session = tf.Session()
    >>> input_tensorflow = tf.constant(input_scipy)
    >>> session.run(pdist(input_tensorflow, metric="lengthy_metric"))
    Traceback (most recent call last):
     ...
    ValueError: tensor_utils.pdist: A 2-d tensor must be passed.

    """

    assert(isinstance(tensor, tf.Tensor)), "tensor_utils.pdist: Input must be a `tensorflow.Tensor` instance."

    if len(tensor.shape.as_list()) != 2:
        raise ValueError('tensor_utils.pdist: A 2-d tensor must be passed.')

    if metric == "euclidean":

        def pairwise_euclidean_distance(tensor):
            def euclidean_distance(tensor1, tensor2):
                return tf.norm(tensor1 - tensor2)

            m = tensor.shape.as_list()[0]

            distances = []
            for i in range(m):
                for j in range(i + 1, m):
                    distances.append(euclidean_distance(tensor[i], tensor[j]))
            return tf.convert_to_tensor(distances)

        metric_function = pairwise_euclidean_distance
    else:
        raise NotImplementedError(
            "tensor_utils.pdist: "
            "Metric '{metric}' currently not supported!".format(metric=metric)

        )

    return metric_function(tensor)


# XXX missing parts of docs (TODO below)
def squareform(tensor):
    """ TODO
        Ported from `scipy.spatial.distance.squareform`
        @2f5aa264724099c03772ed784e7a947d2bea8398, but supports only
        1-d (vector) input

    Parameters
    ----------
    tensor : tensorflow.Tensor

    Returns
    ----------
    redundant_distance_tensor : tensorflow.Tensor

    Examples
    ----------
    May be used in conjunction with `tensor_utils.pdist` to obtain
    a redundant distance matrix:

    >>> import tensorflow as tf
    >>> import numpy as np
    >>> from scipy.spatial.distance import pdist as scipy_pdist, squareform as scipy_squareform
    >>> original_input = np.random.rand(2, 4)
    >>> tf_redundant_distance_tensor = squareform(pdist(tf.constant(original_input)))
    >>> scipy_redundant_distance_matrix = scipy_squareform(scipy_pdist(original_input))
    >>> session = tf.Session()
    >>> tf_redundant_distance_matrix = session.run(tf_redundant_distance_tensor)
    >>> np.allclose(tf_redundant_distance_matrix, scipy_redundant_distance_matrix)
    True

    Contrary to `scipy.spatial.squareform`, conversion of 2D input to
    a condensed distance vector is *not* supported:

    >>> import numpy as np
    >>> import tensorflow as tf
    >>> illegal_input = tf.constant(np.random.rand(4, 4))
    >>> squareform(illegal_input)
    Traceback (most recent call last):
     ...
    NotImplementedError: tensor_utils.squareform: Only 1-d (vector) input is supported!

    """

    assert(isinstance(tensor, tf.Tensor)), "tensor_utils.squareform: Input must be a `tensorflow.Tensor` instance."

    tensor_shape = tensor.shape.as_list()
    n_elements = tensor_shape[0]

    if _is_vector(tensor):
        # vector to matrix
        if n_elements == 0:
            return tf.zeros((1, 1), dtype=tensor.dtype)

        # Grab the closest value to the square root of the number
        # of elements times 2 to see if the number of elements is
        # indeed a binomial coefficient
        dimension = int(np.ceil(np.sqrt(n_elements * 2)))

        # Check that `tensor` is of valid dimensions
        if dimension * (dimension - 1) != n_elements * 2:
            raise ValueError(
                "Incompatible vector size. It must be a binomial "
                "coefficient n choose 2 for some integer n >=2."
            )

        n_total_elements_matrix = dimension ** 2

        # Stitch together an upper triangular matrix for our redundant
        # distance matrix from our condensed distance tensor and
        # two tensors filled with zeros.

        n_diagonal_zeros = dimension
        n_fill_zeros = n_total_elements_matrix - n_elements - n_diagonal_zeros

        condensed_distance_tensor = tf.reshape(tensor, shape=(n_elements, 1))
        diagonal_zeros = tf.zeros(
            shape=(n_diagonal_zeros, 1), dtype=condensed_distance_tensor.dtype
        )
        fill_zeros = tf.zeros(
            shape=(n_fill_zeros, 1), dtype=condensed_distance_tensor.dtype
        )

        def upper_triangular_indices(dimension: int):
            """ For a square matrix with shape (`dimension`, `dimension`),
                return a list of indices into a vector with
                `dimension * dimension` elements that correspond to its
                upper triangular part after reshaping.

            Parameters
            ----------
            dimension : int
                Target dimensionality of the square matrix we want to
                obtain by reshaping a `dimension * dimension` element
                vector.

            Yields
            -------
            index: int
                Indices are indices into a `dimension * dimension` element
                vector that correspond to the upper triangular part of the
                matrix obtained by reshaping it into shape
                `(dimension, dimension)`.

            """

            assert(dimension > 0), "tensor_utils.upper_triangular_indices: Dimension must be positive integer!"

            for row in range(dimension):
                for column in range(row + 1, dimension):
                    element_index = dimension * row + column
                    yield element_index

        # General Idea: Use that redundant distance matrices are symmetric:
        # First construct only an upper triangular part and fill
        # everything else with zeros.
        # To the resulting matrix add its transpose, which results in a full
        # redundant distance matrix.

        all_indices = set(range(n_total_elements_matrix))
        diagonal_indices = list(range(0, n_total_elements_matrix, dimension + 1))
        upper_triangular = list(upper_triangular_indices(dimension))

        remaining_indices = all_indices.difference(
            set(diagonal_indices).union(upper_triangular)
        )

        data = (
            # diagonal zeros of our redundant distance matrix
            diagonal_zeros,
            # upper triangular part of our redundant distance matrix
            condensed_distance_tensor,
            # fill zeros for lower triangular part
            fill_zeros
        )

        indices = (
            tuple(diagonal_indices),
            tuple(upper_triangular),
            tuple(remaining_indices)
        )

        stitch_vector = tf.dynamic_stitch(data=data, indices=indices)

        # reshape into matrix
        upper_triangular = tf.reshape(stitch_vector, (dimension, dimension))

        # redundant distance matrices are symmetric
        lower_triangular = tf.transpose(upper_triangular)

        return upper_triangular + lower_triangular
    else:
        raise NotImplementedError(
            "tensor_utils.squareform: Only 1-d "
            "(vector) input is supported!"
        )


def uninitialized_params(params, session=None):
    """
    Return the list containing all tensorflow.Variable objects present in
    iterable `params` that are not yet initialized.

    Parameters
    ----------
    params : list of tensorflow.Variable objects
        List of parameters to check for initialization.

    Returns
    -------
    params_uninitialized: list of tensorflow.Variable objects
        All `tensorflow.Variable` objects in `params` that were not
        yet initialized in the current graph.

    """

    if session is None:
        session = tf.get_default_session()

    init_flag = session.run(
        tf.stack([tf.is_variable_initialized(v) for v in params])
    )

    return [param for param, flag in zip(params, init_flag) if not flag]
