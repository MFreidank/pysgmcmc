import abc
import numpy as np


class BaseModel(object):
    """
    Abstract base class for all machine learning models.
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self):
        self.X = None
        self.y = None

    @abc.abstractmethod
    def train(self, X: np.ndarray, y: np.ndarray):
        """
        Trains the model on the provided data.

        Parameters
        ----------
        X: np.ndarray (N, D)
            Input data points. The dimensionality of X is (N, D),
            with N as the number of points and D is the number of input dimensions.
        y: np.ndarray (N,)
            The corresponding target values of the input data points.
        """

    def update(self, X: np.ndarray, y: np.ndarray):
        """
        Update the model with the new additional data. Override this function if your
        model allows to do something smarter than simple retraining

        Parameters
        ----------
        X: np.ndarray (N, D)
            Input data points. The dimensionality of X is (N, D),
            with N as the number of points and D is the number of input dimensions.
        y: np.ndarray (N,)
            The corresponding target values of the input data points.

        """
        X = np.append(self.X, X, axis=0)
        y = np.append(self.y, y, axis=0)
        self.train(X, y)

    @abc.abstractmethod
    def predict(self, X_test: np.ndarray):
        """
        Predicts for a given set of test data points the mean and variance of its target values

        Parameters
        ----------
        X_test: np.ndarray (N, D)
            N Test data points with input dimensions D

        Returns
        ----------
        mean: ndarray (N,)
            Predictive mean of the test data points
        var: ndarray (N,)
            Predictive variance of the test data points

        """

    def _check_shapes_train(func):
        def func_wrapper(self, X: np.ndarray, y: np.ndarray, *args, **kwargs):
            assert X.shape[0] == y.shape[0]
            assert len(X.shape) == 2
            assert len(y.shape) == 1
            return func(self, X, y, *args, **kwargs)
        return func_wrapper

    def _check_shapes_predict(func):
        def func_wrapper(self, X, *args, **kwargs):
            assert len(X.shape) == 2
            return func(self, X, *args, **kwargs)

        return func_wrapper

    def get_json_data(self):
        """
        Json getter function'

        Returns
        ----------
            dictionary
        """
        json_data = {'X': self.X if self.X is None else self.X.tolist(),
                     'y': self.y if self.y is None else self.y.tolist(),
                     'hyperparameters': ""}
        return json_data

    def get_incumbent(self):
        """
        Returns the best observed point and its function value

        Returns
        ----------
        incumbent: ndarray (D,)
            current incumbent
        incumbent_value: ndarray (N,)
            the observed value of the incumbent
        """
        best_idx = np.argmin(self.y)
        return self.X[best_idx], self.y[best_idx]


def safe_division(x, y, small_constant=1e-16):
    """ Computes `x / y` after adding a small appropriately signed constant to `y`.
        Adding a small constant avoids division-by-zero artefacts that may
        occur due to precision errors.

    Parameters
    ----------
    x: np.ndarray
        Left-side operand of division.
    y: np.ndarray
        Right-side operand of division.
    small_constant: float, optional
        Small constant to add to/subtract from `y` before computing `x / y`.
        Defaults to `1e-16`.

    Returns
    ----------
    division_result : np.ndarray
        Result of `x / y` after adding a small appropriately signed constant
        to `y` to avoid division by zero.

    Examples
    ----------

    Will safely avoid divisions-by-zero under normal circumstances:

    >>> import numpy as np
    >>> x = np.asarray([1.0])
    >>> inf_tensor = x / 0.0  # will produce "inf" due to division-by-zero
    >>> bool(np.isinf(inf_tensor))
    True
    >>> z = safe_division(x, 0., small_constant=1e-16)  # will avoid division-by-zero
    >>> bool(np.isinf(z))
    False

    To see that simply adding a positive constant may fail, consider the
    following example. Note that this function handles such corner cases correctly:

    >>> import numpy as np
    >>> x, y = np.asarray([1.0]), np.asarray([-1e-16])
    >>> small_constant = 1e-16
    >>> inf_tensor = x / (y + small_constant)  # simply adding constant exhibits division-by-zero
    >>> bool(np.isinf(inf_tensor))
    True
    >>> z = safe_division(x, y, small_constant=1e-16)  # will avoid division-by-zero
    >>> bool(np.isinf(z))
    False

    """
    if (np.asarray(y) == 0).all():
        return np.true_divide(x, small_constant)
    return np.true_divide(x, np.sign(y) * small_constant + y)

# XXX: Write docs for everything below (and maybe merge some of the tests over here).


def zero_one_normalization(X, lower=None, upper=None):

    if lower is None:
        lower = np.min(X, axis=0)
    if upper is None:
        upper = np.max(X, axis=0)

    X_normalized = safe_division(X - lower, upper - lower)

    return X_normalized, lower, upper


def zero_one_unnormalization(X_normalized, lower, upper):
    return lower + (upper - lower) * X_normalized


def zero_mean_unit_var_normalization(X, mean=None, std=None):
    if mean is None:
        mean = np.mean(X, axis=0)
    if std is None:
        std = np.std(X, axis=0)

    X_normalized = safe_division(X - mean, std)

    return X_normalized, mean, std


def zero_mean_unit_var_unnormalization(X_normalized, mean, std):
    return X_normalized * std + mean
