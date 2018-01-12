import abc
import typing

from keras import backend as K

from pysgmcmc.keras_utils import (
    INTEGER_DTYPE, FLOAT_DTYPE, while_loop, logical_and, indicator,
)
from pysgmcmc.custom_typing import KerasTensor


class SamplerLoss(object):
    """ Base class of all keras losses computed on chains of samples.
        Note: Inheriting from this class turns arbitrary (positive) diagnostics
        into losses. Minimizing this resulting loss will allow one to
        maximize the original diagnostic.
        Updating a loss object can be done using a simple call,
        for example for effective sample size ess:
            `loss_tensor = ess(new_sample, iteration)`
        Now, minimizing `loss_tensor` maximizes effective sample size.
    """

    __metaclass__ = abc.ABCMeta

    def __init__(self,
                 parameter_shape: typing.Tuple[int, ...],
                 n_iterations: int,
                 aggregation_function: typing.Callable[[KerasTensor], KerasTensor]=K.min) -> None:
        """ A sampler loss that tracks some property of a growing list of samples.
            Maintains a state of all recorded samples and uses this state to compute
            loss values.

        Parameters
        ----------
        parameter_shape : typing.Tuple[int, ...]
            (Vectorized) shape of our target parameters.
        n_iterations : int
            Number of iterations to perform in total.
            (Needed to set up state for list of recorded samples.)
        aggregation_function : typing.Callable[[KerasTensor], KerasTensor], optional
            Function used to aggregate diagnostic results
            of each dimension into one value.
            Defaults to `keras.backend.min`.

        """
        self.samples = K.zeros((n_iterations, *parameter_shape), dtype=FLOAT_DTYPE)
        self.sample_dimensionality, *_ = parameter_shape
        self.aggregation_function = aggregation_function

    def add_sample(self, new_sample, iteration):
        """TODO: Docstring for add_sample.

        Parameters
        ----------
        new_sample : KerasTensor
            Most recent recorded sample. Recorded at iteration `iteration`.

        iteration : KerasTensor
            Integer tensor that specifies the current iteration number.

        Returns
        ----------
        updated_samples : KerasTensor
            All samples at the given `iteration`, after adding `new_sample`.

        """
        return self.samples[iteration].assign(new_sample)

    @abc.abstractmethod
    def apply_1d(self, samples: KerasTensor) -> KerasTensor:
        """ Apply this diagnostic to a 1d chain of samples for a single dimension/parameter.

        Parameters
        ----------
        samples : KerasTensor
            1-d tensor containing a chain of samples for a single dimension/parameter.

        Returns
        ----------
        diagnostic_1d : KerasTensor
            Result of computing this diagnostic for the given `samples`.

        """
        raise NotImplementedError

    def apply_nd(self, current_samples: typing.List[KerasTensor]):
        diagnostic_values = K.map_fn(
            fn=self.apply_1d,
            elems=K.squeeze(K.transpose(current_samples), axis=0)
        )
        return diagnostic_values

    def __call__(self, new_sample: KerasTensor, iteration: KerasTensor):
        """ Update this loss with a `new_sample` recorded at a given `iteration`.
            Will update internal state, recompute the loss value and return
            a new loss.

        Parameters
        ----------
        new_sample : KerasTensor
            Most recent recorded sample. Recorded at iteration `iteration`.

        iteration : KerasTensor
            Integer tensor that specifies the current iteration number.

        Returns
        ----------
        aggregated_loss : KerasTensor
            Result of applying this loss to all recorded samples after adding
            `new_sample`.
            Hereby, diagnostics are computed dimension-wise and then aggregated using
            the `aggregation_function` of this diagnostic object.

        Examples
        ----------

        For only a single sample, effective sample size cannot be computed
        properly, which we catch by returning `0.0`:

        >>> from numpy import allclose
        >>> x = K.ones((20000, 1))
        >>> ess = EffectiveSampleSize(parameter_shape=K.int_shape(x), n_iterations=2)
        >>> K.get_value(ess(x, iteration=0))
        0.0

        For multiple samples, we get reasonable (negative) effective sample sizes:

        >>> allclose(K.get_value(ess(K.zeros((20000, 1)), iteration=1)), -0.66666669)
        True

        """
        current_samples = self.add_sample(
            new_sample=new_sample, iteration=iteration
        )[:iteration + 1]

        effective_sample_size = K.switch(
            K.greater(iteration, 0),
            -self.aggregation_function(self.apply_nd(current_samples)),
            K.constant(0., dtype=FLOAT_DTYPE)
        )

        return effective_sample_size


class EffectiveSampleSize(SamplerLoss):
    def apply_1d(self, samples: KerasTensor):
        n, = K.int_shape(samples)

        n_tensor = K.constant(n, dtype=INTEGER_DTYPE)
        # mu_hat, var = moments(samples, axis=0)
        mu_hat, var = K.mean(samples, axis=0), K.var(samples, axis=0)

        var_plus = var * K.cast(n_tensor, var.dtype) / K.cast(n_tensor, var.dtype)

        t = K.constant(0, dtype=INTEGER_DTYPE)
        sum_rho = K.constant(0.)
        last_rho = K.constant(0.)

        def body(t, sum_rho, last_rho):
            autocovariance = K.mean(
                (samples[:n_tensor - t] - mu_hat) * (samples[t:] - mu_hat)
            )

            rho = 1. - (var - autocovariance) / var_plus

            sum_rho += K.cast(
                indicator(
                    K.greater_equal(rho, 0.)
                ),
                dtype=FLOAT_DTYPE
            ) * rho

            return t + 1, sum_rho, rho

        def condition(t, sum_rho, last_rho):
            return logical_and(
                K.less(t, n_tensor),
                K.greater_equal(last_rho, 0.)
            )

        _, rho_sum, _ = while_loop(
            condition=condition, body=body, loop_variables=(t, sum_rho, last_rho)
        )

        return K.cast(n, FLOAT_DTYPE) / (1 + 2 * rho_sum)


class Autocorrelation(SamplerLoss):
    def __init__(self,
                 parameter_shape: typing.Tuple[int, ...],
                 n_iterations: int, lag: int=1,
                 aggregation_function: typing.Callable[[KerasTensor], KerasTensor]=K.min) -> None:
        super().__init__(
            parameter_shape=parameter_shape,
            n_iterations=n_iterations,
            aggregation_function=aggregation_function
        )

        self.lag = lag

    def apply_1d(self, samples: KerasTensor):
        autocovariance = K.mean(
            (samples[:-self.lag] - K.mean(samples[:-self.lag])) *
            (samples[self.lag:] - K.mean(samples[self.lag:]))
        )

        variances = K.var(samples[:-self.lag]), K.var(samples[self.lag:])

        return autocovariance / K.sqrt(K.prod(variances))
