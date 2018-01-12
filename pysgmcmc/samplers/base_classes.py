import typing
import numpy as np
from keras import backend as K
from keras.optimizers import TFOptimizer, Optimizer as keras_optimizer
from tensorflow.python.training.optimizer import Optimizer as tensorflow_optimizer
from pysgmcmc.custom_typing import KerasTensor, KerasVariable


def sampler_from_optimizer(optimizer_cls):
    """ Automatically generate a sampler class from a given keras optimizer class.
        Samplers are generators that produce a stream of samples for a given
        energy function.

    Parameters
    ----------
    optimizer_cls : type
        Class type of a subclass of `keras.optimizer.Optimizer` or
        `tensorflow.python.training.optimizer.Optimizer`.

    Returns
    ----------
    sampler_cls : type
        Class type that inherits from `optimizer_cls` and has a more sampling oriented interface.
        Is a generator of a (possibly infinite) stream of samples.

    Examples
    ----------

    Wrapping an existing keras optimizer as a sampler is easy.
    Any arguments that can be passed to the original keras optimizer
    are also valid arguments for the resulting sampler:

    >>> from pysgmcmc.optimizers.sghmc import SGHMC
    >>> import keras
    >>> optimizer_cls = SGHMC
    >>> issubclass(optimizer_cls, keras.optimizers.Optimizer)
    True
    >>> sampler_cls = sampler_from_optimizer(optimizer_cls)
    >>> issubclass(sampler_cls, optimizer_cls)
    True
    >>> from itertools import islice
    >>> from pysgmcmc.samplers.energy_functions import Gmm1, to_negative_log_likelihood
    >>> params = [keras.backend.variable(1.0)]
    >>> loss = to_negative_log_likelihood(Gmm1())(params)
    >>> sampler = sampler_cls(params=params, loss=loss, lr=1e-2)  # we can pass optimizer argument "lr"
    >>> n_samples = 100
    >>> samples = [sample for _, sample in islice(sampler, n_samples)]
    >>> len(samples) == n_samples
    True

    The resulting sampler class has the same class name as the original
    optimizer class:

    >>> sampler_cls.__name__ == "SGHMC"
    True

    Wrapping an existing tensorflow optimizer as a sampler can be done exactly
    the same way:

    >>> from pysgmcmc.optimizers.sgld import SGLD
    >>> from tensorflow.python.training.optimizer import Optimizer as tensorflow_optimizer
    >>> optimizer_cls = SGLD
    >>> issubclass(optimizer_cls, tensorflow_optimizer)
    True
    >>> sampler_cls = sampler_from_optimizer(optimizer_cls)
    >>> issubclass(sampler_cls, optimizer_cls)
    True
    >>> from itertools import islice
    >>> from pysgmcmc.samplers.energy_functions import Gmm1, to_negative_log_likelihood
    >>> params = [keras.backend.variable(1.0)]
    >>> loss = to_negative_log_likelihood(Gmm1())(params)
    >>> sampler = sampler_cls(params=params, loss=loss, lr=1e-2)  # we can pass optimizer argument "lr"
    >>> n_samples = 100
    >>> samples = [sample for _, sample in islice(sampler, n_samples)]
    >>> len(samples) == n_samples
    True

    The resulting sampler class has the same class name as the original
    optimizer class:

    >>> sampler_cls.__name__ == "SGLD"
    True

    """

    class Sampler(optimizer_cls):
        def __init__(self,
                     loss: KerasTensor,
                     params: typing.List[KerasVariable],
                     inputs=None,
                     **optimizer_args) -> None:

            assert issubclass(optimizer_cls, (tensorflow_optimizer, keras_optimizer))

            is_tensorflow_optimizer = False
            super().__init__(**optimizer_args)

            if issubclass(optimizer_cls, tensorflow_optimizer):
                is_tensorflow_optimizer = True
                self.optimizer_object = TFOptimizer(super())

            self.loss = loss
            self.params = params

            if is_tensorflow_optimizer:
                self.updates = self.optimizer_object.get_updates(
                    self.loss, self.params
                )
            else:
                self.updates = self.get_updates(self.loss, self.params)

            inputs = inputs if inputs is not None else []

            self.function = K.function(
                inputs,
                [self.loss] + self.params,
                updates=self.updates,
                name="sampler_function"
            )

        def step(self, inputs=None) -> typing.Tuple[np.ndarray, np.ndarray]:
            if inputs is None:
                inputs = []
            loss, *params = self.function(inputs)
            return loss, params

        def __next__(self) -> typing.Tuple[np.ndarray, np.ndarray]:
            return self.step()

        def __iter__(self):
            return self

    Sampler.__name__ = optimizer_cls.__name__
    return Sampler
