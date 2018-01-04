import typing
import numpy as np
from keras import backend as K
from keras.optimizers import TFOptimizer
from tensorflow.python.training.optimizer import Optimizer as tf_optimizer
from pysgmcmc.custom_typing import KerasTensor, KerasVariable


def sampler_from_optimizer(optimizer_cls):
    class Sampler(optimizer_cls):
        def __init__(self,
                     loss: KerasTensor,
                     params: typing.List[KerasVariable],
                     inputs=None,
                     **optimizer_args) -> None:
            is_tf_optimizer = False
            super().__init__(**optimizer_args)
            if issubclass(optimizer_cls, tf_optimizer):
                is_tf_optimizer = True
                self.optimizer_obj = TFOptimizer(super())
            self.loss = loss
            self.params = params

            if is_tf_optimizer:
                self.updates = self.optimizer_obj.get_updates(
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
