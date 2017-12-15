# Base class
import typing
from keras.optimizers import Optimizer
from keras import backend as K

from pysgmcmc.keras_utils import to_vector
from pysgmcmc.typing import KerasOptimizer, KerasTensor, KerasVariable


def to_hyperoptimizer(optimizer: KerasOptimizer):
    # Turn any keras.optimizer into a metaoptimizer we can use to tune
    # our learning rate parameter
    old_get_updates = optimizer.get_updates

    def new_get_updates(self,
                        gradients: typing.List[KerasTensor],
                        params: typing.List[KerasVariable]) -> typing.List[KerasTensor]:
        self.get_gradients = lambda *args, **kwargs: gradients
        return old_get_updates(loss=None, params=params)
    optimizer.get_updates = new_get_updates
    return optimizer


class Hyperoptimizer(Optimizer):
    def __init__(self, hyperoptimizer: KerasOptimizer, **kwargs) -> None:
        super().__init__(**kwargs)

        self.hyperoptimizer = to_hyperoptimizer(hyperoptimizer)

    def get_gradients(self,
                      loss: KerasTensor,
                      params: typing.List[KerasVariable]):
        gradient = to_vector(
            super().get_gradients(loss=loss, params=params)
        )

        self.dfdx = K.expand_dims(gradient, axis=1)
        return gradient

    def hypergradient_update(self,
                             dfdx: KerasTensor,
                             dxdlr: KerasTensor) -> KerasTensor:
        gradient = K.reshape(K.dot(K.transpose(dfdx), dxdlr), self.lr.shape)
        hyperupdates = self.hyperoptimizer.get_updates(
            self.hyperoptimizer,
            gradients=[gradient], params=[self.lr]
        )

        self.updates.extend(hyperupdates)
        *_, lr_t = hyperupdates
        return lr_t
