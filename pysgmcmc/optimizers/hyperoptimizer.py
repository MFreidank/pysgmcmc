# Base class
import typing
from keras.optimizers import Optimizer, clip_norm
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
        def new_get_gradients(*args, **kwargs):
            grads = gradients
            if hasattr(self, 'clipnorm') and self.clipnorm > 0:
                norm = K.sqrt(sum([K.sum(K.square(g)) for g in grads]))
                grads = [clip_norm(g, self.clipnorm, norm) for g in grads]
            if hasattr(self, 'clipvalue') and self.clipvalue > 0:
                grads = [K.clip(g, -self.clipvalue, self.clipvalue) for g in grads]
            return grads
        self.get_gradients = new_get_gradients
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
