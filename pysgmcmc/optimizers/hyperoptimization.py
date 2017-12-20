import typing

from keras import backend as K
from keras.optimizers import Adam

from pysgmcmc.typing import KerasOptimizer, KerasTensor, KerasVariable


def to_hyperoptimizer(optimizer):
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


class Hyperoptimizer(object):
    def __init__(self,
                 hyperoptimizer: KerasOptimizer=Adam(lr=1e-5),
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self.hyperoptimizer = to_hyperoptimizer(hyperoptimizer)

    def hypergradient_update(self,
                             dfdx: KerasTensor,
                             dxdlr: KerasTensor,
                             hyperparameter: KerasVariable) -> typing.List[KerasTensor]:

        gradient = K.reshape(K.dot(K.transpose(dfdx), dxdlr), hyperparameter.shape)

        hyperupdates = self.hyperoptimizer.get_updates(
            self.hyperoptimizer,
            gradients=[gradient], params=[hyperparameter]
        )

        return hyperupdates
