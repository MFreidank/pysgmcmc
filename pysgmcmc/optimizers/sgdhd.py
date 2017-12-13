# vim:foldmethod=marker
import typing
from keras import backend as K
from pysgmcmc.keras_utils import (
    keras_control_dependencies, to_vector, n_dimensions, updates_for
)
from pysgmcmc.optimizers.hyperoptimizer import Hyperoptimizer

from keras.optimizers import Adam
from pysgmcmc.typing import KerasOptimizer, KerasTensor, KerasVariable


class SGDHD(Hyperoptimizer):
    def __init__(self,
                 lr: float=0.0,
                 hyperoptimizer: KerasOptimizer=Adam(),
                 seed: int=None,
                 **kwargs):
        super(SGDHD, self).__init__(hyperoptimizer=hyperoptimizer, **kwargs)
        self.seed = seed

        with K.name_scope(self.__class__.__name__):
            self.iterations = K.variable(0, dtype="int64", name="iterations")
            self.lr = K.variable(lr, name="lr")

    def get_updates(self,
                    loss: KerasTensor,
                    params: typing.List[KerasVariable]) -> typing.List[KerasTensor]:
        self.updates = [K.update_add(self.iterations, 1)]

        n_params = n_dimensions(params)

        self.dxdlr = K.zeros((n_params, 1))

        x = to_vector(params)

        dfdx = K.expand_dims(to_vector(K.gradients(loss, params)), axis=1)

        lr_t = self.hypergradient_update(dfdx=dfdx, dxdlr=-self.dxdlr)

        # NOTE: function f for dfdx *can* be loss function, but it might just
        # as well be any other function that takes parameters and returns
        # a value that we want to minimize, e.g. sampler specific errors etc.
        # lr_t = self.hypergradient(dfdx=dfdx, dxdlr=self.u)

        # self.updates.append((self.lr, lr_t))

        x = x - lr_t * K.reshape(dfdx, x.shape)

        with keras_control_dependencies([lr_t]):
            self.updates.append(K.update(self.dxdlr, dfdx))

            updates = updates_for(params, update_tensor=x)

            self.updates.extend([
                (param, K.reshape(update, param.shape))
                for param, update in zip(params, updates)
            ])

        return self.updates
