from collections import OrderedDict
import sympy
import typing

from keras.optimizers import Adam, SGD
from keras import backend as K

from pysgmcmc.optimizers.hyperoptimization import Hyperoptimizer
from pysgmcmc.keras_utils import (
    to_vector, updates_for, n_dimensions, sympy_to_keras
)
from pysgmcmc.custom_typing import KerasOptimizer, KerasTensor, KerasVariable


class SGDHD(Hyperoptimizer, SGD):
    def __init__(self,
                 lr: float=0.01,
                 hyperoptimizer: KerasOptimizer=Adam(lr=1e-5),
                 hyperloss=None,
                 **kwargs) -> None:

        with K.name_scope(self.__class__.__name__):
            super().__init__(
                hyperoptimizer=hyperoptimizer,
                hyperloss=None,
                lr=lr,
                **kwargs
            )

    def get_updates(self,
                    loss: KerasTensor,
                    params: typing.List[KerasVariable]) -> typing.List[KerasVariable]:
        self.updates = [K.update_add(self.iterations, 1)]

        n_params = n_dimensions(params)

        dfdx = to_vector(K.gradients(loss, params))
        dxdlr = K.zeros((n_params, 1))

        x_sympy, lr_sympy, dfdx_sympy = sympy.symbols("x lr dfdx")

        *hyperupdates, lr_t = self.hypergradient_update(
            loss=loss, params=params,
            dxdlr=dxdlr,
            hyperparameter=self.lr
        )

        tensors = OrderedDict((
            (x_sympy, to_vector(params)),
            (lr_sympy, lr_t),
            (dfdx_sympy, dfdx)
        ))

        update_sympy = x_sympy - lr_sympy * dfdx_sympy
        dxdlr_sympy = sympy.diff(x_sympy - lr_sympy * dfdx_sympy, lr_sympy)

        x_t = sympy_to_keras(
            sympy_expression=update_sympy,
            sympy_tensors=tuple(tensors.keys()),
            tensorflow_tensors=tuple(tensors.values())
        )

        dxdlr_t = sympy_to_keras(
            sympy_expression=dxdlr_sympy,
            sympy_tensors=tuple(tensors.keys()),
            tensorflow_tensors=tuple(tensors.values())
        )

        print(dxdlr.shape, dxdlr_t.shape)
        self.updates.append((dxdlr, K.expand_dims(dxdlr_t, axis=1)))

        updates = updates_for(params, update_tensor=x_t)

        self.updates.extend([
            (param, K.reshape(update, param.shape))
            for param, update in zip(params, updates)
        ])

        return self.updates
