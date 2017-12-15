import sympy
import typing

from keras.optimizers import Adam
from keras import backend as K

from pysgmcmc.optimizers.hyperoptimizer import Hyperoptimizer
from pysgmcmc.keras_utils import (
    to_vector, updates_for, n_dimensions
)
from pysgmcmc.typing import KerasOptimizer, KerasTensor, KerasVariable


class SGDHD(Hyperoptimizer):
    def __init__(self,
                 lr: float=0.01,
                 hyperoptimizer: KerasOptimizer=Adam(),
                 **kwargs):

        super().__init__(hyperoptimizer=hyperoptimizer, **kwargs)

        with K.name_scope(self.__class__.__name__):
            self.iterations = K.variable(0, dtype="int64", name="iterations")
            self.lr = K.variable(lr, name="lr")

    def get_updates(self,
                    loss: KerasTensor,
                    params: typing.List[KerasVariable]) -> typing.List[KerasVariable]:
        self.updates = [K.update_add(self.iterations, 1)]

        n_params = n_dimensions(params)
        dfdx = to_vector(K.gradients(loss, params))
        dxdlr = K.zeros((n_params, 1))

        tensornames = ("x", "lr", "dfdx")

        sympy_tensors = {
            tensorname: sympy.symbols(tensorname) for tensorname in tensornames
        }

        tensorflow_tensors = {
            "x": to_vector(params),
            "lr": self.hypergradient_update(dfdx=K.expand_dims(dfdx), dxdlr=dxdlr),
            # "lr": self.lr,
            "dfdx": dfdx,
        }

        x, lr, dfdx = sympy_tensors["x"], sympy_tensors["lr"], sympy_tensors["dfdx"]

        update_sympy = x - lr * dfdx
        dxdlr_sympy = sympy.diff(x - lr * dfdx, lr)

        x_t = sympy.lambdify(
            (x, lr, dfdx), update_sympy, K.backend()
        )(tensorflow_tensors["x"],
          tensorflow_tensors["lr"],
          tensorflow_tensors["dfdx"])

        dxdlr_t = sympy.lambdify(
            (x, lr, dfdx), dxdlr_sympy, K.backend()
        )(tensorflow_tensors["x"],
          tensorflow_tensors["lr"],
          tensorflow_tensors["dfdx"])

        self.updates.append((dxdlr, K.expand_dims(dxdlr_t, axis=1)))

        updates = updates_for(params, update_tensor=x_t)

        self.updates.extend([
            (param, K.reshape(update, param.shape))
            for param, update in zip(params, updates)
        ])

        return self.updates
