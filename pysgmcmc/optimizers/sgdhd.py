# vim:foldmethod=marker
from keras import backend as K
from keras.optimizers import Optimizer
from pysgmcmc.keras_utils import (
    keras_control_dependencies, to_vector, tensor_size, keras_split, n_dimensions
)

from keras.optimizers import Adam
from pysgmcmc.optimizers import to_metaoptimizer


class SGDHD(Optimizer):
    def __init__(self, lr=0.0, metaoptimizer=Adam(), seed=None, **kwargs):
        super(SGDHD, self).__init__(**kwargs)
        self.seed = seed

        with K.name_scope(self.__class__.__name__):
            self.iterations = K.variable(0, dtype="int64", name="iterations")
            self.lr = K.variable(lr, name="lr")

            self.metaoptimizer = to_metaoptimizer(metaoptimizer)

    def hypergradient_update(self, dfdx, dxdlr):
        gradient = K.reshape(K.dot(K.transpose(dfdx), dxdlr), self.lr.shape)
        metaupdates = self.metaoptimizer.get_updates(
            self.metaoptimizer,
            gradients=[gradient], params=[self.lr]
        )

        self.updates.extend(metaupdates)
        *_, lr_t = metaupdates
        return lr_t

    def get_updates(self, loss, params):
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

            param_sizes = [tensor_size(param) for param in params]

            for param, param_t in zip(params, keras_split(x, param_sizes, axis=0)):
                self.updates.append((param, K.reshape(param_t, param.shape)))

        return self.updates
