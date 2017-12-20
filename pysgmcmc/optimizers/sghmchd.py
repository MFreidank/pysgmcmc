# vim:foldmethod=marker
import typing
import sympy
from keras import backend as K
from keras.optimizers import Optimizer, Adam
from pysgmcmc.keras_utils import (
    keras_control_dependencies,
    n_dimensions, to_vector, updates_for
)
from pysgmcmc.optimizers.sghmc import SGHMC
from pysgmcmc.typing import KerasTensor, KerasVariable
from collections import OrderedDict


def to_tensorflow(sympy_expression, sympy_tensors, tensorflow_tensors):
    return sympy.lambdify(sympy_tensors, sympy_expression, "tensorflow")(*tensorflow_tensors)


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


class SGHMCHD(SGHMC):
    def __init__(self,
                 hyperoptimizer=Adam(lr=1e-5),
                 lr: float=0.01,
                 independent_stepsizes: bool=True,
                 mdecay: float=0.05,
                 burn_in_steps: int=3000,
                 scale_grad: float=1.0,
                 seed: int=None,
                 **kwargs) -> None:
        super(SGHMCHD, self).__init__(**kwargs)
        self.seed = seed
        self.hyperoptimizer = to_hyperoptimizer(hyperoptimizer)

        with K.name_scope(self.__class__.__name__):
            self.iterations = K.variable(0, dtype="int64", name="iterations")

            self.lr = K.variable(lr, name="lr")

            #  Initialize Graph Constants {{{ #
            self.noise = K.constant(0., name="noise")

            self.scale_grad = K.constant(scale_grad, name="scale_grad")

            self.burn_in_steps = K.constant(
                burn_in_steps, dtype="int64", name="burn_in_steps"
            )

            self.mdecay = K.constant(mdecay, name="mdecay")
            #  }}} Initialize Graph Constants #

    def hypergradient_update(self, dfdx, dxdlr):
        gradient = K.reshape(K.dot(K.transpose(dfdx), dxdlr), self.lr.shape)

        hyperupdates = self.hyperoptimizer.get_updates(
            self.hyperoptimizer,
            gradients=[gradient], params=[self.lr]
        )

        return hyperupdates

    def _burning_in(self):
        return self.iterations <= self.burn_in_steps

    def _during_burn_in(self,
                        variable,
                        update_value):
        return K.switch(self._burning_in(), update_value, K.identity(variable))

    def get_updates(self, loss, params):
        self.all_updates = [K.update_add(self.iterations, 1)]

        n_params = n_dimensions(params)

        #  Initialize internal sampler parameters {{{ #
        self._initialize_parameters(n_params=n_params)
        self.dxdlr = K.zeros((n_params,), name="dxdlr")

        #  }}} Initialize internal sampler parameters #

        #  Sympy graph for hypergradient with respect to learning rate {{{ #
        v_hat, sympy_gradient, momentum = sympy.symbols(
            "v_hat sympy_gradient momentum"
        )

        lr, scale_grad, mdecay, noise = sympy.symbols("lr scale_grad mdecay noise")

        random_sample_ = sympy.symbols("random_sample")
        x_ = sympy.symbols("x")

        minv_ = 1. / sympy.sqrt(v_hat)

        lr_scaled = lr / sympy.sqrt(scale_grad)
        noise_scale = (
            2. * (lr_scaled ** 2) * mdecay * minv_ -
            2. * (lr_scaled ** 3) * (minv_ ** 2) * noise -
            (lr_scaled ** 4)
        )

        sigma = sympy.sqrt(noise_scale)

        sample = sigma * random_sample_

        momentum_ = (
            momentum - (lr ** 2) * minv_ * sympy_gradient -
            mdecay * momentum + sample
        )

        x_t_ = x_ + momentum_
        dxdlr_ = sympy.diff(x_t_, lr)

        x = to_vector(params)
        gradient = to_vector(K.gradients(loss, params))

        self.random_sample = K.random_normal(shape=self.momentum.shape)

        #  Hypergradient Update to tune the learning rate {{{ #

        # Run hyperoptimizer update, skip increment of iteration counter.
        _, *hyperupdates = self.hypergradient_update(
            dfdx=K.expand_dims(gradient, axis=1),
            dxdlr=K.expand_dims(self.dxdlr, axis=1)
        )

        self.all_updates.extend(hyperupdates)

        # recover tuned learning rate
        *_, lr_t = hyperupdates

        #  }}} Hypergradient Update to tune the learning rate #

        # maps sympy symbols to their corresponding tensorflow tensor
        tensors = OrderedDict([
            (v_hat, self.v_hat), (momentum, self.momentum),
            (sympy_gradient, gradient), (lr, lr_t),
            (scale_grad, self.scale_grad), (noise, self.noise),
            (mdecay, self.mdecay), (random_sample_, self.random_sample),
            (x_, x)
        ])

        #  }}} Sympy graph for hypergradient with respect to learning rate #

        with keras_control_dependencies([lr_t]):
            # Update gradient of learning rate with respect to parameters
            # by evaluating our sympy graph.
            dxdlr_t = to_tensorflow(
                dxdlr_, tuple(tensors.keys()), tuple(tensors.values())
            )
            self.all_updates.append((self.dxdlr, dxdlr_t))

            with keras_control_dependencies([dxdlr_t]):
                # SGHMC Update, skip increment of iteration counter.
                _, *sghmc_updates = super().get_updates(loss, params)
                self.all_updates.extend(sghmc_updates)

        return self.all_updates
