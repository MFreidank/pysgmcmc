# vim:foldmethod=marker
import typing
from keras import backend as K
from keras.optimizers import Optimizer
from pysgmcmc.keras_utils import (
    keras_control_dependencies,
    n_dimensions, to_vector, updates_for
)
from pysgmcmc.custom_typing import KerasTensor, KerasVariable


class SGHMC(Optimizer):
    def __init__(self,
                 lr: float=0.01,
                 mdecay: float=0.05,
                 burn_in_steps: int=3000,
                 scale_grad: float=1.0,
                 seed: int=None,
                 **kwargs) -> None:
        super(SGHMC, self).__init__(**kwargs)
        self.seed = seed

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

            self._initialized = False

    def _burning_in(self):
        return self.iterations <= self.burn_in_steps

    def _during_burn_in(self,
                        variable,
                        update_value):
        return K.switch(self._burning_in(), update_value, K.identity(variable))

    def _initialize_parameters(self, n_params):
        if not self._initialized:
            self._initialized = True
            self.tau = K.ones((n_params,), name="tau")
            self.r = K.variable(1. / (self.tau.initialized_value() + 1), name="r")
            self.g = K.ones((n_params,), name="g")
            self.v_hat = K.ones((n_params,), name="v_hat")
            self.minv = K.variable(1. / K.sqrt(self.v_hat.initialized_value()))
            self.momentum = K.zeros((n_params,), name="momentum")
            self.dxdlr = K.zeros((n_params,), name="dxdlr")
            self.random_sample = K.random_normal(shape=self.momentum.shape)

    def get_updates(self,
                    loss: KerasTensor,
                    params: typing.List[KerasVariable]) -> typing.List[KerasTensor]:
        self.updates = [K.update_add(self.iterations, 1)]

        n_params = n_dimensions(params)

        self._initialize_parameters(n_params=n_params)

        x = to_vector(params)
        gradient = to_vector(K.gradients(loss, params))

        #  Burn-in logic {{{ #

        r_t = self._during_burn_in(
            self.r, 1. / (self.tau + 1.)
        )
        self.updates.append((self.r, r_t))

        with keras_control_dependencies([r_t]):
            tau_t = self._during_burn_in(
                self.tau,
                1. + self.tau - self.tau *
                (self.g * self.g / self.v_hat)
            )
            self.updates.append((self.tau, tau_t))

            minv_t = self._during_burn_in(
                self.minv, 1. / K.sqrt(self.v_hat)
            )
            self.updates.append((self.minv, minv_t))

            with keras_control_dependencies([tau_t, minv_t]):
                g_t = self._during_burn_in(
                    self.g, self.g - self.g * r_t + r_t * gradient
                )
                self.updates.append((self.g, g_t))

                v_hat_t = self._during_burn_in(
                    self.v_hat,
                    self.v_hat - self.v_hat * r_t + r_t * K.square(gradient)
                )
                self.updates.append((self.v_hat, v_hat_t))

            #  }}} Burn-in logic #

                with keras_control_dependencies([g_t, v_hat_t]):

                    #  Draw random normal sample {{{ #

                    # Bohamiann paper, Equation 10: variance of normal sample

                    # 2 * epsilon ** 2 * mdecay * Minv - 0 (noise is 0) - epsilon ** 4
                    # = 2 * epsilon ** 2 * epsilon * v_hat^{-1/2} * C * Minv
                    # = 2 * epsilon ** 3 * v_hat^{-1/2} * C * v_hat^{-1/2} - epsilon ** 4

                    # (co-) variance of normal sample
                    lr_scaled = (
                        self.lr / K.sqrt(self.scale_grad)
                    )

                    noise_scale = (
                        2. * K.square(lr_scaled) * self.mdecay * minv_t -
                        2. * K.pow(lr_scaled, 3) *
                        K.square(minv_t) * self.noise - lr_scaled ** 4
                    )

                    # turn into stddev
                    sigma = K.sqrt(
                        K.clip(
                            noise_scale,
                            min_value=1e-16,
                            max_value=float("inf")
                        )
                    )

                    sample = sigma * self.random_sample

                    #  }}} Draw random sample #

                    #  Parameter Update {{{ #

                    # Equation 10: right side, where:
                    # Minv = v_hat^{-1/2}, Mdecay = epsilon * v_hat^{-1/2} C
                    momentum_t = (
                        self.momentum - K.square(self.lr) * minv_t * gradient -
                        self.mdecay * self.momentum + sample
                    )
                    self.updates.append((self.momentum, momentum_t))

                    # Equation 10: left side
                    x = x + momentum_t

                    updates = updates_for(params, update_tensor=x)

                    self.updates.extend([
                        (param, K.reshape(update, param.shape))
                        for param, update in zip(params, updates)
                    ])

                    #  }}} Parameter Update #
                    return self.updates
