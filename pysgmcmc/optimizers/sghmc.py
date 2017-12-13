# vim:foldmethod=marker
import typing
from keras import backend as K
from keras.optimizers import Optimizer
from pysgmcmc.keras_utils import (
    safe_division, safe_sqrt, keras_control_dependencies,
    n_dimensions, to_vector, updates_for
)
from pysgmcmc.typing import KerasTensor, KerasVariable


class SGHMC(Optimizer):
    def __init__(self,
                 lr: float=0.01,
                 mdecay: float=0.05,
                 burn_in_steps: int=3000,
                 scale_grad: float=1.0,
                 seed: int=None,
                 **kwargs):
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

    def _during_burn_in(self,
                        variable: KerasVariable,
                        update_value: KerasTensor)-> KerasTensor:
        return K.switch(
            self.iterations <= self.burn_in_steps,
            update_value, K.identity(variable)
        )

    def get_updates(self,
                    loss: KerasTensor,
                    params: typing.List[KerasVariable]) -> typing.List[KerasTensor]:
        self.updates = [K.update_add(self.iterations, 1)]

        #  Initialize internal sampler parameters {{{ #
        n_params = n_dimensions(params)
        tau = K.ones((n_params,), name="tau")

        r = K.variable(safe_division(1., (tau.initialized_value() + 1)), name="r")

        g = K.ones((n_params,), name="g")

        v_hat = K.ones((n_params,), name="v_hat")

        minv = K.variable(safe_division(1., K.sqrt(v_hat.initialized_value())))

        momentum = K.zeros((n_params,), name="momentum")

        #  }}} Initialize internal sampler parameters #

        gradient = to_vector(self.get_gradients(loss, params))
        x = to_vector(params)

        r_t = self._during_burn_in(
            r, safe_division(1., (tau + 1.))
        )
        self.updates.append((r, r_t))

        with keras_control_dependencies([r_t]):
            tau_t = self._during_burn_in(
                tau, 1. + tau - tau * safe_division(g * g, v_hat)
            )
            self.updates.append((tau, tau_t))

            minv_t = self._during_burn_in(
                minv, safe_division(1., safe_sqrt(v_hat))
            )
            self.updates.append((minv, minv_t))

            with keras_control_dependencies([tau_t, minv_t]):
                g_t = self._during_burn_in(
                    g, g - g * r_t + r_t * gradient
                )
                self.updates.append((g, g_t))

                v_hat_t = self._during_burn_in(
                    v_hat, v_hat - v_hat * r_t + r_t * K.square(gradient)
                )
                self.updates.append((v_hat, v_hat_t))

            #  }}} Burn-in logic #

                with keras_control_dependencies([g_t, v_hat_t]):

                    #  Draw random normal sample {{{ #

                    # Equation 10, variance of normal sample

                    # 2 * epsilon ** 2 * mdecay * Minv - 0 (noise is 0) - epsilon ** 4
                    # = 2 * epsilon ** 2 * epsilon * v_hat^{-1/2} * C * Minv
                    # = 2 * epsilon ** 3 * v_hat^{-1/2} * C * v_hat^{-1/2} - epsilon ** 4

                    # (co-) variance of normal sample
                    lr_scaled = safe_division(
                        self.lr, safe_sqrt(self.scale_grad)
                    )

                    noise_scale = (
                        2. * K.square(lr_scaled) * self.mdecay * minv_t -
                        2. * K.pow(lr_scaled, 3) * K.square(minv_t) * self.noise -
                        lr_scaled ** 4
                    )

                    # turn into stddev
                    sigma = safe_sqrt(
                        K.clip(
                            noise_scale,
                            min_value=1e-16,
                            max_value=float("inf")
                        )
                    )

                    sample = sigma * K.random_normal(shape=momentum.shape)

                    #  }}} Draw random sample #

                    #  HMC Update {{{ #

                    # Equation 10: right side, where:
                    # Minv = v_hat^{-1/2}, Mdecay = epsilon * v_hat^{-1/2} C
                    momentum_t = K.update_add(
                        momentum,
                        - K.square(self.lr) * minv_t * gradient -
                        self.mdecay * momentum + sample
                    )
                    self.updates.append((momentum, momentum_t))

                    # Equation 10: left side
                    x = x + momentum_t

                    updates = updates_for(params, update_tensor=x)

                    self.updates.extend([
                        (param, K.reshape(update, param.shape))
                        for param, update in zip(params, updates)
                    ])

                    #  }}} HMC Update #
            return self.updates
