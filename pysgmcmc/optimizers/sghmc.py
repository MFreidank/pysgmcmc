# vim:foldmethod=marker
from keras import backend as K
from keras.optimizers import Optimizer
from pysgmcmc.keras_utils import (
    vectorize, safe_division, safe_sqrt, keras_control_dependencies
)


class SGHMC(Optimizer):
    def __init__(self, lr=0.01, mdecay=0.05, burn_in_steps=1, scale_grad=1.0,
                 seed=None, **kwargs):
        super(SGHMC, self).__init__(**kwargs)
        self.seed = seed
        with K.name_scope(self.__class__.__name__):
            self.iterations = K.variable(0, dtype="int64", name="iterations")
            self.lr = K.variable(lr, name="lr")

            #  Initialize Graph Constants {{{ #
            self.noise = K.constant(0., name="noise")

            self.scale_grad = K.constant(scale_grad, name="scale_grad")

            self.lr_scaled = safe_division(
                self.lr, K.sqrt(self.scale_grad)
            )

            self.burn_in_steps = K.constant(
                burn_in_steps, dtype="int64", name="burn_in_steps"
            )

            self.mdecay = K.constant(mdecay, name="mdecay")

            #  }}} Initialize Graph Constants #

    def _update_during_burn_in(self, variable, update_value):
        return K.switch(
            self.iterations < self.burn_in_steps,
            update_value,
            variable
        )

    def get_updates(self, loss, params):
        self.updates = [K.update_add(self.iterations, 1)]

        vectorized_params = [vectorize(param) for param in params]
        vectorized_gradients = [
            vectorize(gradient) for gradient in self.get_gradients(loss, params)
        ]

        #  Initialize internal sampler parameters {{{ #
        tau = [
            K.variable(K.ones_like(parameter), name="tau_{}".format(i))
            for i, parameter in enumerate(vectorized_params)
        ]

        r = [
            K.variable(
                safe_division(1., (tau[i].initialized_value() + 1)),
                name="r_{}".format(i)
            )
            for i, tau_i in enumerate(tau)
        ]

        g = [
            K.variable(K.ones_like(parameter), name="g_{}".format(i))
            for i, parameter in enumerate(vectorized_params)
        ]

        v_hat = [
            K.variable(K.ones_like(param), name="v_hat_{}".format(i))
            for i, param in enumerate(vectorized_params)
        ]

        minv = [
            K.variable(
                safe_division(1., K.sqrt(v_hat_i.initialized_value())),
                name="minv_{}".format(i),
            )
            for i, v_hat_i in enumerate(v_hat)
        ]

        momentum = [
            K.variable(K.zeros_like(parameter), name="momentum_{}".format(i))
            for i, parameter in enumerate(vectorized_params)
        ]

        #  }}} Initialize internal sampler parameters #

        for i, (param, grad) in enumerate(zip(params, vectorized_gradients)):
            vectorized_param = vectorized_params[i]
            #  Burn-in logic {{{ #
            r_t = self._update_during_burn_in(
                r[i], 1. / (tau[i] + 1)
            )

            self.updates.append(r_t)

            # r_t should always use the old value of tau
            with keras_control_dependencies([r_t]):
                tau_t = self._update_during_burn_in(
                    tau[i],
                    tau[i] + safe_division(
                        -g[i] * g[i] * tau[i], v_hat[i]
                    ) + 1
                )

                # minv = v_hat^{-1/2} = 1 / sqrt(v_hat)
                minv_t = self._update_during_burn_in(
                    minv[i],
                    safe_division(1., safe_sqrt(v_hat[i]))
                )
                # tau_t, minv_t should always use the old values of G, v_hat
                with keras_control_dependencies([tau_t, minv_t]):
                    g_t = self._update_during_burn_in(
                        g[i], g[i] - r_t * g[i] + r_t * grad
                    )

                    v_hat_t = self._update_during_burn_in(
                        v_hat[i], v_hat[i] - r_t * v_hat[i] + r_t * grad ** 2
                    )

            #  }}} Burn-in logic #

                    with keras_control_dependencies([g_t, v_hat_t]):

                        #  Draw random normal sample {{{ #

                        # Equation 10, variance of normal sample

                        # 2 * epsilon ** 2 * mdecay * Minv - 0 (noise is 0) - epsilon ** 4
                        # = 2 * epsilon ** 2 * epsilon * v_hat^{-1/2} * C * Minv
                        # = 2 * epsilon ** 3 * v_hat^{-1/2} * C * v_hat^{-1/2} - epsilon ** 4

                        # (co-) variance of normal sample
                        noise_scale = (
                            2. * self.lr_scaled ** 2 * self.mdecay * minv_t -
                            2. * self.lr_scaled ** 3 *
                            K.square(minv_t) * self.noise - self.lr_scaled ** 4
                        )

                        # turn into stddev
                        sigma = K.sqrt(
                            K.clip(
                                noise_scale,
                                min_value=1e-16,
                                max_value=float("inf")
                            )
                        )

                        sample = sigma * K.random_normal(shape=vectorized_param.shape)

                        #  }}} Draw random sample #

                        #  HMC Update {{{ #

                        # Equation 10: right side, where:
                        # Minv = v_hat^{-1/2}, Mdecay = epsilon * v_hat^{-1/2} C
                        momentum_ = K.update_add(
                            momentum[i],
                            - self.lr ** 2 * minv_t * grad -
                            self.mdecay * momentum[i] + sample
                        )

                        # Equation 10: left side
                        vectorized_theta_t = vectorized_param + momentum_

                        self.updates.append(
                            K.update(
                                param,
                                K.reshape(vectorized_theta_t, param.shape)
                            )
                        )

                        #  }}} HMC Update #
        return self.updates

    def get_config(self):
        # return dict mapping hyperparameter name to hyperparameter value
        # compare keras optimizers for reference
        raise NotImplementedError()
