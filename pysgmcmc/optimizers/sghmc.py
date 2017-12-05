# vim:foldmethod=marker
import numpy as np
from keras import backend as K
from keras.optimizers import Optimizer
from pysgmcmc.keras_utils import (
    vectorize, safe_division, safe_sqrt, keras_control_dependencies
)


class SGHMC(Optimizer):
    def __init__(self, parameter_shapes,
                 lr=0.01, mdecay=0.05, burn_in_steps=3000, scale_grad=1.0,
                 seed=None, **kwargs):
        super(SGHMC, self).__init__(**kwargs)
        self.seed = seed

        vectorized_parameter_shapes = [
            (np.prod(shape), 1) for shape in parameter_shapes
        ]

        if isinstance(lr, float):
            initial_learning_rates = [lr] * len(vectorized_parameter_shapes)

        with K.name_scope(self.__class__.__name__):
            self.iterations = K.variable(0, dtype="int64", name="iterations")
            self.learning_rates = [
                K.variable(lr, name="lr")
                for lr in initial_learning_rates
            ]

            #  Initialize Graph Constants {{{ #
            self.noise = K.constant(0., name="noise")

            self.scale_grad = K.constant(scale_grad, name="scale_grad")

            self.learning_rates_scaled = [
                safe_division(
                    lr, safe_sqrt(self.scale_grad)
                )
                for lr in self.learning_rates
            ]

            self.burn_in_steps = K.constant(
                burn_in_steps, dtype="int64", name="burn_in_steps"
            )

            self.mdecay = K.constant(mdecay, name="mdecay")

            #  }}} Initialize Graph Constants #

            #  Initialize internal sampler parameters {{{ #
            self.taus = [
                K.ones(parameter_shape, name="tau_{}".format(i))
                for i, parameter_shape in enumerate(vectorized_parameter_shapes)
            ]

            self.rs = [
                K.variable(
                    safe_division(1., (tau.initialized_value() + 1)),
                    name="r_{}".format(i)
                )
                for i, tau in enumerate(self.taus)
            ]

            self.gs = [
                K.ones(parameter_shape, name="g_{}".format(i))
                for i, parameter_shape in enumerate(vectorized_parameter_shapes)
            ]

            self.v_hats = [
                K.ones(parameter_shape, name="v_hat_{}".format(i))
                for i, parameter_shape in enumerate(vectorized_parameter_shapes)
            ]

            self.minvs = [
                K.variable(
                    safe_division(1., K.sqrt(v_hat.initialized_value())),
                    name="minv_{}".format(i),
                )
                for i, v_hat in enumerate(self.v_hats)
            ]

            self.momentums = [
                K.zeros(parameter_shape, name="momentum_{}".format(i))
                for i, parameter_shape in enumerate(vectorized_parameter_shapes)
            ]

            #  }}} Initialize internal sampler parameters #

    def _update_during_burn_in(self, variable, update_value):
        return K.update(
            variable,
            K.switch(
                self.iterations <= self.burn_in_steps,
                update_value,
                K.identity(variable)
            )
        )

    def get_updates(self, loss, params):
        self.updates = [K.update_add(self.iterations, 1)]

        vectorized_params = [vectorize(param) for param in params]

        gradients = [
            vectorize(gradient) for gradient in K.gradients(loss, params)
        ]

        sghmc_parameters = zip(
            self.taus, self.rs, self.gs,
            self.v_hats, self.minvs, self.momentums,
            self.learning_rates, self.learning_rates_scaled
        )

        parameters_with_gradients = zip(vectorized_params, gradients)

        loop_variables = zip(
            params, parameters_with_gradients, sghmc_parameters
        )

        for parameter, (theta, gradient), optimizer_parameters in loop_variables:
            (tau, r, g, v_hat, minv, momentum,
             learning_rate, learning_rate_scaled) = optimizer_parameters

            #  Burn-in logic {{{ #
            r_t = self._update_during_burn_in(
                r, safe_division(1., (tau + 1.))
            )

            # r_t should always use the old value of tau
            with keras_control_dependencies([r_t]):
                tau_t = self._update_during_burn_in(
                    tau,
                    1. + tau - tau * safe_division(
                        g * g, v_hat
                    )
                )

                # minv = v_hat^{-1/2} = 1 / sqrt(v_hat)
                minv_t = self._update_during_burn_in(
                    minv,
                    safe_division(1., safe_sqrt(v_hat))
                )
                # tau_t, minv_t should always use the old values of G, v_hat
                with keras_control_dependencies([tau_t, minv_t]):
                    g_t = self._update_during_burn_in(
                        g, g - g * r_t + r_t * gradient
                    )

                    v_hat_t = self._update_during_burn_in(
                        v_hat, v_hat - v_hat * r_t + r_t * K.square(gradient)
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
                            2. * K.square(learning_rate_scaled) *
                            self.mdecay * minv_t - 2. *
                            K.pow(learning_rate_scaled, 3) * K.square(minv_t) *
                            self.noise - learning_rate_scaled ** 4
                        )

                        # turn into stddev
                        sigma = safe_sqrt(
                            K.clip(
                                noise_scale,
                                min_value=1e-16,
                                max_value=float("inf")
                            )
                        )

                        sample = sigma * K.random_normal(shape=theta.shape)

                        #  }}} Draw random sample #

                        #  HMC Update {{{ #

                        # Equation 10: right side, where:
                        # Minv = v_hat^{-1/2}, Mdecay = epsilon * v_hat^{-1/2} C
                        momentum_t = K.update_add(
                            momentum,
                            - K.square(learning_rate) * minv_t * gradient -
                            self.mdecay * momentum + sample
                        )

                        # Equation 10: left side
                        theta_t = theta + momentum_t

                        parameter_new = K.reshape(theta_t, parameter.shape)

                        if getattr(parameter, 'constraint', None) is not None:
                            parameter_new = parameter.constraint(
                                parameter_new
                            )

                        self.updates.extend([
                            (parameter, parameter_new),
                            momentum_t,
                            r_t, g_t, v_hat_t, tau_t, minv_t
                        ])

                        #  }}} HMC Update #
        return self.updates

    def get_config(self):
        config = {
            'learning_rates': list(map(float, K.batch_get_value(self.learning_rates))),
            'learning_rates_scaled': list(map(float, K.batch_get_value(self.learning_rates_scaled))),
            'noise': float(K.get_value(self.noise)),
            'scale_grad': float(K.get_value(self.scale_grad)),
            'burn_in_steps': float(K.get_value(self.burn_in_steps)),
            'iterations': float(K.get_value(self.iterations)),
            'mdecay': float(K.get_value(self.mdecay)),
        }

        config.update({
            tensor.name: K.get_value(tensor) for tensor in
            self.taus + self.rs + self.gs + self.v_hats + self.minvs +
            self.momentums
        })

        config.update(super().get_config())
        return config
