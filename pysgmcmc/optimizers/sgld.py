from keras import backend as K
from keras.optimizers import Optimizer
from pysgmcmc.keras_utils import vectorize, safe_division

# XXX: This does not work, find the bug and fix it


class SGLD(Optimizer):
    def __init__(self, lr=0.01, mdecay=0.05, burn_in_steps=1, scale_grad=1.0,
                 A=1.0, seed=None, **kwargs):
        super(SGLD, self).__init__(**kwargs)
        with K.name_scope(self.__class__.__name__):
            self.iterations = K.variable(0, dtype="int64", name="iterations")
            self.lr = K.variable(lr, name="lr")

            #  Initialize Graph Constants {{{ #
            self.noise = K.constant(0., name="noise")
            self.A = K.constant(A, name="A")

            self.scale_grad = K.constant(scale_grad, name="scale_grad")

            self.lr_scaled = safe_division(
                self.lr, K.sqrt(K.constant(scale_grad))
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
        gradients = self.get_gradients(loss, params)
        self.updates = [K.update_add(self.iterations, 1)]

        vectorized_params = [vectorize(param) for param in params]
        vectorized_gradients = [vectorize(gradient) for gradient in gradients]

        #  Initialize internal sampler parameters {{{ #
        tau = [
            K.variable(K.ones_like(parameter), name="tau_{}".format(i))
            for i, parameter in enumerate(vectorized_params)
        ]

        r = [
            K.variable(
                safe_division(1., (tau_i.initialized_value() + 1)),
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

        self.minv = [
            K.variable(
                safe_division(1., K.sqrt(v_hat_i.initialized_value())),
                name="minv_{}".format(i),
            )
            for i, v_hat_i in enumerate(v_hat)
        ]

        #  }}} Initialize internal sampler parameters #

        for index, (vectorized_parameter, gradient) in enumerate(zip(vectorized_params, vectorized_gradients)):
            parameter = params[index]
            tau_, r_, g_, v_hat_ = tau[index], r[index], g[index], v_hat[index]
            minv_ = self.minv[index]

            self.updates.append(
                self._update_during_burn_in(r_, safe_division(1., (tau_ + 1)))
            )
            self.updates.append(
                self._update_during_burn_in(
                    tau_, tau_ + safe_division(-g_ * g_ * tau_, v_hat_) + 1,
                )
            )
            self.updates.append(
                self._update_during_burn_in(
                    minv_, safe_division(1., K.sqrt(v_hat_))
                )
            )

            self.updates.append(
                self._update_during_burn_in(
                    g_, g_ - r_ * g_ + r_ * gradient
                )
            )

            self.updates.append(
                self._update_during_burn_in(
                    v_hat_, v_hat_ - r_ * v_hat_ + r_ * gradient ** 2
                )
            )

            noise_scale = (
                2. * self.lr *
                safe_division(
                    minv_ * (self.A - self.noise),
                    self.scale_grad
                )
            )

            sigma = K.sqrt(
                K.clip(noise_scale, min_value=1e-16, max_value=float("inf"))
            )

            sample = sigma * K.random_normal(vectorized_parameter.shape)

            vectorized_new_parameter = (
                vectorized_parameter - self.lr * minv_ * self.A * gradient + sample
            )
            self.updates.append(
                K.update(
                    parameter,
                    K.reshape(vectorized_new_parameter, parameter.shape)
                )
            )
        return self.updates

    def get_config(self):
        raise NotImplementedError()
