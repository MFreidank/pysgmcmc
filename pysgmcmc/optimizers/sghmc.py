from keras import backend as K
from keras.optimizers import Optimizer
from pysgmcmc.keras_utils import vectorize, safe_division


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

        # XXX Keep everything in lists? I think this might be an issue with
        # pythons reference handling

        for index, (vectorized_parameter, gradient) in enumerate(zip(vectorized_params, vectorized_gradients)):
            parameter = params[index]
            tau_, r_, g_, v_hat_ = tau[index], r[index], g[index], v_hat[index]
            minv_, momentum_ = minv[index], momentum[index]

            r_t = self._update_during_burn_in(r_, safe_division(1., (tau_ + 1)))
            self.updates.append(r_t)

            tau_t = self._update_during_burn_in(
                tau_, tau_ + safe_division(-g_ * g_ * tau_, v_hat_) + 1,
            )
            self.updates.append(tau_t)

            minv_t = self._update_during_burn_in(
                minv_, safe_division(1., K.sqrt(v_hat_))
            )
            self.updates.append(minv_t)

            g_t = self._update_during_burn_in(
                g_, g_ - r_t * g_ + r_t * gradient
            )
            self.updates.append(g_t)

            v_hat_t = self._update_during_burn_in(
                v_hat_, v_hat_ - r_t * v_hat_ + r_t * gradient ** 2
            )
            self.updates.append(v_hat_t)

            noise_scale = (
                2. * self.lr_scaled ** 2. *
                self.mdecay * minv_t - 2. *
                self.lr_scaled ** 3. *
                K.square(minv_t) * self.noise - self.lr_scaled ** 4
            )

            sigma = K.sqrt(K.clip(noise_scale, min_value=1e-16, max_value=float("inf")))

            sample = sigma * K.random_normal(
                shape=vectorized_parameter.shape, seed=self.seed
            )

            momentum_t = K.update_add(
                momentum_,
                - self.lr ** 2 * minv_t * gradient -
                self.mdecay * momentum_ + sample

            )
            self.updates.append(momentum_t)

            vectorized_new_parameter = vectorized_parameter + momentum_t

            self.updates.append(
                K.update(
                    parameter,
                    K.reshape(vectorized_new_parameter, parameter.shape)
                )
            )
        return self.updates

    def get_config(self):
        # return dict mapping hyperparameter name to hyperparameter value
        # compare keras optimizers for reference
        raise NotImplementedError()
