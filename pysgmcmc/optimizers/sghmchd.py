# vim:foldmethod=marker
from keras import backend as K
from keras.optimizers import Optimizer, Adam

from pysgmcmc.keras_utils import (
    keras_control_dependencies, to_vector, n_dimensions,
    safe_division, safe_sqrt, updates_for
)

from pysgmcmc.optimizers import to_metaoptimizer


# XXX: Compute correct gradient wrt learning rate for sghmc and use it
# to do metaupdates for lr_t


class SGHMCHD(Optimizer):
    # Hypergradient descent version of SGHMC
    def __init__(self, lr=0.0, burn_in_steps=3000, mdecay=0.05,
                 scale_grad=1.0, metaoptimizer=Adam(), seed=None, **kwargs):
        super().__init__(**kwargs)
        self.seed = seed

        with K.name_scope(self.__class__.__name__):
            self.iterations = K.variable(0, dtype="int64", name="iterations")
            self.lr = K.variable(lr, name="lr")
            self.metaoptimizer = to_metaoptimizer(metaoptimizer)

            #  Initialize Graph constants {{{ #

            self.noise = K.constant(0., name="noise")
            self.scale_grad = K.constant(scale_grad, name="scale_grad")

            self.burn_in_steps = K.constant(burn_in_steps, dtype="int64", name="burn_in_steps")
            self.mdecay = K.constant(mdecay, name="mdecay")

            #  }}} Initialize Graph constants #

    def hypergradient_update(self, dfdx, dxdlr):
        gradient = K.reshape(K.dot(K.transpose(dfdx), dxdlr), self.lr.shape)
        metaupdates = self.metaoptimizer.get_updates(
            self.metaoptimizer, gradients=[gradient], params=[self.lr]
        )

        self.updates.extend(metaupdates)
        *_, lr_t = metaupdates
        return lr_t

    def during_burn_in(self, variable, update_value):
        return K.switch(
            self.iterations <= self.burn_in_steps,
            update_value, K.identity(variable)
        )

    def get_updates(self, loss, params):
        self.updates = [K.update_add(self.iterations, 1)]

        n_params = n_dimensions(params)

        #  Initialize internal sampler parameters {{{ #
        tau = K.ones(n_params, name="tau")
        # XXX Is this even needed?
        r = K.variable(
            safe_division(1., (tau.initialized_value() + 1)), name="r"
        )

        g = K.ones(n_params, name="g")
        v_hat = K.ones(n_params, name="v_hat")
        minv = K.variable(safe_division(1., K.sqrt(v_hat.initialized_value())), name="minv")

        momentum = K.zeros(n_params, name="momentum")
        #  }}} Initialize internal sampler parameters #

        x = to_vector(params)
        dfdx = K.expand_dims(to_vector(K.gradients(loss, params)), axis=1)

        # lr_t = self.hypergradient_update(dfdx=dfdx, dxdlr=-self.dxdlr)
        lr_t = self.learning_rate


        # XXX: SGHMC Update using new learning rate lr_t
        r_t = self._during_burn_in(r, safe_division(1. (tau + 1.)))
        self.updates.append(K.update(r, r_t))

        with keras_control_dependencies([r_t]):
            tau_t = self._during_burn_in(
                tau, 1. + tau - tau * safe_division(g * g, v_hat)
            )

            self.updates.append(K.update(tau, tau_t))

            minv_t = self._during_burn_in(
                minv, safe_division(1., safe_sqrt(v_hat))
            )

            self.updates.append(K.update(minv, minv_t))

            with keras_control_dependencies([tau_t, minv_t]):
                g_t = self._during_burn_in(
                    g, g - g * r_t + r_t * dfdx
                )

                self.updates.append(K.update(minv, minv_t))

                v_hat_t = self._during_burn_in(
                    v_hat, v_hat - v_hat * r_t + r_t * K.square(dfdx)
                )

                self.updates.append(K.update(v_hat, v_hat_t))

                with keras_control_dependencies([g_t, v_hat_t]):
                    lr_scaled = safe_division(
                        lr_t, safe_sqrt(self.scale_grad)
                    )
                    noise_scale = (
                        2. * K.square(lr_scaled) * self.mdecay * minv_t -
                        2. * K.pow(lr_scaled, 3) * K.square(minv_t) *
                        self.noise - lr_scaled ** 4
                    )

                    sigma = safe_sqrt(
                        K.clip(
                            noise_scale,
                            min_value=1e-16,
                            max_value=float("inf")
                        )
                    )

                    sample = sigma * K.random_normal(shape=momentum.shape)

                    momentum_t = (
                        momentum - K.square(lr_t) * minv_t * dfdx -
                        self.mdecay * momentum + sample
                    )
                    self.updates.append(K.update(momentum, momentum_t))

                    x = x + momentum_t

                    self.updates.extend(updates_for(params, update_tensor=x))
                    # XXX: Updates for learning rate not yet done, check
                    # that base sghmc works when implemented this way,
                    # then proceed towards tuning learning rate
                    return self.updates
