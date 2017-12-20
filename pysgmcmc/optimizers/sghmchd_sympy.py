# vim:foldmethod=marker
import typing
import sympy
from keras import backend as K
from keras.optimizers import Optimizer, Adam
from pysgmcmc.keras_utils import (
    keras_control_dependencies,
    n_dimensions, to_vector, updates_for
)
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


class SGHMCHD(Optimizer):
    def __init__(self,
                 hyperoptimizer=Adam(lr=1e-5, clipnorm=1.),
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
        self.updates = [K.update_add(self.iterations, 1)]

        n_params = n_dimensions(params)

        #  Initialize internal sampler parameters {{{ #
        self.tau = K.ones((n_params,), name="tau")

        self.r = K.variable(1. / (self.tau.initialized_value() + 1), name="r")

        self.g = K.ones((n_params,), name="g")

        self.v_hat = K.ones((n_params,), name="v_hat")

        self.minv = K.variable(1. / K.sqrt(self.v_hat.initialized_value()))

        self.momentum = K.zeros((n_params,), name="momentum")

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

        random_sample = K.random_normal(shape=self.momentum.shape)

        #  Hypergradient Update to tune the learning rate {{{ #

        # Run hyperoptimizer update.
        hyperupdates = self.hypergradient_update(
            dfdx=K.expand_dims(gradient, axis=1),
            dxdlr=K.expand_dims(self.dxdlr, axis=1)
        )

        self.updates.extend(hyperupdates)

        # recover tuned learning rate
        *_, lr_t = hyperupdates

        #  }}} Hypergradient Update to tune the learning rate #

        # maps sympy symbols to their corresponding tensorflow tensor
        tensors = OrderedDict([
            (v_hat, self.v_hat), (momentum, self.momentum),
            (sympy_gradient, gradient), (lr, lr_t),
            (scale_grad, self.scale_grad), (noise, self.noise),
            (mdecay, self.mdecay), (random_sample_, random_sample), (x_, x)
        ])

        #  }}} Sympy graph for hypergradient with respect to learning rate #

        with keras_control_dependencies([lr_t]):
            # Update gradient of learning rate with respect to parameters
            # by evaluating our sympy graph.
            dxdlr_t = to_tensorflow(
                dxdlr_, tuple(tensors.keys()), tuple(tensors.values())
            )
            self.updates.append((self.dxdlr, dxdlr_t))

            #  Standard SGHMC Update {{{ #

            #  Burn-in logic {{{ #

            with keras_control_dependencies([dxdlr_t]):
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
                                lr_t / K.sqrt(self.scale_grad)
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

                            sample = sigma * random_sample

                            #  }}} Draw random sample #

                            #  Parameter Update {{{ #

                            # Equation 10: right side, where:
                            # Minv = v_hat^{-1/2}, Mdecay = epsilon * v_hat^{-1/2} C
                            momentum_t = (
                                self.momentum - K.square(lr_t) * minv_t * gradient -
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

            #  }}} Standard SGHMC Update #

                    return self.updates
