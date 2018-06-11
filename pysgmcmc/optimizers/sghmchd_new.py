# vim:foldmethod=marker
import typing
from keras import backend as K
from keras.optimizers import Optimizer
from pysgmcmc.keras_utils import (
    INTEGER_DTYPE, FLOAT_DTYPE,
    keras_control_dependencies,
    n_dimensions, to_vector, updates_for,
)
from collections import OrderedDict
from pysgmcmc.custom_typing import KerasTensor, KerasVariable
from pysgmcmc.optimizers.sghmc import SGHMC
# from pysgmcmc.optimizers.hyperoptimization import to_hyperoptimizer
import sympy

from keras.optimizers import Adam


def keras_heaviside(x):
    flag1 = x == 0
    flag2 = K.cast(x > 0, "float32")
    print(flag1, flag2)
    return K.ones_like(x) * flag1 + (0.5 * flag2)


def to_hyperoptimizer(optimizer):

    def hypergradient_update(hyperoptimizer, gradients, params):
        hyperoptimizer.get_gradients = lambda *args, **kwargs: gradients
        return hyperoptimizer.get_updates(loss=None, params=params)

    optimizer.hypergradient_update = lambda *args, **kwargs: hypergradient_update(optimizer, *args, **kwargs)

    return optimizer


class Hyperoptimizer(object):
    def __init__(self, optimizer=Adam(lr=1e-5)):
        self.optimizer = optimizer

    def hypergradient_update(self, gradients, params):

        self.optimizer.get_gradients = lambda *args, **kwargs: gradients

        return self.optimizer.get_updates(loss=None, params=params)


class SGHMCHD(Optimizer):

    def __init__(self,
                 hyperoptimizer=Adam(lr=1e-5),
                 hypergradients_for=("lr",),
                 lr: float=0.01,
                 mdecay: float=0.05,
                 burn_in_steps: int=3000,
                 scale_grad: float=1.0,
                 seed: int=None,
                 **kwargs) -> None:
        super(SGHMCHD, self).__init__(**kwargs)
        self.seed = seed

        self.hypergradients_for = hypergradients_for

        with K.name_scope(self.__class__.__name__):
            self.iterations = K.variable(0, dtype=INTEGER_DTYPE, name="iterations")
            self.lr_value = lr
            # self.lr = K.variable(lr, name="lr", dtype=FLOAT_DTYPE)

            #  Initialize Graph Constants {{{ #
            self.noise = K.constant(0., name="noise")

            self.scale_grad_value = scale_grad
            self.scale_grad = K.constant(
                scale_grad, name="scale_grad", dtype=FLOAT_DTYPE
            )

            self.burn_in_steps = K.constant(
                burn_in_steps, name="burn_in_steps", dtype=INTEGER_DTYPE
            )

            self.mdecay_value = mdecay

            # self.mdecay = K.constant(mdecay, name="mdecay", dtype=FLOAT_DTYPE)
            #  }}} Initialize Graph Constants #

            self._initialized = False
            self.hyperoptimizer = to_hyperoptimizer(hyperoptimizer)

    def _burning_in(self):
        """ Return a boolean keras tensor that is `True` only during burn-in phase.

        Returns
        ----------
        is_burning_in: KerasTensor
            Boolean keras tensor that is `True` only during burn-in phase.
            Burn-in phase ends when `self.iterations > self.burn_in_steps`.

        Examples
        ----------
        For a positive amount of burn-in steps, this is `True` initially:

        >>> from keras import backend as K
        >>> sampler = SGHMC(burn_in_steps=1)
        >>> K.get_value(sampler._burning_in())
        True

        If the number of performed iterations is equal to the number of
        burn-in steps, it becomes `False`:

        >>> from keras import backend as K
        >>> sampler = SGHMC(burn_in_steps=0)
        >>> K.get_value(sampler._burning_in())
        False


        """
        return self.iterations < self.burn_in_steps

    def _during_burn_in(self,
                        variable: KerasVariable,
                        update_value: KerasTensor) -> KerasTensor:
        """ Return `update_value` during burn-in phase else `keras.backend.identity(variable)`.

        Parameters
        ----------
        variable: KerasVariable
            A keras variable that should be updated during burn-in phase.
        update_value: KerasTensor
            A value that serves to update `variable` during burn-in phase.

        Returns
        ----------
        update_tensor: KerasTensor
            `update_value` if `self._burning_in()` is true,
            `keras.backend.identity(variable)` otherwise.

        """
        return K.switch(self._burning_in(), update_value, K.identity(variable))

    def _initialize_parameters(self, num_target_params: int):
        if not self._initialized:
            self._initialized = True
            self.tau = K.ones((num_target_params,), name="tau", dtype=FLOAT_DTYPE)
            self.r = K.variable(
                1. / (self.tau.initialized_value() + 1),
                name="r",
                dtype=FLOAT_DTYPE
            )
            self.g = K.ones((num_target_params,), name="g", dtype=FLOAT_DTYPE)
            self.v_hat = K.ones((num_target_params,), name="v_hat", dtype=FLOAT_DTYPE)
            self.minv = K.variable(
                1. / K.sqrt(self.v_hat.initialized_value()),
                dtype=FLOAT_DTYPE
            )
            self.momentum = K.zeros((num_target_params,), name="momentum", dtype=FLOAT_DTYPE)
            self.dxdlr = K.zeros((num_target_params,), name="dxdlr", dtype=FLOAT_DTYPE)
            self.random_sample = K.random_normal(
                shape=self.momentum.shape, seed=self.seed, dtype=FLOAT_DTYPE
            )
            self.lr = K.variable(
                K.ones((num_target_params,)) * self.lr_value, name="lr", dtype=FLOAT_DTYPE
            )

            #  Initialize Graph Constants {{{ #
            # self.noise = K.constant(0., name="noise")

            self.mdecay = K.variable(
                K.ones((num_target_params,)) * self.mdecay_value, name="mdecay", dtype=FLOAT_DTYPE
            )
            #  }}} Initialize Graph Constants #

            from collections import OrderedDict

            # XXX: Check that the values here are updated properly on iterations..
            self.keras_tensors = OrderedDict((
                ("v_hat", self.v_hat), ("momentum", self.momentum), ("lr", self.lr,),
                ("mdecay", self.mdecay), ("noise", self.noise),
                ("random_sample", self.random_sample)
            ))

            self.dxdh = OrderedDict(
                (self.keras_tensors[tensor_name], K.zeros((num_target_params,)))
                for tensor_name in self.hypergradients_for
            )


    def sympy_derivative(self, tensor, keras_tensors):
        symbols = OrderedDict(
            (tensorname, sympy.symbols(tensorname))
            for tensorname in keras_tensors
        )
        print(symbols)

        v_hat = symbols["v_hat"]
        momentum, lr = symbols["momentum"], symbols["lr"]
        mdecay, noise = symbols["mdecay"], symbols["noise"]
        gradient, random_sample = symbols["gradient"], symbols["random_sample"]

        minv_t = 1. / sympy.sqrt(v_hat)
        lr_scaled = lr / sympy.sqrt(self.scale_grad_value)

        noise_scale = (
            2. * (lr_scaled ** 2) * mdecay * minv_t -
            2. * (lr_scaled ** 3) * (minv_t ** 2) * noise -
            (lr_scaled ** 4)
        )

        # TODO: Poor man's clipping?
        sigma = sympy.sqrt(sympy.Max(noise_scale, 1e-16))
        sample = sigma * random_sample

        momentum = (
            momentum - (lr ** 2) * minv_t * gradient - mdecay * momentum + sample
        )

        tensorname = None
        for name, tensor_ in keras_tensors.items():
            if tensor == tensor_:
                tensorname = name
                break

        sympy_derivative = sympy.diff(momentum, symbols[tensorname])
        print(sympy_derivative)
        from math import sqrt as math_sqrt
        return sympy.lambdify(
            args=symbols.values(),
            expr=sympy_derivative,
            modules={
                "sqrt": lambda v: math_sqrt(v) if isinstance(v, int) else K.sqrt(v),
                "Heaviside": keras_heaviside,
                "Max": lambda v1, v2: K.clip(v2, min_value=v1, max_value=float("inf"))
            }
        )(*keras_tensors.values())

    def get_updates(self,
                    loss: KerasTensor,
                    params: typing.List[KerasVariable]) -> typing.List[KerasTensor]:
        """ Perform an update iteration of this optimizer.
            Update `params` and internal hyperparameters to minimize `loss`.

        Parameters
        ----------
        loss: KerasTensor
            Tensor representing a loss value that should be minimized.
            Loss should depend on `params`.
        params: typing.List[KerasVariable]
            List of parameters that we want to update to minimize `loss`.

        Returns
        ----------
        updates: typing.List[KerasTensor]
            List of tensors that specify assignments to all internal and
            target parameters of this sampler.

        Examples
        ----------
        TODO GIVE EXAMPLE OF A SINGLE SGHMC STEP

        """

        self.updates = [K.update_add(self.iterations, 1)]
        num_params = n_dimensions(params)
        self._initialize_parameters(num_target_params=num_params)
        x = to_vector(params)
        gradient = to_vector(K.gradients(loss, params))


        # Chain Rule application: dfdh = dxdh * dfdx
        # where dfdx == gradient
        hyperparameters = self.dxdh.keys()
        dfdh = [dxdh * gradient for dxdh in self.dxdh.values()]

        # Update hyperparameters with corresponding hypergradients.
        hyperupdates = self.hyperoptimizer.hypergradient_update(
            gradients=dfdh, params=hyperparameters
        )

        self.updates.extend(hyperupdates)

        with keras_control_dependencies(hyperupdates):
            self.keras_tensors["gradient"] = gradient
            # Compute derivatives of parameters x with respect to hyperparameters.
            for hyperparameter, dxdh in self.dxdh.items():
                dfdh = dxdh * gradient

                self.updates.append(
                    (dxdh, self.sympy_derivative(hyperparameter, self.keras_tensors))
                )


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
