# vim:foldmethod=marker
import typing
from keras import backend as K
from keras.optimizers import Optimizer
from pysgmcmc.keras_utils import (
    INTEGER_DTYPE, FLOAT_DTYPE,
    keras_control_dependencies,
    n_dimensions, to_vector, updates_for,
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
        """ Stochastic Gradient Hamiltonian Monte-Carlo Sampler.
            Uses a burn-in prodedure to adapt its own hyperparameters during
            the initial stages of sampling.

            See [1] for more details on this burn-in procedure.\n
            See [2] for more details on Stochastic Gradient Hamiltonian Monte Carlo.

            [1] J. T. Springenberg, A. Klein, S. Falkner, F. Hutter
                        In Advances in Neural Information Processing Systems 29 (2016).\n
                        `Bayesian Optimization with Robust Bayesian Neural Networks. <http://aad.informatik.uni-freiburg.de/papers/16-NIPS-BOHamiANN.pdf>`_

            [2] T. Chen, E. B. Fox, C. Guestrin
                In Proceedings of Machine Learning Research 32 (2014).\n
                `Stochastic Gradient Hamiltonian Monte Carlo <https://arxiv.org/pdf/1402.4102.pdf>`_

        Parameters
        ----------
        lr: float, optional
            Leapfrog stepsize parameter of this sampler.
            Defaults to `0.01`.
        mdecay: float, optional
            Controls (constant) momentum decay per time-step.
            Defaults to `0.05`.
            For reference see:
            `Bayesian Optimization with Robust Bayesian Neural Networks. <http://aad.informatik.uni-freiburg.de/papers/16-NIPS-BOHamiANN.pdf>`_
        burn_in_steps: int, optional
            Number of burn-in steps to perform.
            In each burn-in step, this sampler will adapt its own internal
            hyperparameters to decrease its error.
            Defaults to `3000`.
        scale_grad: float, optional
            Value that is used to scale the magnitude of the noise used for sampling.
            In a typical batches-of-data setting this usually corresponds to
            the number of datapoints of the entire dataset.
        seed: int, optional
            Random seed to use.
            Defaults to `None`.

        """
        super(SGHMC, self).__init__(**kwargs)
        self.seed = seed

        with K.name_scope(self.__class__.__name__):
            self.iterations = K.variable(0, dtype=INTEGER_DTYPE, name="iterations")
            self.lr = K.variable(lr, name="lr", dtype=FLOAT_DTYPE)

            #  Initialize Graph Constants {{{ #
            self.noise = K.constant(0., name="noise")

            self.scale_grad = K.constant(
                scale_grad, name="scale_grad", dtype=FLOAT_DTYPE
            )

            self.burn_in_steps = K.constant(
                burn_in_steps, name="burn_in_steps", dtype=INTEGER_DTYPE
            )

            self.mdecay = K.constant(mdecay, name="mdecay", dtype=FLOAT_DTYPE)
            #  }}} Initialize Graph Constants #

            self._initialized = False

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
        """TODO: Docstring for _during_burn_in.

        Parameters
        ----------
        variable: KerasVariable
            A keras variable that should be updated during burn-in phase.
        update_value: KerasTensor
            A value that serves to update `variable` during burn-in phase.

        Returns
        ----------
        update_tensor: KerasTensor
            TODO

        """
        return K.switch(self._burning_in(), update_value, K.identity(variable))

    def _initialize_parameters(self, n_params: int):
        """TODO: Docstring for _initialize_parameters.

        Parameters
        ----------
        n_params: int
            TODO

        """
        if not self._initialized:
            self._initialized = True
            self.tau = K.ones((n_params,), name="tau", dtype=FLOAT_DTYPE)
            self.r = K.variable(
                1. / (self.tau.initialized_value() + 1),
                name="r",
                dtype=FLOAT_DTYPE
            )
            self.g = K.ones((n_params,), name="g", dtype=FLOAT_DTYPE)
            self.v_hat = K.ones((n_params,), name="v_hat", dtype=FLOAT_DTYPE)
            self.minv = K.variable(
                1. / K.sqrt(self.v_hat.initialized_value()),
                dtype=FLOAT_DTYPE
            )
            self.momentum = K.zeros((n_params,), name="momentum", dtype=FLOAT_DTYPE)
            self.dxdlr = K.zeros((n_params,), name="dxdlr", dtype=FLOAT_DTYPE)
            self.random_sample = K.random_normal(
                shape=self.momentum.shape, seed=self.seed, dtype=FLOAT_DTYPE
            )

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
            TODO

        Examples
        ----------
        TODO GIVE EXAMPLE OF A SINGLE SGHMCHD STEP

        """
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
