# vim:foldmethod=marker
import typing
from collections import OrderedDict

import sympy
from keras import backend as K
from keras.optimizers import Adam

from pysgmcmc.keras_utils import (
    keras_control_dependencies,
    n_dimensions, to_vector, sympy_to_keras
)
from pysgmcmc.optimizers.sghmc import SGHMC
from pysgmcmc.optimizers.hyperoptimization import Hyperoptimizer
from pysgmcmc.custom_typing import KerasOptimizer, KerasTensor, KerasVariable


class SGHMCHD(Hyperoptimizer, SGHMC):
    def __init__(self,
                 hyperoptimizer: KerasOptimizer=Adam(lr=1e-5),
                 hyperloss: typing.Union[None, typing.Callable[..., KerasTensor]]=None,
                 lr: float=0.01,
                 mdecay: float=0.05,
                 burn_in_steps: int=3000,
                 scale_grad: float=1.0,
                 seed: int=None,
                 **kwargs) -> None:
        """TODO: Docstring for __init__.

        Parameters
        ----------
        hyperloss: typing.Union[None, typing.Callable[..., KerasTensor]]
            TODO: Documentation
        hyperoptimizer: KerasOptimizer, optional
            Keras optimizer used to tune learning rate.
            If `hyperloss` is given it specifies the loss used to
            tune the learning rate. Otherwise, this hyperoptimizer will
            use this optimizers target loss to tune the learning rate.
            Defaults to `None`.
        lr: float, optional
            Initial leapfrog stepsize of this sampler.
            Note that this sampler tunes its stepsize using
            hypergradient optimization with a given `hyperoptimizer`.
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

        with K.name_scope(self.__class__.__name__):
            super(SGHMCHD, self).__init__(
                hyperoptimizer=hyperoptimizer, hyperloss=hyperloss,
                lr=lr, mdecay=mdecay, burn_in_steps=burn_in_steps,
                scale_grad=scale_grad, seed=seed, **kwargs
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
        TODO GIVE EXAMPLE OF A SINGLE SGHMCHD STEP AND SHOW STEPSIZE CHANGES
        AS WELL

        """
        self.sghmcd_updates = [K.update_add(self.iterations, 1)]

        n_params = n_dimensions(params)

        #  Initialize internal sampler parameters {{{ #
        self._initialize_parameters(n_params=n_params)
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

        #  Hypergradient Update to tune the learning rate {{{ #

        # Run hyperoptimizer update, skip increment of iteration counter.
        _, *hyperupdates = self.hypergradient_update(
            loss=loss, params=params,
            dxdlr=K.expand_dims(self.dxdlr, axis=1),
            hyperparameter=self.lr
        )

        self.sghmcd_updates.extend(hyperupdates)

        # recover tuned learning rate
        *_, lr_t = hyperupdates

        #  }}} Hypergradient Update to tune the learning rate #

        # maps sympy symbols to their corresponding tensorflow tensor
        tensors = OrderedDict([
            (v_hat, self.v_hat), (momentum, self.momentum),
            (sympy_gradient, gradient), (lr, lr_t),
            (scale_grad, self.scale_grad), (noise, self.noise),
            (mdecay, self.mdecay), (random_sample_, self.random_sample),
            (x_, x)
        ])

        #  }}} Sympy graph for hypergradient with respect to learning rate #

        with keras_control_dependencies([lr_t]):
            # Update gradient of learning rate with respect to parameters
            # by evaluating our sympy graph.
            dxdlr_t = sympy_to_keras(
                dxdlr_, tuple(tensors.keys()), tuple(tensors.values())
            )
            self.sghmcd_updates.append((self.dxdlr, dxdlr_t))

            with keras_control_dependencies([dxdlr_t]):
                # SGHMC Update, skip increment of iteration counter.
                _, *sghmc_updates = super().get_updates(loss, params)
                self.sghmcd_updates.extend(sghmc_updates)

        return self.sghmcd_updates
