# vim:foldmethod=marker
import typing
from keras import backend as K
from keras.optimizers import Optimizer
from arspy.ars import adaptive_rejection_sampling

from pysgmcmc.keras_utils import (
    INTEGER_DTYPE, FLOAT_DTYPE,
    n_dimensions, to_vector, updates_for,
)
from pysgmcmc.custom_typing import KerasTensor, KerasVariable


def sample_relativistic_momentum(m: float, c: float, n_params: int,
                                 bounds: typing.Tuple[float, float]=(float("-inf"), float("inf")),
                                 seed: int=None):
    def generate_relativistic_logpdf(m: float, c: float):
        def relativistic_log_pdf(p):
            from numpy import sqrt
            return -m * c ** 2 * sqrt(p ** 2 / (m ** 2 * c ** 2) + 1.)
        return relativistic_log_pdf

    momentum_log_pdf = generate_relativistic_logpdf(m=m, c=c)
    return adaptive_rejection_sampling(
        logpdf=momentum_log_pdf, a=-10.0, b=10.0,
        domain=bounds, n_samples=n_params, seed=seed
    )


class RelativisticSGHMC(Optimizer):
    def __init__(self,
                 lr: float=0.001,
                 mass: float=1.0,
                 speed_of_light: float=1.0,
                 D: float=1.0,
                 bhat: float=0.0,
                 burn_in_steps: int=3000,
                 scale_grad: float=1.0,
                 seed: int=None,
                 **kwargs) -> None:

        super(RelativisticSGHMC, self).__init__(**kwargs)

        self.seed = seed
        self.mass, self.speed_of_light = mass, speed_of_light

        with K.name_scope(self.__class__.__name__):
            self.iterations = K.variable(0, dtype=INTEGER_DTYPE, name="iterations")
            self.lr = K.variable(lr, name="lr", dtype=FLOAT_DTYPE)

            #  Initialize Graph Constants {{{ #

            self.scale_grad = K.constant(
                scale_grad, name="scale_grad", dtype=FLOAT_DTYPE
            )

            self.burn_in_steps = K.constant(
                burn_in_steps, name="burn_in_steps", dtype=INTEGER_DTYPE
            )

            self.D = K.constant(D, name="D")
            self.bhat = K.constant(bhat, name="bhat")
            self.m = K.constant(mass, name="mass")
            self.c = K.constant(speed_of_light, name="speed_of_light")

            #  }}} Initialize Graph Constants #

    def get_updates(self,
                    params: typing.List[KerasVariable],
                    loss: KerasTensor) -> typing.List[KerasTensor]:
        self.updates = [K.update_add(self.iterations, 1)]

        x = to_vector(params)

        log_likelihood = -loss

        gradient = to_vector(K.gradients(log_likelihood, params))

        n_params = n_dimensions(params)

        p = K.variable(
            sample_relativistic_momentum(
                m=self.mass, c=self.speed_of_light,
                n_params=n_params, seed=self.seed
            )
        )

        p_grad = (
            self.lr * p /
            (self.m * K.sqrt(p * p / (K.square(self.m) * K.square(self.c) + 1)))
        )

        random_sample = K.sqrt(
            self.lr * (2. * self.D - self.lr * self.bhat) *
            K.random_normal(gradient.shape)
        )

        p_t = K.update_add(
            p, self.lr * gradient + random_sample - self.D * p_grad
        )

        self.updates.append(p_t)

        p_grad_new = (
            self.lr * p_t /
            (self.m * K.sqrt(p_t * p_t / (K.square(self.m) * K.square(self.c)) + 1.))
        )

        x = x + p_grad_new

        updates = updates_for(params, update_tensor=x)

        self.updates.extend([
            (param, K.reshape(update, param.shape))
            for param, update in zip(params, updates)
        ])

        return self.updates

        #  }}} Parameter Update #
