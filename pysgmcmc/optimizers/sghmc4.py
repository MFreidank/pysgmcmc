import typing
import tensorflow as tf
from keras import backend as K
from keras.optimizers import Optimizer
from pysgmcmc.keras_utils import (
    INTEGER_DTYPE, FLOAT_DTYPE,
    keras_control_dependencies,
    n_dimensions, to_vector, updates_for,
)
from pysgmcmc.custom_typing import KerasTensor, KerasVariable
import tensorflow as tf


class SGHMC(Optimizer):
    def __init__(self,
                 lr: float=0.01,
                 mdecay: float=0.05,
                 noise: float=0.,
                 burn_in_steps: int=3000,
                 scale_grad: float=1.0,
                 seed: int=None,
                 **kwargs) -> None:
        super(SGHMC, self).__init__(**kwargs)
        self.seed = seed
        self.initial_lr = lr
        self.initial_mdecay = mdecay
        self.burn_in_steps = K.constant(burn_in_steps, dtype="int64")
        self.scale_grad = K.constant(scale_grad, dtype=K.floatx())
        self.initial_noise = noise

        self.iterations = K.variable(0, dtype="int64", name="iterations")

    def burning_in(self):
        return self.iterations < self.burn_in_steps

    def noise_sample(self, shape):
        return K.random_normal(shape=shape, seed=self.seed)

    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)

        self.updates = [K.update_add(self.iterations, 1)]

        for (theta, grad) in zip(params, grads):
            epsilon = K.constant(self.initial_lr, shape=theta.shape)
            mdecay = K.constant(self.initial_mdecay, shape=theta.shape)
            noise = K.constant(self.initial_noise, shape=theta.shape)
            xi = K.ones(theta.shape)
            g = K.ones(theta.shape)
            g2 = K.ones(theta.shape)
            p = K.zeros(theta.shape)

            r_t = 1. / (xi + 1.)

            g_t = (1. - r_t) * g + r_t * grad
            g2_t = (1. - r_t) * g2 + r_t * grad**2
            xi_t = 1. + xi * (1. - g * g / (g2 + 1e-16))
            Minv = 1. / (K.sqrt(g2 + 1e-16) + 1e-16)

            burning_in = self.iterations < self.burn_in_steps

            self.updates.extend([
                (g, K.switch(burning_in, g_t, K.identity(g))),
                (g2, K.switch(burning_in, g2_t, K.identity(g2))),
                (xi, K.switch(burning_in, xi_t, K.identity(xi))),
            ])


            epsilon_scaled = epsilon / K.sqrt(self.scale_grad)
            noise_scale = 2. * K.square(epsilon_scaled) * mdecay * Minv - 2. * epsilon_scaled ** 3 * K.square(Minv) * noise
            sigma = K.sqrt(K.maximum(noise_scale, 1e-16))
            sample_t = self.noise_sample(shape=theta.shape) * sigma
            p_t = p - K.square(epsilon) * Minv * grad - mdecay * p + sample_t
            theta_t = theta + p_t
            self.updates.extend([(theta, theta_t), (p, p_t)])

        return self.updates
