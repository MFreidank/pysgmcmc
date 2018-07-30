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
        print("using new")
        print("SEED:", seed)
        self.seed = seed
        self.epsilon = K.constant(lr, K.floatx())
        self.mdecay = K.constant(mdecay, K.floatx())
        self.burn_in_steps = K.constant(burn_in_steps, "int64")
        self.scale_grad = K.constant(scale_grad, K.floatx())
        self.noise = K.constant(noise, K.floatx())

        self.iterations = K.variable(0, dtype="int64", name="iterations")

    def burning_in(self):
        return self.iterations < self.burn_in_steps

    def noise_sample(self, shape):
        return K.random_normal(shape=shape, seed=self.seed)

    def get_updates(self, loss, params):
        import tensorflow as tf
        grads = [
            tf.convert_to_tensor(grad) for grad in
            self.get_gradients(loss, params)
        ]
        print("GRADS", grads)

        self.updates = [K.update_add(self.iterations, 1)]

        for (theta, grad) in zip(params, grads):
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

            self.epsilon_scaled = self.epsilon / K.sqrt(self.scale_grad)
            noise_scale = 2. * self.epsilon_scaled ** 2 * self.mdecay * Minv - 2. * self.epsilon_scaled ** 3 * K.square(Minv) * self.noise
            sigma = K.sqrt(K.maximum(noise_scale, 1e-16))
            sample_t = self.noise_sample(shape=theta.shape) * sigma
            p_t = p - self.epsilon**2 * Minv * grad - self.mdecay * p + sample_t
            theta_t = theta + p_t
            self.updates.extend([(theta, theta_t), (p, p_t)])

        return self.updates
