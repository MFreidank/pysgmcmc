import tensorflow as tf
from pysgmcmc.sampling import MCMCSampler

from pysgmcmc.tensor_utils import vectorize, unvectorize


class RelativisticHMCSampler(MCMCSampler):
    def __init__(self, params, cost_fun,
                 epsilon=0.001,
                 speed_of_light=1.0,
                 mass=1.0,
                 n_iters=10,
                 seed=None, batch_generator=None,
                 dtype=tf.float64, session=tf.get_default_session()):
        super().__init__(
            params, seed=seed, batch_generator=batch_generator,
            dtype=dtype, session=session
        )

        Epsilon = tf.constant(epsilon, dtype=dtype)
        C = tf.constant(speed_of_light, dtype=dtype)
        Mass = tf.constant(mass, dtype=dtype)

        self.Cost = cost_fun(params)

        # XXX: Initialize momentum using sample_rel_p
        grads = [
            vectorize(gradient)
            for gradient in tf.grads(self.Cost, params)
        ]

        # XXX: Update momentum using grads
        for Momentum, Grad in zip(momentums, grads):
            pass

        # XXX: tensorflow while loop that updates
        # variables

    def relativistic_momentum_logpdf(self, M, C, P):
        sqrt_term = tf.sqrt(
            P ** 2 / (M ** 2 * C ** 2) + 1
        )
        return -M * tf.square(C) * sqrt_term

    def sample_relativistic_momentum(self, M, C, n_params, bounds=(float("-inf"), float("inf"))):
        # XXX: Perform ars magic here
        raise NotImplementedError("Do we really need this to use tensorflow?")
