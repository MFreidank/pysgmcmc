import tensorflow as tf
from pysgmcmc.sampling import MCMCSampler


class RelativisticHMCSampler(MCMCSampler):
    def __init__(self, params, cost_fun, batch_generator=None, seed=None,
                 epsilon=0.001, niters=10, speed_of_light=1.0, mass=1.0,
                 dtype=tf.float64, session=tf.get_default_session()):

        super().__init__(
            params=params,
            seed=seed, batch_generator=batch_generator,
            dtype=dtype, session=session
        )

        Epsilon = tf.constant(epsilon, dtype=dtype)

        # self.Cost = cost_fun(params)






