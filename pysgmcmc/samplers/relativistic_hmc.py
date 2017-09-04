import tensorflow as tf
from pysgmcmc.sampling import MCMCSampler

from pysgmcmc.tensor_utils import vectorize, unvectorize


class RelativisticHMCSampler(MCMCSampler):
    def __init__(self, params, cost_fun, seed=None, epsilon=0.01,
                 m=1., c=1.,
                 session=tf.get_default_session(), dtype=tf.float64):
        super().__init__(params=params, seed=seed, dtype=dtype, session=session)

        Epsilon = tf.constant(epsilon, dtype=dtype)
        self.Cost = cost_fun(params)

        grads = [vectorize(gradient) for gradient in
                 tf.gradients(self.Cost, params)]

        m = tf.constant(m, dtype=dtype)
        c = tf.constant(c, dtype=dtype)

        # Momentum
        P = [tf.Variable(tf.zeros_like(Param, dtype=dtype),
                         dtype=dtype, name="P_{}".format(i),
                         trainable=False)
             for i, Param in enumerate(self.vectorized_params)]

        n_params = len(params)
        self.P_t = [None] * n_params
        self.Theta_t = [None] * n_params

        for i, (Param, Grad) in enumerate(zip(params, grads)):
            Vectorized_Param = self.vectorized_params[i]

            P_half = tf.assign_add(
                P[i],
                -0.5 * Epsilon * Grad
            )

            Minv = tf.divide(
                1.,
                tf.sqrt(
                    tf.divide(
                        tf.matmul(P_half, P_half, transpose_a=True),
                        tf.square(m) * tf.square(c)
                    ) + 1.
                )
            )

            Vectorized_Theta_t = tf.assign_add(
                Vectorized_Param,
                Epsilon * Minv * P_half
            )

            self.Theta_t[i] = tf.assign(
                Param,
                unvectorize(
                    Vectorized_Theta_t, original_shape=Param.shape
                ),
                name="Theta_t_{}".format(i)
            )

        with tf.control_dependencies(self.Theta_t):

            cost_next = cost_fun(self.Theta_t)

            grad_next = [vectorize(gradient) for gradient in tf.gradients(cost_next, self.Theta_t)]

            for i in range(len(self.P_t)):
                self.P_t[i] = tf.assign_add(
                    P[i],
                    -0.5 * Epsilon * grad_next[i]
                )

    def __next__(self):
        params, cost, _ = self.session.run(
            [self.Theta_t, self.Cost, self.P_t]
        )

        if len(params) == 1:
            # unravel single-element lists to scalars
            params = params[0]

        return params, cost
