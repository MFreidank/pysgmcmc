# vim:foldmethod=marker
# XXX: Remove unnecessary imports

import tensorflow as tf
from pysgmcmc.sampling import MCMCSampler

from pysgmcmc.tensor_utils import (
    vectorize, unvectorize  # , safe_divide, safe_sqrt
)


class RelativisticSGHMCSampler(MCMCSampler):
    def __init__(self, params, cost_fun, epsilon=0.001, m=1.0, c=1.0,
                 D=1.0, n_iters=10, batch_generator=None, seed=None,
                 dtype=tf.float64, session=tf.get_default_session()):

        """ Relativistic Stochastic Gradient Hamiltonian Monte-Carlo Sampler.

            See [1] for more details on Relativistic SGHMC.

            [1] X. Lu, V. Perrone, L. Hasenclever, Y. W. Teh, S. J. Vollmer
                Relativistic Monte Carlo

        Parameters
        ----------
        params : list of tensorflow.Variable objects
            Target parameters for which we want to sample new values.

        Cost : tensorflow.Tensor
            1-d Cost tensor that depends on `params`.
            Frequently denoted as U(theta) in literature.

        epsilon : float, optional
            Value that is used as learning rate parameter for the sampler,
            also denoted as discretization parameter in literature.
            Defaults to `0.001`.

        m : float, optional
            mass constant.
            Defaults to `1.0`.

        c : float, optional
            "Speed of light" constant.
            Defaults to `1.0`.

        D : float, optional
            Diffusion constant.
            Defaults to `1.0`.

        n_iters : int, optional
            Number of iterations of the sampler to perform for each single
            call to `next(sampler)`.
            Defaults to `10`.

        batch_generator : BatchGenerator, optional
            Iterable which returns dictionaries to feed into
            tensorflow.Session.run() calls to evaluate the cost function.
            Defaults to `None` which indicates that no batches shall be fed.

        seed : int, optional
            Random seed to use.
            Defaults to `None`.

        dtype : tensorflow.DType, optional
            Type of elements of `tensorflow.Tensor` objects used in this sampler.
            Defaults to `tensorflow.float64`.

        session : tensorflow.Session, optional
            Session object which knows about the external part of the graph
            (which defines `Cost`, and possibly batches).
            Used internally to evaluate (burn-in/sample) the sampler.

        """

        super().__init__(params=params, batch_generator=batch_generator,
                         dtype=dtype, session=session, seed=seed)

        m = tf.constant(m, dtype=dtype)
        c = tf.constant(c, dtype=dtype)
        D = tf.constant(D, dtype=dtype)

        self.n_iters = n_iters

        stepsize = tf.constant(epsilon)

        # NOTE: This might eventually use our burn-in
        Best = tf.constant(0.0, dtype=dtype)  # Bhat variance estimator

        self.Cost = cost_fun(params)

        grads = [vectorize(gradient) for gradient in
                 tf.gradients(self.Cost, params)]

        # XXX: Needs to return a list of tensorflow.Variable objects
        p = self._sample_relativistic_momentum(
            m, c
        )

        self.Theta_t = [None] * len(params)

        for index, (Param, Grad) in enumerate(zip(self.params, grads)):
            Vectorized_Param = self.vectorized_params[index]

            s_p = p[index]

            P_grad = stepsize * tf.divide(
                s_p, (m * tf.sqrt(s_p * s_p / (m ** 2 * c ** 2) + 1))
            )

            n = tf.sqrt(
                stepsize * (2 * D - stepsize * Best) *
                tf.random_normal(shape=Param.shape, dtype=dtype, seed=seed)
            )

            P_t = tf.assign_add(
                p[index],
                stepsize * Grad + n - D * P_grad
            )

            P_grad2 = stepsize * tf.divide(
                P_t, (m * tf.sqrt(P_t * P_t / (m ** 2 * c ** 2) + 1))
            )

            Vectorized_Theta_t = tf.assign_add(
                Vectorized_Param,
                P_grad2
            )

            self.Theta_t[index] = tf.assign(
                Param,
                unvectorize(
                    Vectorized_Theta_t, original_shape=Param.shape
                ),
            )

    def __next__(self, feed_vals=dict()):
        for _ in range(self.n_iters):
            super().__next__(feed_vals=feed_vals)

        return super().__next__(feed_vals=feed_vals)

    def _sample_relativistic_momentum(self, m, c, bounds=(float("-inf"), float("inf"))):
        def relativistic_momentum_logpdf(P):
            return -m * c ** 2 * tf.sqrt(P ** 2 / (m ** 2 * c ** 2) + 1)

        # XXX: Find good python implementation of Adaptive Rejection Sampling
        # (or roll our own if that does not exist anywhere)
        def ars(logpdf, a: float, b: float, domain, n_samples: int):
            raise NotImplementedError()

        return ars(
            relativistic_momentum_logpdf, -10.0, 10.0, bounds, len(self.params)
        )
