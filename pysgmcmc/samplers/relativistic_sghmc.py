# vim:foldmethod=marker
# XXX: Polish reference to paper in docstring
import numpy as np
import tensorflow as tf
from pysgmcmc.sampling import MCMCSampler

from pysgmcmc.tensor_utils import (
    vectorize, unvectorize, safe_divide, safe_sqrt
)


class RelativisticSGHMCSampler(MCMCSampler):
    def __init__(self, params, Cost, seed=None, epsilon=0.01, m=1.0, c=0.6,
                 D=1.0, scale_grad=1.0, batch_generator=None,
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

        seed : int, optional
            Random seed to use.
            Defaults to `None`.

        epsilon : float, optional
            Value that is used as learning rate parameter for the sampler,
            also denoted as discretization parameter in literature.
            Defaults to `0.01`.

        m : float, optional
            mass constant.
            Defaults to `1.0`.

        c : float, optional
            "Speed of light constant"
            Defaults to `0.6`.

        D : float, optional
            Diffusion constant
            Defaults to `1.0`.

        scale_grad : float, optional
            Value that is used to scale the magnitude of the noise used
            during sampling. In a typical batches-of-data setting this usually
            corresponds to the number of examples in the entire dataset.
            Defaults to `1.0` which corresponds to no scaling.

        batch_generator : BatchGenerator, optional
            Iterable which returns dictionaries to feed into
            tensorflow.Session.run() calls to evaluate the cost function.
            Defaults to `None` which indicates that no batches shall be fed.

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

        self.c = tf.constant(c, dtype=dtype)
        self.m = tf.constant(m, dtype=dtype)

        self.Cost = Cost

        # Epsilon = tf.constant(epsilon, dtype=dtype)
        Epsilon = tf.constant(epsilon / np.sqrt(scale_grad), dtype=dtype)

        grads = [vectorize(gradient) for gradient in
                 tf.gradients(self.Cost, params)]

        #  Sampler Variables {{{ #

        #  Noise estimate of stochastic gradient (B_hat) {{{ #

        Tau = [tf.Variable(tf.ones_like(Param, dtype=dtype),
                           dtype=dtype, name="Tau_{}".format(i),
                           trainable=False)
               for i, Param in enumerate(self.vectorized_params)]

        R = [tf.Variable(1. / (Tau[i].initialized_value() + 1),
                         name="R_{}".format(i), trainable=False)
             for i, Param in enumerate(self.vectorized_params)]

        G = [tf.Variable(tf.ones_like(Param, dtype=dtype),
                         dtype=dtype, name="G_{}".format(i),
                         trainable=False)
             for i, Param in enumerate(self.vectorized_params)]

        V_hat = [tf.Variable(tf.ones_like(Param, dtype=dtype),
                             dtype=dtype, name="V_hat_{}".format(i),
                             trainable=False)
                 for i, Param in enumerate(self.vectorized_params)]

        B_hat = [tf.Variable(0.5 * Epsilon * V_hat.initialized_value(),
                             name="B_hat_{}".format(i), dtype=dtype)
                 for i, V_hat in enumerate(V_hat)]

        #  }}} Noise estimate of stochastic gradient (B_hat) #

        #  Diffusion Terms {{{ #
        D = [tf.Variable(tf.ones_like(Param, dtype=dtype),
                         name="D_{}".format(i),
                         dtype=dtype, trainable=False)
             for i, Param in enumerate(self.vectorized_params)]

        #  }}} Diffusion Terms #

        # Momentum
        P = [tf.Variable(tf.ones_like(Param, dtype=dtype),
                         dtype=dtype, name="P_{}".format(i),
                         trainable=False)
             for i, Param in enumerate(self.vectorized_params)]

        # Mass term
        M_update = [tf.Variable(self._mass_update(P_0.initialized_value()),
                                name="Mass_update_{}".format(i),
                                dtype=dtype)
                    for i, P_0 in enumerate(P)]

        #  }}} Sampler Parameters #

        self.Theta_t = [None] * len(params)

        for i, (Param, Grad) in enumerate(zip(params, grads)):
            Vectorized_Param = self.vectorized_params[i]

            R_t = tf.assign(R[i], 1. / (Tau[i] + 1), name="R_t_{}".format(i))

            # R_t should always use the old value of Tau
            with tf.control_dependencies([R_t]):
                Tau_t = tf.assign_add(
                    Tau[i],
                    safe_divide(-G[i] * G[i] * Tau[i], V_hat[i]) + 1,
                    name="Tau_t_{}".format(i)
                )

                # Tau_t should always use the old values of G, V_hat
                with tf.control_dependencies([Tau_t]):
                    G_t = tf.assign_add(
                        G[i],
                        -R_t * G[i] + R_t * Grad,
                        name="G_t_{}".format(i)
                    )

                    V_hat_t = tf.assign_add(
                        V_hat[i],
                        - R_t * V_hat[i] + R_t * Grad ** 2,
                        name="V_hat_t_{}".format(i)
                    )

                    with tf.control_dependencies([G_t, V_hat_t]):
                        B_hat_t = tf.assign(
                            B_hat[i],
                            0.5 * Epsilon * V_hat_t
                        )

                        #  Draw random sample {{{ #

                        Noise_scale = Epsilon * (2 * D[i] - Epsilon * B_hat_t)
                        Sigma = safe_sqrt(Noise_scale)
                        Sample = self._draw_noise_sample(
                            Sigma=Sigma, Shape=Vectorized_Param.shape
                        )

                        #  }}} Draw random sample #

                        # Equation (9) upper part
                        P_t = tf.assign_add(
                            P[i],
                            -Epsilon * Grad - Epsilon * D[i] * M_update[i] +
                            Sample,
                            name="P_t_{}".format(i)
                        )

                        # Update mass term for Equation (9) lower part
                        M_update_t = tf.assign(
                            M_update[i],
                            self._mass_update(P_t),
                            name="M_update_t_{}".format(i)
                        )

                        # Equation (9) lower part
                        Vectorized_Theta_t = tf.assign_add(
                            Vectorized_Param,
                            Epsilon * M_update_t
                        )

                        self.Theta_t[i] = tf.assign(
                            Param,
                            unvectorize(
                                Vectorized_Theta_t,
                                original_shape=Param.shape
                            ),
                            name="Theta_t_{}".format(i)
                        )

    def _mass_update(self, P):
        # Equation (10) in the paper
        return safe_divide(
            P,
            tf.sqrt(
                tf.divide(
                    tf.matmul(P, P, transpose_a=True), tf.square(self.c)
                ) + tf.square(self.m)
            )
        )
