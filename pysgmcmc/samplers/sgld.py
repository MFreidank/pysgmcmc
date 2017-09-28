# vim: foldmethod=marker

import tensorflow as tf
from pysgmcmc.samplers.base_classes import BurnInMCMCSampler
from pysgmcmc.stepsize_schedules import ConstantStepsizeSchedule

from pysgmcmc.tensor_utils import (
    vectorize, unvectorize, safe_divide, safe_sqrt
)


class SGLDSampler(BurnInMCMCSampler):
    """ Stochastic Gradient Langevin Dynamics Sampler that uses a burn-in
        procedure to adapt its own hyperparameters during the initial stages
        of sampling.

        See [1] for more details on this burn-in procedure.
        See [2] for more details on Stochastic Gradient Langevin Dynamics.

        [1] J. T. Springenberg, A. Klein, S. Falkner, F. Hutter
            In Advances in Neural Information Processing Systems 29 (2016).\n
            `Bayesian Optimization with Robust Bayesian Neural Networks. <http://aad.informatik.uni-freiburg.de/papers/16-NIPS-BOHamiANN.pdf>`_

        [2] M.Welling, Y. W. Teh
            In International Conference on Machine Learning (ICML) 28 (2011).\n
            `Bayesian Learning via Stochastic Gradient Langevin Dynamics. <https://www.ics.uci.edu/~welling/publications/papers/stoclangevin_v6.pdf>`_

    """

    def __init__(self, params, cost_fun, batch_generator=None,
                 stepsize_schedule=ConstantStepsizeSchedule(0.01),
                 burn_in_steps=3000, A=1.0, scale_grad=1.0,
                 session=tf.get_default_session(), dtype=tf.float64, seed=None):
        """ Initialize the sampler parameters and set up a tensorflow.Graph
            for later queries.

        Parameters
        ----------
        params : list of tensorflow.Variable objects
            Target parameters for which we want to sample new values.

        cost_fun : callable
            Function that takes `params` as input and returns a
            1-d `tensorflow.Tensor` that contains the cost-value.
            Frequently denoted with `U` in literature.

        batch_generator : BatchGenerator, optional
            Iterable which returns dictionaries to feed into
            tensorflow.Session.run() calls to evaluate the cost function.
            Defaults to `None` which indicates that no batches shall be fed.

        stepsize_schedule : pysgmcmc.stepsize_schedules.StepsizeSchedule
            Iterator class that produces a stream of stepsize values that
            we can use in our samplers.
            See also: `pysgmcmc.stepsize_schedules`

        burn_in_steps: int, optional
            Number of burn-in steps to perform. In each burn-in step, this
            sampler will adapt its own internal parameters to decrease its error.
            Defaults to `3000`.\n
            For reference see:
            `Bayesian Optimization with Robust Bayesian Neural Networks. <http://aad.informatik.uni-freiburg.de/papers/16-NIPS-BOHamiANN.pdf>`_

        A : float, optional
            TODO Doku
            Defaults to `1.0`.

        scale_grad : float, optional
            Value that is used to scale the magnitude of the noise used
            during sampling. In a typical batches-of-data setting this usually
            corresponds to the number of examples in the entire dataset.

        session : tensorflow.Session, optional
            Session object which knows about the external part of the graph
            (which defines `Cost`, and possibly batches).
            Used internally to evaluate (burn-in/sample) the sampler.

        dtype : tensorflow.DType, optional
            Type of elements of `tensorflow.Tensor` objects used in this sampler.
            Defaults to `tensorflow.float64`.

        seed : int, optional
            Random seed to use.
            Defaults to `None`.

        See Also
        ----------
        tensorflow_mcmc.sampling.mcmc_base_classes.BurnInMCMCSampler:
            Base class for `SGLDSampler` that specifies how actual sampling
            is performed (using iterator protocol, e.g. `next(sampler)`).

        """

        super().__init__(
            params=params, cost_fun=cost_fun, batch_generator=batch_generator,
            burn_in_steps=burn_in_steps, seed=seed,
            session=session, dtype=dtype
        )

        n_params = len(params)

        #  Initialize graph constants {{{ #

        A = tf.constant(A, name="A", dtype=dtype)
        Noise = tf.constant(0., name="noise", dtype=dtype)
        Scale_grad = tf.constant(scale_grad, name="scale_grad", dtype=dtype)

        #  }}} Initialize graph constants #

        grads = [vectorize(gradient) for gradient in
                 tf.gradients(self.Cost, params)]

        #  Initialize internal sampler parameters {{{ #

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

        #  Initialize mass matrix inverse {{{ #

        self.Minv = [tf.Variable(tf.divide(tf.constant(1., dtype=dtype),
                                 tf.sqrt(V_hat[i].initialized_value())),
                                 name="Minv_{}".format(i), trainable=False)
                     for i, Param in enumerate(self.vectorized_params)]

        #  }}} Initialize mass matrix inverse #

        #  }}} Initialize internal sampler parameters #

        self.Minv_t = [None] * n_params  # gets burned-in

        for i, (Param, Grad) in enumerate(zip(params, grads)):

            Vectorized_Param = self.vectorized_params[i]

            #  Burn-in logic {{{ #
            R_t = tf.assign(R[i], 1. / (Tau[i] + 1.), name="R_t_{}".format(i))
            # R_t should always use the old value of Tau
            with tf.control_dependencies([R_t]):
                Tau_t = tf.assign_add(
                    Tau[i],
                    safe_divide(-G[i] * G[i] * Tau[i], V_hat[i]) + 1,
                    name="Tau_t_{}".format(i)
                )

                self.Minv_t[i] = tf.assign(
                    self.Minv[i],
                    safe_divide(1., safe_sqrt(V_hat[i])),
                    name="Minv_t_{}".format(i)
                )
                # Tau_t, Minv_t should always use the old values of G, G2
                with tf.control_dependencies([Tau_t, self.Minv_t[i]]):
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

            #  }}} Burn-in logic #
                    with tf.control_dependencies([G_t, V_hat_t]):
                        #  Draw random sample {{{ #

                        Sigma = safe_sqrt(
                            2. * self.Epsilon *
                            safe_divide(
                                (self.Minv_t[i] * (A - Noise)), Scale_grad
                            )
                        )

                        Sample = self._draw_noise_sample(
                            Sigma=Sigma, Shape=Vectorized_Param.shape
                        )

                        #  }}} Draw random sample #

                        #  SGLD Update {{{ #

                        Vectorized_Theta_t = tf.assign_add(
                            Vectorized_Param,
                            - self.Epsilon * self.Minv_t[i] * A * Grad + Sample,
                        )
                        self.Theta_t[i] = tf.assign(
                            Param,
                            unvectorize(
                                Vectorized_Theta_t, original_shape=Param.shape
                            ),
                            name="Theta_t_{}".format(i)
                        )

                        #  }}} SGLD Update #
