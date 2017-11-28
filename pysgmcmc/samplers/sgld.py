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
            `Bayesian Optimization with Robust Bayesian Neural Networks.
            <http://aad.informatik.uni-freiburg.de/papers/16-NIPS-BOHamiANN.pdf>`_

        [2] M.Welling, Y. W. Teh
            In International Conference on Machine Learning (ICML) 28 (2011).\n
            `Bayesian Learning via Stochastic Gradient Langevin Dynamics.
            <https://www.ics.uci.edu/~welling/publications/papers/stoclangevin_v6.pdf>`_

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
            (which defines `cost`, and possibly batches).
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
        noise = tf.constant(0., name="noise", dtype=dtype)
        scale_grad = tf.constant(scale_grad, name="scale_grad", dtype=dtype)

        #  }}} Initialize graph constants #

        grads = [vectorize(gradient) for gradient in
                 tf.gradients(self.cost, params)]

        #  Initialize internal sampler parameters {{{ #

        tau = [tf.Variable(tf.ones_like(param, dtype=dtype),
                           dtype=dtype, name="tau_{}".format(i),
                           trainable=False)
               for i, param in enumerate(self.vectorized_params)]

        R = [tf.Variable(1. / (tau[i].initialized_value() + 1),
                         name="R_{}".format(i), trainable=False)
             for i, param in enumerate(self.vectorized_params)]

        g = [tf.Variable(tf.ones_like(param, dtype=dtype),
                         dtype=dtype, name="g_{}".format(i),
                         trainable=False)
             for i, param in enumerate(self.vectorized_params)]

        self.momentum = [tf.Variable(tf.ones_like(param, dtype=dtype),
                         dtype=dtype, name="v_hat_{}".format(i),
                         trainable=False)
                         for i, param in enumerate(self.vectorized_params)]

        #  Initialize mass matrix inverse {{{ #

        minv = [tf.Variable(tf.divide(tf.constant(1., dtype=dtype),
                            tf.sqrt(self.momentum[i].initialized_value())),
                            name="minv_{}".format(i), trainable=False)
                for i, param in enumerate(self.vectorized_params)]

        #  }}} Initialize mass matrix inverse #

        #  }}} Initialize internal sampler parameters #

        self.minv_t = [None] * n_params  # gets burned-in

        for i, (param, grad) in enumerate(zip(params, grads)):

            vectorized_param = self.vectorized_params[i]

            #  Burn-in logic {{{ #
            r_t = tf.assign(R[i], 1. / (tau[i] + 1.), name="r_t_{}".format(i))
            # r_t should always use the old value of tau
            with tf.control_dependencies([r_t]):
                tau_t = tf.assign_add(
                    tau[i],
                    safe_divide(-g[i] * g[i] * tau[i], self.momentum[i]) + 1,
                    name="tau_t_{}".format(i)
                )

                self.minv_t[i] = tf.assign(
                    minv[i],
                    safe_divide(1., safe_sqrt(self.momentum[i])),
                    name="minv_t_{}".format(i)
                )
                # tau_t, minv_t should always use the old values of g, g2
                with tf.control_dependencies([tau_t, self.minv_t[i]]):
                    g_t = tf.assign_add(
                        g[i],
                        -r_t * g[i] + r_t * grad,
                        name="g_t_{}".format(i)
                    )

                    v_hat_t = tf.assign_add(
                        self.momentum[i],
                        - r_t * self.momentum[i] + r_t * grad ** 2,
                        name="v_hat_t_{}".format(i)
                    )

            #  }}} Burn-in logic #
                    with tf.control_dependencies([g_t, v_hat_t]):
                        #  Draw random sample {{{ #

                        sigma = safe_sqrt(
                            2. * self.epsilon *
                            safe_divide(
                                (self.minv_t[i] * (A - noise)), scale_grad
                            )
                        )

                        sample = self._draw_noise_sample(
                            sigma=sigma, shape=vectorized_param.shape
                        )

                        #  }}} Draw random sample #

                        #  SGLD Update {{{ #

                        vectorized_theta_t = tf.assign_add(
                            vectorized_param,
                            - self.epsilon * self.minv_t[i] * A * grad + sample,
                        )
                        self.theta_t[i] = tf.assign(
                            param,
                            unvectorize(
                                vectorized_theta_t, original_shape=param.shape
                            ),
                            name="Theta_t_{}".format(i)
                        )

                        #  }}} SGLD Update #
