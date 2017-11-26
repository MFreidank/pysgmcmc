# vim: foldmethod=marker
import tensorflow as tf
from pysgmcmc.samplers.base_classes import BurnInMCMCSampler

from pysgmcmc.tensor_utils import (
    vectorize, unvectorize, safe_divide, safe_sqrt,
)

from pysgmcmc.stepsize_schedules import ConstantStepsizeSchedule


class SGHMCSampler(BurnInMCMCSampler):
    """ Stochastic Gradient Hamiltonian Monte-Carlo Sampler that uses a burn-in
        procedure to adapt its own hyperparameters during the initial stages
        of sampling.

        See [1] for more details on this burn-in procedure.\n
        See [2] for more details on Stochastic Gradient Hamiltonian Monte-Carlo.

        [1] J. T. Springenberg, A. Klein, S. Falkner, F. Hutter
            In Advances in Neural Information Processing Systems 29 (2016).\n

            `Bayesian Optimization with Robust Bayesian Neural Networks. <http://aad.informatik.uni-freiburg.de/papers/16-NIPS-BOHamiANN.pdf>`_


        [2] T. Chen, E. B. Fox, C. Guestrin
            In Proceedings of Machine Learning Research 32 (2014).\n
            `Stochastic Gradient Hamiltonian Monte Carlo <https://arxiv.org/pdf/1402.4102.pdf>`_
    """

    def __init__(self, params, cost_fun, batch_generator=None,
                 stepsize_schedule=ConstantStepsizeSchedule(0.01),
                 burn_in_steps=3000, mdecay=0.05, scale_grad=1.0,
                 session=tf.get_default_session(), dtype=tf.float64, seed=None):
        """ Initialize the sampler parameters and set up a tensorflow.Graph
            for later queries.

        parameters
        ----------
        params : list of tensorflow.Variable objects
            Target parameters for which we want to sample new values.

        cost_fun : callable
            Function that takes `params` as input and returns a
            1-d `tensorflow.Tensor` that contains the cost-value.
            Frequently denoted with `U` in literature.

        batch_generator : iterable, optional
            Iterable which returns dictionaries to feed into
            tensorflow.Session.run() calls to evaluate the cost function.
            Defaults to `None` which indicates that no batches shall be fed.

        stepsize_schedule : pysgmcmc.stepsize_schedules.StepsizeSchedule
            Iterator class that produces a stream of stepsize values that
            we can use in our samplers.
            See also: `pysgmcmc.stepsize_schedules`

        burn_in_steps : int, optional
            Number of burn-in steps to perform. In each burn-in step, this
            sampler will adapt its own internal parameters to decrease its error.
            Defaults to `3000`.\n
            For reference see:
            `Bayesian Optimization with Robust Bayesian Neural Networks. <http://aad.informatik.uni-freiburg.de/papers/16-NIPS-BOHamiANN.pdf>`_

        mdecay : float, optional
            (Constant) momentum decay per time-step.
            Defaults to `0.05`.\n
            For reference see:
            `Bayesian Optimization with Robust Bayesian Neural Networks. <http://aad.informatik.uni-freiburg.de/papers/16-NIPS-BOHamiANN.pdf>`_

        scale_grad : float, optional
            Value that is used to scale the magnitude of the noise used
            during sampling. In a typical batches-of-data setting this usually
            corresponds to the number of examples in the entire dataset.
            Defaults to `1.0` which corresponds to no scaling.

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
        pysgmcmc.sampling.BurnInMCMCSampler:
            Base class for `SGHMCSampler` that specifies how actual sampling
            is performed (using iterator protocol, e.g. `next(sampler)`).

        """

        # Set up BurnInMCMCSampler base class:
        # initialize member variables common to all samplers
        # and run initializers for all uninitialized variables in `params`
        # (to avoid errors in the graph definitions below).
        super().__init__(
            params=params, cost_fun=cost_fun, burn_in_steps=burn_in_steps,
            batch_generator=batch_generator,
            seed=seed, dtype=dtype, session=session,
            stepsize_schedule=stepsize_schedule
        )

        #  Initialize graph constants {{{ #

        noise = tf.constant(0., name="noise", dtype=dtype)

        scale_grad = tf.constant(scale_grad, dtype=dtype, name="scale_grad")

        epsilon_scaled = tf.divide(self.epsilon, tf.sqrt(scale_grad), name="epsilon_scaled")

        mdecay = tf.constant(mdecay, name="mdecay", dtype=dtype)

        #  }}} Initialize graph constants #

        grads = [vectorize(gradient) for gradient in
                 tf.gradients(self.cost, params)]

        #  Initialize internal sampler parameters {{{ #

        tau = [tf.Variable(tf.ones_like(param, dtype=dtype),
                           dtype=dtype, name="tau_{}".format(i),
                           trainable=False)
               for i, param in enumerate(self.vectorized_params)]

        r = [tf.Variable(1. / (tau[i].initialized_value() + 1),
                         name="R_{}".format(i), trainable=False)
             for i, param in enumerate(self.vectorized_params)]

        g = [tf.Variable(tf.ones_like(param, dtype=dtype),
                         dtype=dtype, name="g_{}".format(i),
                         trainable=False)
             for i, param in enumerate(self.vectorized_params)]

        v_hat = [tf.Variable(tf.ones_like(param, dtype=dtype),
                             dtype=dtype, name="v_hat_{}".format(i),
                             trainable=False)
                 for i, param in enumerate(self.vectorized_params)]

        # Initialize Mass matrix inverse
        minv = [tf.Variable(tf.divide(tf.constant(1., dtype=dtype),
                            tf.sqrt(v_hat[i].initialized_value())),
                            name="minv_{}".format(i), trainable=False)
                for i, param in enumerate(self.vectorized_params)]

        # Initialize momentum, denoted with v in
        # the Bohamiann paper linked in __init__ docstring
        self.momentum = [tf.Variable(tf.zeros_like(param, dtype=dtype),
                         dtype=dtype, name="momentum_{}".format(i),
                         trainable=False)
                         for i, param in enumerate(self.vectorized_params)]

        #  }}} Initialize internal sampler parameters #

        self.minv_t = [None] * len(params)  # gets burned-in
        self.v_t = [None] * len(params)  # new momentum

        # R_t = 1/ (tau + 1), shouldn't it be: 1 / tau according to terms?
        # It is not, and changing it to that breaks everything!
        # Why?

        for i, (param, grad) in enumerate(zip(params, grads)):
            vectorized_param = self.vectorized_params[i]
            #  Burn-in logic {{{ #
            r_t = tf.assign(r[i], 1. / (tau[i] + 1), name="r_t_{}".format(i))

            # r_t should always use the old value of tau
            with tf.control_dependencies([r_t]):
                tau_t = tf.assign_add(
                    tau[i],
                    safe_divide(-g[i] * g[i] * tau[i], v_hat[i]) + 1,
                    name="tau_t_{}".format(i)
                )

                # minv = v_hat^{-1/2} = 1 / sqrt(v_hat)
                self.minv_t[i] = tf.assign(
                    minv[i],
                    safe_divide(1., safe_sqrt(v_hat[i])),
                    name="minv_t_{}".format(i)
                )
                # tau_t, minv_t should always use the old values of G, v_hat
                with tf.control_dependencies([tau_t, self.minv_t[i]]):
                    g_t = tf.assign_add(
                        g[i],
                        -r_t * g[i] + r_t * grad,
                        name="g_t_{}".format(i)
                    )

                    v_hat_t = tf.assign_add(
                        v_hat[i],
                        - r_t * v_hat[i] + r_t * grad ** 2,
                        name="v_hat_t_{}".format(i)
                    )

            #  }}} Burn-in logic #

                    with tf.control_dependencies([g_t, v_hat_t]):

                        #  Draw random normal sample {{{ #

                        # Equation 10, variance of normal sample

                        # 2 * epsilon ** 2 * mdecay * Minv - 0 (noise is 0) - epsilon ** 4
                        # = 2 * epsilon ** 2 * epsilon * v_hat^{-1/2} * C * Minv
                        # = 2 * epsilon ** 3 * v_hat^{-1/2} * C * v_hat^{-1/2} - epsilon ** 4

                        # (co-) variance of normal sample
                        noise_scale = (
                            tf.constant(2., dtype=dtype) *
                            epsilon_scaled ** tf.constant(2., dtype=dtype) *
                            mdecay * self.minv_t[i] - tf.constant(2., dtype=dtype) *
                            epsilon_scaled ** tf.constant(3., dtype) *
                            tf.square(self.minv_t[i]) * noise - epsilon_scaled ** 4
                        )

                        # turn into stddev
                        sigma = tf.sqrt(tf.maximum(noise_scale, 1e-16),
                                        name="sigma_{}".format(i))

                        sample = self._draw_noise_sample(
                            sigma=sigma, shape=vectorized_param.shape
                        )

                        #  }}} Draw random sample #

                        #  HMC Update {{{ #

                        # Equation 10: right side, where:
                        # Minv = v_hat^{-1/2}, Mdecay = epsilon * v_hat^{-1/2} C
                        self.v_t[i] = tf.assign_add(
                            self.momentum[i],
                            - self.epsilon ** 2 * self.minv_t[i] * grad -
                            mdecay * self.momentum[i] + sample,
                            name="v_t_{}".format(i)
                        )

                        # Equation 10: left side
                        vectorized_Theta_t = tf.assign_add(
                            vectorized_param, self.v_t[i]
                        )

                        self.theta_t[i] = tf.assign(
                            param,
                            unvectorize(
                                vectorized_Theta_t, original_shape=param.shape
                            ),
                            name="theta_t_{}".format(i)
                        )

                        #  }}} HMC Update #
