#!/usr/bin/python3
# -*- coding: iso-8859-15 -*-
"""Abstract base classes for all MCMC methods. Helps unify our sampler interface."""
# XXX NEXT METHOD NEEDS DOKU FOR FEED_DICT
import abc
import tensorflow as tf

from pysgmcmc.tensor_utils import vectorize, uninitialized_params
from pysgmcmc.stepsize_schedules import (
    ConstantStepsizeSchedule,
    DualAveragingStepsizeSchedule,
)


__all__ = (
    "MCMCSampler",
    "BurnInMCMCSampler",
)


class MCMCSampler(object):

    """Generic base class for all MCMC samplers."""

    __metaclass__ = abc.ABCMeta

    def __init__(self, params, cost_fun, batch_generator=None,
                 stepsize_schedule=ConstantStepsizeSchedule(0.01),
                 session=tf.get_default_session(), dtype=tf.float64, seed=None):
        """
        Initialize the sampler base class. Sets up member variables and
        initializes uninitialized target parameters in the current
        `tensorflow.Graph`.

        Parameters
        ------------
        params : list of `tensorflow.Variable` objects
            Target parameters for which we want to sample new values.

        cost_fun : callable
            Function that takes `params` as input and returns a
            1-d `tensorflow.Tensor` that contains the cost-value.
            Frequently denoted with `U` in literature.

        batch_generator : `BatchGenerator`, optional
            Iterable which returns dictionaries to feed into
            tensorflow.Session.run() calls to evaluate the cost function.
            Defaults to `None` which indicates that no batches shall be fed.

        stepsize_schedule : pysgmcmc.stepsize_schedules.StepsizeSchedule
            Iterator class that produces a stream of stepsize values that
            we can use in our samplers.
            See also: `pysgmcmc.stepsize_schedules`

        session : `tensorflow.Session`, optional
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
        ------------
        pysgmcmc.sampling.BurnInMCMCSampler:
            Abstract base class for samplers that perform a burn-in phase
            to tune their own hyperparameters.
            Inherits from `sampling.MCMCSampler`.

        """
        # Sanitize inputs
        assert batch_generator is None or hasattr(batch_generator, "__next__")
        assert seed is None or isinstance(seed, int)

        assert isinstance(session, (tf.Session, tf.InteractiveSession))
        assert isinstance(dtype, tf.DType)

        assert callable(cost_fun)

        self.dtype = dtype

        self.n_iterations = 0

        self.seed = seed

        assert hasattr(stepsize_schedule, "update")
        assert hasattr(stepsize_schedule, "__next__")
        assert hasattr(stepsize_schedule, "stepsize")

        self.stepsize_schedule = stepsize_schedule

        self.batch_generator = batch_generator
        self.session = session

        self.params = params

        # set up costs
        self.cost_fun = cost_fun
        self.cost = cost_fun(self.params)

        # compute vectorized clones of all parameters
        self.vectorized_params = [vectorize(param) for param in self.params]

        self.epsilon = tf.Variable(
            self.stepsize_schedule.stepsize,
            dtype=self.dtype,
            name="epsilon",
            trainable=False
        )

        # Initialize uninitialized parameters before usage in any sampler.
        init = tf.variables_initializer(
            uninitialized_params(
                session=self.session,
                params=self.params + self.vectorized_params + [self.epsilon]
            )
        )
        self.session.run(init)

        # query this later to determine the next sample
        self.theta_t = [None] * len(params)

    def reset(self):
        """
        Reset all of this samplers parameters back to their initial value.
        Useful if we want to do multiple leapfrog steps from the same initial
        starting parameters (e.g. in `find_reasonable_epsilon` heuristic).

        Note: This does not allow full sampler chain reproducibility, even
        with a fixed random seed.  Tensorflow is currently not designed to
        allow resetting of its random streams,
        so `_draw_noise_sample` will return different random variables even
        after resetting sampler parameters.
        """
        # XXX: Make this sampler parameter aware instead of resetting *all*
        # global variables.
        self.session.run(tf.global_variables_initializer())

    def _next_batch(self):
        """
        Get a dictionary mapping `tensorflow.Placeholder` onto their corresponding feedable minibatch data.
        Each dictionary can directly be fed into `tensorflow.Session`.
        Returns an empty dictionary if `self.batch_generator` is `None`,
        i.e. if no batches are needed to compute the cost function.
        (e.g. the cost function depends only on the target parameters).

        Returns
        -------
        batch:
            Dictionary that maps `tensorflow.Placeholder` objects onto
            `ndarray` objects that can be fed for them.
            Returns an empty `dict` if `self.batch_generator` is `None`,
            i.e. if no batches are needed to compute the cost function
            (e.g. the cost function depends only on the target parameters).

        Examples
        ----------
        Extracting batches without any `batch_generator` function simply
        returns an empty `dict`:

        >>> import tensorflow as tf
        >>> import numpy as np
        >>> from itertools import islice
        >>> from pysgmcmc.samplers.sghmc import SGHMCSampler
        >>> session = tf.Session()
        >>> x = tf.Variable(1.0)
        >>> dist = tf.contrib.distributions.Normal(loc=0., scale=1.)
        >>> sampler = SGHMCSampler(
        ...     params=[x], cost_fun=lambda x: -dist.log_prob(x),
        ...     session=session, dtype=tf.float32
        ... )
        >>> session.close()
        >>> sampler._next_batch()
        {}

        A simple case with batches would look like this:

        >>> import tensorflow as tf
        >>> from pysgmcmc.models.bayesian_neural_network import generate_batches
        >>> from pysgmcmc.samplers.sghmc import SGHMCSampler
        >>> session = tf.Session()
        >>> N, D = 100, 3  # 100 datapoints with 3 features each
        >>> X = np.asarray([np.random.uniform(-10, 10, D) for _ in range(N)])
        >>> y = np.asarray([np.random.choice([0., 1.]) for _ in range(N)])
        >>> x_placeholder, y_placeholder = tf.placeholder(dtype=tf.float64), tf.placeholder(dtype=tf.float64)
        >>> batch_size = 10
        >>> batch_generator = generate_batches(
        ...     x=X, y=y,
        ...     x_placeholder=x_placeholder, y_placeholder=y_placeholder,
        ...     batch_size=batch_size
        ... )
        >>> sampler = SGHMCSampler(
        ...     params=[x], cost_fun=lambda x: x,  # cost function is just a dummy
        ...     session=session, dtype=tf.float32,
        ...     batch_generator=batch_generator
        ... )
        >>> batch_dict = sampler._next_batch()
        >>> session.close()
        >>> set(batch_dict.keys()) == set((x_placeholder, y_placeholder))
        True
        >>> batch_dict[x_placeholder].shape, batch_dict[y_placeholder].shape
        ((10, 3), (10, 1))

        """
        if self.batch_generator is not None:
            return next(self.batch_generator)
        return dict()

    # XXX: Doku
    def _next_stepsize(self):
        if self.n_iterations == 0 and self.stepsize_schedule.initialize_from_heuristic:
            self.stepsize_schedule.find_reasonable_epsilon(self)

        epsilon = next(self.stepsize_schedule)
        return {self.epsilon: epsilon}

    def _draw_noise_sample(self, sigma, shape):
        """
        Generate a single random normal sample with shape `shape` and standard deviation `sigma`.

        Parameters
        ----------
        sigma : tensorflow.Tensor
            Standard deviation of the noise.

        shape : tensorflow.Tensor
            Shape that the noise sample should have.

        Returns
        -------
        noise_sample: tensorflow.Tensor
            Random normal sample with shape `Shape` and
            standard deviation `Sigma`.

        """
        return sigma * tf.random_normal(
            shape=shape, dtype=self.dtype, seed=self.seed
        )

    # Conform to iterator protocol.
    # For reference see:
    # https://docs.python.org/3/library/stdtypes.html#iterator-types

    def __iter__(self):
        """
        Allows using samplers as iterators.

        Examples
        ----------
        Extract the first three thousand samples (with costs) from a sampler:

        >>> import tensorflow as tf
        >>> import numpy as np
        >>> from itertools import islice
        >>> from pysgmcmc.samplers.sghmc import SGHMCSampler
        >>> session = tf.Session()
        >>> x = tf.Variable(1.0)
        >>> dist = tf.contrib.distributions.Normal(loc=0., scale=1.)
        >>> n_burn_in, n_samples = 1000, 2000
        >>> sampler = SGHMCSampler(
        ...     params=[x], burn_in_steps=n_burn_in,
        ...     cost_fun=lambda x: -dist.log_prob(x),
        ...     session=session, dtype=tf.float32
        ... )
        >>> session.run(tf.global_variables_initializer())
        >>> burn_in_samples = list(islice(sampler, n_burn_in))  # perform all burn_in steps
        >>> samples = list(islice(sampler, n_samples))
        >>> len(burn_in_samples), len(samples)
        (1000, 2000)
        >>> session.close()
        >>> tf.reset_default_graph()  # to avoid polluting test environment

        """
        return self

    def __next__(self, feed_dict=None):
        """
        Compute and return the next sample and next cost values for this sampler.

        Returns
        --------
        sample: list of numpy.ndarray objects
            Sampled values are a `numpy.ndarray` for each target parameter.

        cost: numpy.ndarray (1,)
            Current cost value of the last evaluated target parameter values.

        Examples
        --------
        Extract the next sample (with costs) from a sampler:

        >>> import tensorflow as tf
        >>> import numpy as np
        >>> from itertools import islice
        >>> from pysgmcmc.samplers.sghmc import SGHMCSampler
        >>> session = tf.Session()
        >>> x = tf.Variable(1.0)
        >>> dist = tf.contrib.distributions.Normal(loc=0., scale=1.)
        >>> n_burn_in = 1000
        >>> sampler = SGHMCSampler(params=[x], burn_in_steps=n_burn_in, cost_fun=lambda x:-dist.log_prob(x), session=session, dtype=tf.float32)
        >>> session.run(tf.global_variables_initializer())
        >>> sample, cost = next(sampler)
        >>> session.close()
        >>> tf.reset_default_graph()  # to avoid polluting test environment

        Additional values to feed during computation can be given as `feed_dict`
        and will be forwarded to our `tensorflow.Session` object:

        >>> import tensorflow as tf
        >>> import numpy as np
        >>> from itertools import islice
        >>> from pysgmcmc.samplers.sghmc import SGHMCSampler
        >>> session = tf.Session()
        >>> x = tf.Variable(1.0)
        >>> dist = tf.contrib.distributions.Normal(loc=0., scale=1.)
        >>> n_burn_in = 1000
        >>> sampler = SGHMCSampler(params=[x], burn_in_steps=n_burn_in, cost_fun=lambda x:-dist.log_prob(x), session=session, dtype=tf.float32)
        >>> session.run(tf.global_variables_initializer())
        >>> sample, cost = sampler.__next__(feed_dict={sampler.epsilon: 1.0})  # fix stepsize to 1.0
        >>> session.close()
        >>> tf.reset_default_graph()  # to avoid polluting test environment

        """
        assert (feed_dict is None or hasattr(feed_dict, "update"))
        # Ensure self.theta_t and self.cost are defined
        assert hasattr(self, "theta_t") and hasattr(self, "cost")

        params, cost, _ = self.leapfrog(feed_dict=feed_dict)

        self.stepsize_schedule.update(params, cost)

        self.n_iterations += 1
        return params, cost

    def leapfrog(self, feed_dict=None):
        assert (feed_dict is None or hasattr(feed_dict, "update"))
        # Ensure self.theta_t and self.cost are defined
        assert hasattr(self, "theta_t") and hasattr(self, "cost")

        if feed_dict is None:
            feed_dict = dict()

        feed_dict.update(self._next_batch())

        if self.epsilon not in feed_dict:
            feed_dict.update(self._next_stepsize())

        params, cost, momentum, _ = self.session.run(
            [self.params, self.cost, self.momentum, self.theta_t], feed_dict=feed_dict
        )

        if len(params) == 1:
            # unravel single-element lists to scalars
            params = params[0]

        if len(momentum) == 1:
            momentum = momentum[0]

        return params, cost, momentum


class BurnInMCMCSampler(MCMCSampler):
    """ Base class for MCMC samplers that use a burn-in procedure to
        estimate their mass matrix.
        Details of how this burn-in is performed are left to be
        specified in the individual samplers that inherit from this class.
    """

    __metaclass__ = abc.ABCMeta

    def __init__(self, params, cost_fun, batch_generator=None,
                 stepsize_schedule=ConstantStepsizeSchedule(0.01),
                 burn_in_steps=3000,
                 session=tf.get_default_session(), dtype=tf.float64, seed=None):
        """
        Initializes the corresponding MCMCSampler super object and
        sets member variables.

        Parameters
        ----------
        params : list of `tensorflow.Variable` objects
            Target parameters for which we want to sample new values.

        cost_fun : callable
            Function that takes `params` as input and returns a
            1-d `tensorflow.Tensor` that contains the cost-value.
            Frequently denoted with `U` in literature.

        batch_generator : `BatchGenerator`, optional
            Iterable which returns dictionaries to feed into
            tensorflow.Session.run() calls to evaluate the cost function.
            Defaults to `None` which indicates that no batches shall be fed.

        stepsize_schedule : pysgmcmc.stepsize_schedules.StepsizeSchedule
            Iterator class that produces a stream of stepsize values that
            we can use in our samplers.
            See also: `pysgmcmc.stepsize_schedules`

        burn_in_steps : int
            Number of burn-in steps to perform. In each burn-in step, this
            sampler will adapt its own internal parameters to decrease its error.
            Defaults to `3000`.

        session : `tensorflow.Session`, optional
            Session object which knows about the external part of the graph
            (which defines `Cost`, and possibly batches).
            Used internally to evaluate (burn-in/sample) the sampler.

        dtype : tensorflow.DType, optional
            Type of elements of `tensorflow.Tensor` objects used in this sampler.
            Defaults to `tensorflow.float64`.

        seed : int, optional
            Random seed to use.
            Defaults to `None`.

        See also
        ----------
        pysgmcmc.sampling.MCMCSampler:
            Super class of this class. Has generic methods shared by all
            MCMC samplers implemented as part of this framework.

        pysgmcmc.samplers.sghmc.SGHMCSampler:
            Instantiation of this class.
            Uses SGHMC to sample from the target distribution after burn-in.

        pysgmcmc.samplers.sgld.SGLDSampler:
            Instantiation of this class.
            Uses SGLD to sample from the target distribution after burn-in.

        """
        # Sanitize inputs
        assert isinstance(burn_in_steps, int)

        super().__init__(params=params, cost_fun=cost_fun,
                         stepsize_schedule=stepsize_schedule,
                         batch_generator=batch_generator,
                         seed=seed, dtype=dtype, session=session)

        self.burn_in_steps = burn_in_steps

    @property
    def is_burning_in(self) -> bool:
        """ Check if this sampler is still in burn-in phase.
            Used during graph construction to insert conditionals into the
            graph that will make the sampler skip all burn-in operations
            after the burn-in phase is over.

        Returns
        -------
        is_burning_in: boolean
            `True` if `self.n_iterations <= self.burn_in_steps`, otherwise `False`.
        """

        return self.n_iterations < self.burn_in_steps

    def __next__(self, feed_dict=None):
        """ Perform a sampler step:
            Compute and return the next sample and next cost values
            for this sampler.

            While `self.is_burning_in` returns `True`
            (while the sampler has not yet performed `self.burn_in_steps`
            steps) this will also adapt the samplers mass matrix in a
            sampler-specific way to improve performance.

        Returns
        -------
        sample: list of numpy.ndarray objects
            Sampled values are a `numpy.ndarray` for each target parameter.

        cost: numpy.ndarray (1,)
            Current cost value of the last evaluated target parameter values.

        """
        assert (feed_dict is None or hasattr(feed_dict, "update"))

        if feed_dict is None:
            feed_dict = dict()

        if self.is_burning_in:
            # feed next batch and stepsize
            feed_dict.update(self._next_batch())
            feed_dict.update(self._next_stepsize())

            # perform a burn-in step = adapt the samplers mass matrix inverse
            params, cost, self.minv = self.session.run(
                [self.theta_t, self.cost, self.minv_t],
                feed_dict=feed_dict
            )

            self.stepsize_schedule.update(params, cost)

            self.n_iterations += 1
            return params, cost

        # "standard" MCMC sampling
        if self.burn_in_steps > 0:
            assert hasattr(self, "minv_t")
            assert hasattr(self, "minv")

            # feed tuned inverse of mass matrix (minv) during sampling
            feed_dict = dict(zip(self.minv_t, self.minv))

        return super().__next__(feed_dict=feed_dict)

from keras import backend as K


def sampler_from_optimizer(optimizer_cls):
    class Sampler(optimizer_cls):
        def __init__(self, loss, params, inputs=None, **optimizer_args):
            optimizer_args["parameter_shapes"] = [
                param.shape for param in params
            ]
            super().__init__(**optimizer_args)
            self.loss = loss
            self.params = params

            self.updates = self.get_updates(self.loss, self.params)
            inputs = inputs if inputs is not None else []

            self.function = K.function(
                inputs,
                [self.loss] + self.params,
                updates=self.updates,
                name="sampler_function"
            )

        def step(self, inputs=None):
            if inputs is None:
                inputs = []
            loss, *params = self.function(inputs)
            return loss, params

        def __next__(self):
            return self.step()

        def __iter__(self):
            return self

    Sampler.__name__ = optimizer_cls.__name__
    return Sampler
