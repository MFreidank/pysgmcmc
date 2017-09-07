# vim: foldmethod=marker
"""
Abstract base classes for all MCMC methods. Helps unify our sampler interface.
"""
import abc
import tensorflow as tf

from pysgmcmc.tensor_utils import vectorize, uninitialized_params


class MCMCSampler(object):
    """ Generic base class for all MCMC samplers.  """
    __metaclass__ = abc.ABCMeta

    def __init__(self, params, seed=None, batch_generator=None,
                 dtype=tf.float64, session=tf.get_default_session()):
        """
        Initialize the sampler base class. Sets up member variables and
        initializes uninitialized target parameters in the current
        `tensorflow.Graph`.

        Parameters
        ------------
        params : list of `tensorflow.Variable` objects
            Target parameters for which we want to sample new values.

        seed : int, optional
            Random seed to use.
            Defaults to `None`.

        batch_generator : `BatchGenerator`, optional
            Iterable which returns dictionaries to feed into
            tensorflow.Session.run() calls to evaluate the cost function.
            Defaults to `None` which indicates that no batches shall be fed.

        dtype : tensorflow.DType, optional
            Type of elements of `tensorflow.Tensor` objects used in this sampler.
            Defaults to `tensorflow.float64`.

        session : `tensorflow.Session`, optional
            Session object which knows about the external part of the graph
            (which defines `Cost`, and possibly batches).
            Used internally to evaluate (burn-in/sample) the sampler.

        See Also
        ------------
        pysgmcmc.sampling.BurnInMCMCSampler:
            Abstract base class for samplers that perform a burn-in phase
            to tune their own hyperparameters.
            Inherits from `sampling.MCMCSampler`.

        """

        # Sanitize inputs
        assert(batch_generator is None or hasattr(batch_generator, "__next__"))
        assert(seed is None or type(seed) == int)

        assert(isinstance(session, tf.Session))
        assert(isinstance(dtype, tf.DType))

        self.dtype = dtype

        self.n_iterations = 0

        self.seed = seed

        self.batch_generator = batch_generator
        self.session = session

        self.params = params

        # compute vectorized clones of all parameters
        self.vectorized_params = [vectorize(param) for param in self.params]

        # Initialize uninitialized parameters before usage in any sampler.
        init = tf.variables_initializer(
            uninitialized_params(
                session=self.session, params=self.params + self.vectorized_params
            )
        )
        self.session.run(init)

        self.Theta_t = [None] * len(params)  # query this later to get next sample

    def _next_batch(self):
        """ Get a dictionary mapping `tensorflow.Placeholder` onto
            their corresponding feedable minibatch data.
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
        >>> sampler = SGHMCSampler(params=[x], cost_fun=lambda x: -dist.log_prob(x), session=session, dtype=tf.float32)
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
        >>> batch_generator = generate_batches(X=X, y=y, x_placeholder=x_placeholder, y_placeholder=y_placeholder, batch_size=batch_size)
        >>> sampler = SGHMCSampler(params=[x], cost_fun=lambda x: x, session=session, dtype=tf.float32, batch_generator=batch_generator)  # cost function is just a dummy
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

    def _draw_noise_sample(self, Sigma, Shape):
        """ Generate a single random normal sample with shape `Shape` and
            standard deviation `Sigma`.

        Parameters
        ----------
        Sigma : tensorflow.Tensor
            Standard deviation of the noise.

        Shape : tensorflow.Tensor
            Shape that the noise sample should have.

        Returns
        -------
        noise_sample: tensorflow.Tensor
            Random normal sample with shape `Shape` and
            standard deviation `Sigma`.

        """
        return Sigma * tf.random_normal(
            shape=Shape, dtype=self.dtype, seed=self.seed
        )

    # Conform to iterator protocol.
    # For reference see:
    # https://docs.python.org/3/library/stdtypes.html#iterator-types

    def __iter__(self):
        """ Allows using samplers as iterators.

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
        >>> sampler = SGHMCSampler(params=[x], burn_in_steps=n_burn_in, cost_fun=lambda x: -dist.log_prob(x), session=session, dtype=tf.float32)
        >>> session.run(tf.global_variables_initializer())
        >>> burn_in_samples = list(islice(sampler, n_burn_in))  # perform all burn_in steps
        >>> samples = list(islice(sampler, n_samples))
        >>> len(burn_in_samples), len(samples)
        (1000, 2000)
        >>> session.close()
        >>> tf.reset_default_graph()  # to avoid polluting test environment

        """
        return self

    def __next__(self, feed_vals=dict()):
        """ Compute and return the next sample and
            next cost values for this sampler.

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

        """
        if not hasattr(self, "Theta_t") or not hasattr(self, "Cost"):
            # Ensure self.Theta_t and self.Cost are defined
            raise ValueError(
                "MCMCSampler subclass attempted to compute the next sample "
                "with corresponding costs, but one of the "
                "two necessary sampler member variables 'Theta_t' and 'Cost' "
                "were not found in the samplers instance dictionary."
            )

        feed_vals.update(self._next_batch())
        params, cost = self.session.run(
            [self.Theta_t, self.Cost], feed_dict=feed_vals
        )

        if len(params) == 1:
            # unravel single-element lists to scalars
            params = params[0]

        self.n_iterations += 1  # increment iteration counter

        return params, cost


class BurnInMCMCSampler(MCMCSampler):
    """ Base class for MCMC samplers that use a burn-in procedure to
        estimate their mass matrix.
        Details of how this burn-in is performed are left to be
        specified in the individual samplers that inherit from this class.
    """

    __metaclass__ = abc.ABCMeta

    def __init__(self, params, burn_in_steps, seed=None,
                 batch_generator=None, dtype=tf.float64,
                 session=tf.get_default_session()):
        """
        Initializes the corresponding MCMCSampler super object and
        sets member variables.

        Parameters
        ----------
        params : list of `tensorflow.Variable` objects
            Target parameters for which we want to sample new values.

        burn_in_steps: int
            Number of burn-in steps to perform. In each burn-in step, this
            sampler will adapt its own internal parameters to decrease its error.
            For reference see: TODO ADD PAPER REFERENCE HERE

        seed : int, optional
            Random seed to use.
            Defaults to `None`.

        batch_generator : `BatchGenerator`, optional
            Iterable which returns dictionaries to feed into
            tensorflow.Session.run() calls to evaluate the cost function.
            Defaults to `None` which indicates that no batches shall be fed.

        dtype : tensorflow.DType, optional
            Type of elements of `tensorflow.Tensor` objects used in this sampler.
            Defaults to `tensorflow.float64`.

        session : `tensorflow.Session`, optional
            Session object which knows about the external part of the graph
            (which defines `Cost`, and possibly batches).
            Used internally to evaluate (burn-in/sample) the sampler.

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
        assert(type(burn_in_steps) == int)

        super().__init__(params=params, batch_generator=batch_generator,
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

    def __next__(self):
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
        if self.is_burning_in:
            # perform a burn-in step = adapt the samplers mass matrix inverse
            params, cost, self.minv = self.session.run(
                [self.Theta_t, self.Cost, self.Minv_t],
                feed_dict=self._next_batch()
            )
            self.n_iterations += 1
            return params, cost
        else:
            # "standard" MCMC sampling
            return super().__next__(feed_vals=dict(zip(self.Minv_t, self.minv)))
