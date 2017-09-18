# vim: foldmethod=marker
"""
Abstract base classes for all MCMC methods. Helps unify our sampler interface.
"""
import abc
import tensorflow as tf
from enum import Enum

from pysgmcmc.tensor_utils import vectorize, uninitialized_params

__all__ = (
    "MCMCSampler",
    "BurnInMCMCSampler",
    "Sampler"
)


class MCMCSampler(object):
    """ Generic base class for all MCMC samplers.  """
    __metaclass__ = abc.ABCMeta

    def __init__(self, params, cost_fun, batch_generator=None,
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

        assert(isinstance(session, (tf.Session, tf.InteractiveSession)))
        assert(isinstance(dtype, tf.DType))

        assert(hasattr(cost_fun, "__call__"))

        self.dtype = dtype

        self.n_iterations = 0

        self.seed = seed

        self.batch_generator = batch_generator
        self.session = session

        self.params = params

        # set up costs
        self.cost_fun = cost_fun
        self.Cost = cost_fun(self.params)

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
        >>> batch_generator = generate_batches(x=X, y=y, x_placeholder=x_placeholder, y_placeholder=y_placeholder, batch_size=batch_size)
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

    def __next__(self, feed_vals=None):
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
        if feed_vals is None:
            feed_vals = dict()

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

    def __init__(self, params, cost_fun, batch_generator=None,
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
        assert(isinstance(burn_in_steps, int))

        super().__init__(params=params, cost_fun=cost_fun,
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

    def __next__(self, feed_vals=None):
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
        if feed_vals is None:
            feed_vals = dict()

        feed_vals.update(self._next_batch())

        if self.is_burning_in:
            # perform a burn-in step = adapt the samplers mass matrix inverse
            params, cost, self.minv = self.session.run(
                [self.Theta_t, self.Cost, self.Minv_t],
                feed_dict=feed_vals
            )
            self.n_iterations += 1
            return params, cost

        # "standard" MCMC sampling
        return super().__next__(feed_vals=dict(zip(self.Minv_t, self.minv)))


class Sampler(Enum):
    """ Enumeration type for all samplers we support. """

    SGHMC = "SGHMC"
    RelativisticSGHMC = "RelativisticSGHMC"
    SGLD = "SGLD"

    @staticmethod
    def is_supported(sampling_method):
        """
        Static method that returns true if `sampling_method` is a
        supported sampler (e.g. there is an entry for it in `Sampler` enum).

        Examples
        ----------

        Supported sampling methods give `True`:

        >>> Sampler.is_supported(Sampler.SGHMC)
        True

        Other input types give `False`:

        >>> Sampler.is_supported(0)
        False
        >>> Sampler.is_supported("test")
        False

        """
        return sampling_method in Sampler

    @classmethod
    def get_sampler(cls, sampling_method, **sampler_args):
        """ Return a sampler object for supported `sampling_method`, where all
            default values for parameters in keyword dictionary `sampler_args`
            are overwritten.

        Parameters
        ----------
        sampling_method : Sampler
            Enum corresponding to sampling method to return a sampler for.

        **sampler_args : dict
            Keyword arguments that contain all input arguments to the desired
            the constructor of the sampler for the specified `sampling_method`.

        Returns
        ----------
        sampler : Subclass of `sampling.MCMCSampler`
            A sampler instance that implements the specified `sampling_method`
            and is initialized with inputs `sampler_args`.

        Examples
        ----------
        We can use this method to construct a sampler for a given
        sampling method and override default values by providing them as
        keyword arguments:

        >>> import tensorflow as tf
        >>> params = [tf.Variable(0.)]
        >>> cost_fun = lambda params: tf.reduce_sum(params) # dummy cost function
        >>> session=tf.Session()
        >>> with tf.Session() as session: sampler = Sampler.get_sampler(Sampler.SGHMC, session=session, params=params, cost_fun=cost_fun, dtype=tf.float32)
        >>> type(sampler)
        <class 'pysgmcmc.samplers.sghmc.SGHMCSampler'>
        >>> sampler.dtype
        tf.float32
        >>> session.close()

        Construction of SGLD sampler:

        >>> import tensorflow as tf
        >>> params = [tf.Variable(0.)]
        >>> cost_fun = lambda params: tf.reduce_sum(params) # dummy cost function
        >>> session=tf.Session()
        >>> with tf.Session() as session: sampler = Sampler.get_sampler(Sampler.SGLD, session=session, params=params, cost_fun=cost_fun, dtype=tf.float32)
        >>> type(sampler)
        <class 'pysgmcmc.samplers.sgld.SGLDSampler'>
        >>> sampler.dtype
        tf.float32
        >>> session.close()

        Construction of Relativistic SGHMC sampler:

        >>> import tensorflow as tf
        >>> params = [tf.Variable(0.)]
        >>> cost_fun = lambda params: tf.reduce_sum(params) # dummy cost function
        >>> session=tf.Session()
        >>> with tf.Session() as session: sampler = Sampler.get_sampler(Sampler.RelativisticSGHMC, session=session, params=params, cost_fun=cost_fun, dtype=tf.float32)
        >>> type(sampler)
        <class 'pysgmcmc.samplers.relativistic_sghmc.RelativisticSGHMCSampler'>
        >>> sampler.dtype
        tf.float32
        >>> session.close()

        Sampler arguments that do not have a default *must* be provided as keyword
        argument, otherwise this method will raise an exception:

        >>> sampler = Sampler.get_sampler(Sampler.SGHMC, dtype=tf.float32)
        Traceback (most recent call last):
          ...
        ValueError: sampling.Sampler.get_sampler: params was not overwritten as sampler argument in `sampler_args` and does not have any default value in SGHMCSampler.__init__Please pass an explicit value for this parameter.

        If an **optional** argument is not provided as keyword argument,
        the corresponding default value is used.
        If we do not provide/overwrite the `dtype` keyword argument,
        the samplers default value of `tf.float64` is used:

        >>> import tensorflow as tf
        >>> params = [tf.Variable(0., dtype=tf.float64)]
        >>> cost_fun = lambda params: tf.reduce_sum(params) # dummy cost function
        >>> with tf.Session() as session: sampler = Sampler.get_sampler(Sampler.SGHMC, session=session, params=params, cost_fun=cost_fun)
        >>> sampler.dtype
        tf.float64

        If a keyword argument that is provided does not represent a valid
        parameter of the corresponding `sampling_method`, a `ValueError` is
        raised:

        >>> import tensorflow as tf
        >>> params = [tf.Variable(0., dtype=tf.float64)]
        >>> cost_fun = lambda params: tf.reduce_sum(params) # dummy cost function
        >>> with tf.Session() as session: sampler = Sampler.get_sampler(Sampler.SGHMC, unknown_argument=None, session=session, params=params, cost_fun=cost_fun)
        Traceback (most recent call last):
          ...
        ValueError: sampling.Sampler.get_sampler: 'SGHMCSampler' does not take any parameter with name 'unknown_argument' which was specified as argument to this sampler. Please ensure, that you only specify sampler arguments that fit the corresponding sampling method.
        For your choice of sampling method ('Sampler.SGHMC'), supported parameters are:
        -params
        -cost_fun
        -batch_generator
        -epsilon
        -burn_in_steps
        -mdecay
        -scale_grad
        -session
        -dtype
        -seed

        """

        assert(cls.is_supported(sampling_method))

        if sampling_method == cls.SGHMC:
            from pysgmcmc.samplers.sghmc import SGHMCSampler as Sampler
        elif sampling_method == cls.SGLD:
            from pysgmcmc.samplers.sgld import SGLDSampler as Sampler
        elif sampling_method == cls.RelativisticSGHMC:
            from pysgmcmc.samplers.relativistic_sghmc import RelativisticSGHMCSampler as Sampler
        else:
            assert(False)

        from inspect import signature, _empty

        # look up all initializer parameters with their (potential)
        # default values
        all_sampler_parameters = signature(Sampler.__init__).parameters

        try:
            undefined_parameter = next(
                parameter_name for parameter_name in sampler_args
                if parameter_name not in all_sampler_parameters
            )
        except StopIteration:
            pass
        else:
            raise ValueError(
                "sampling.Sampler.get_sampler: '{sampler_name}' "
                "does not take any parameter with name '{parameter}' "
                "which was specified as argument to this sampler. "
                "Please ensure, that you only specify sampler arguments "
                "that fit the corresponding sampling method.\n"
                "For your choice of sampling method ('{sampler}'), supported parameters are:\n"
                "{valid_parameters}".format(
                    sampler_name=Sampler.__name__,
                    sampler=sampling_method,
                    parameter=undefined_parameter,
                    valid_parameters="\n".join(
                        ["-{}".format(parameter_name)
                         for parameter_name in all_sampler_parameters
                         if parameter_name != "self"]
                    )
                )
            )

        def parameter_value(parameter_name):
            """ Determine the value to assign to the parameter
                with name `parameter_name`.
                If `parameter_name` is overwritten (if it is a key in
                `sampler_args`) use the value provided in `sampler_args`.
                Otherwise, fall back to the default value provided in
                the samplers `init` method.

            Parameters
            ----------
            parameter_name : string
                Name of the parameter that we want to determine the value for.

            Returns
            -------
            value : object
                Value of sampler parameter with name `parameter_name` that
                will be passed to the initializer of the sampler.

            """

            default_value = all_sampler_parameters[parameter_name].default

            if parameter_name not in sampler_args and default_value is _empty:
                raise ValueError(
                    "sampling.Sampler.get_sampler: "
                    "{param_name} was not overwritten as sampler argument "
                    "in `sampler_args` and does not have any default value "
                    "in {sampler}.__init__"
                    "Please pass an explicit value for this parameter.".format(
                        param_name=parameter_name, sampler=Sampler.__name__
                    )
                )

            return sampler_args.get(parameter_name, default_value)

        sampler_args = {
            parameter_name: parameter_value(parameter_name)
            for parameter_name in all_sampler_parameters
            if parameter_name != "self"  # never pass `self` during construction
        }

        return Sampler(**sampler_args)
