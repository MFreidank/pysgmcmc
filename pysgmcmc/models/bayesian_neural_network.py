# vim: foldmethod=marker


#  Imports {{{ #
from collections import deque
from enum import Enum
from itertools import islice
import logging
from time import time
import numpy as np
import tensorflow as tf

from pysgmcmc.models.base_model import (
    BaseModel,
    zero_mean_unit_var_normalization,
    zero_mean_unit_var_unnormalization
)


from pysgmcmc.samplers.sghmc import SGHMCSampler
from pysgmcmc.samplers.sgld import SGLDSampler

from pysgmcmc.data_batches import generate_batches
from pysgmcmc.tensor_utils import safe_divide


# XXX: This needs a decision: do we do without this entirely? If not,
# where does it live?
# from pysgmcmc.sampling import SamplingMethod
class SamplingMethod(Enum):
    """ Enumeration type for all sampling methods we support. """

    SGHMC = "SGHMC"
    SGLD = "SGLD"

    # will automatically generate tests for these when running "make test"
    autotested = (
        SGHMC,
        SGLD,
    )

    @staticmethod
    def is_supported(sampling_method):
        """
        Static method that returns true if `val` is a supported sampling
        method (e.g. there is an entry for it in `SamplingMethod` enum).

        Examples
        ----------

        Supported sampling methods give `True`:

        >>> SamplingMethod.is_supported(SamplingMethod.SGHMC)
        True

        Other input types give `False`:

        >>> SamplingMethod.is_supported(0)
        False
        >>> SamplingMethod.is_supported("test")
        False

        """
        return sampling_method in SamplingMethod

    @staticmethod
    def get_sampler(sampling_method, **sampler_args):
        """TODO: Docstring for get_sampler.

        Parameters
        ----------
        sampling_method : SamplingMethod
            Enum corresponding to sampling method to return a sampler for.

        **sampler_args : dict
            Keyword arguments that contain all input arguments to the desired
            the constructor of the sampler for the specified `sampling_method`.

        Returns
        -------
        sampler : Subclass of `sampling.MCMCSampler`
            A sampler instance that implements the specified `sampling_method`
            and is initialized with inputs `sampler_args`.

        """
        if sampling_method == SamplingMethod.SGHMC:
            sampler = SGHMCSampler(
                batch_generator=sampler_args["batch_generator"],
                seed=sampler_args["seed"],
                cost_fun=sampler_args["cost_fun"],
                params=sampler_args["params"],
                epsilon=sampler_args["epsilon"],
                mdecay=sampler_args["mdecay"],
                scale_grad=sampler_args["scale_grad"],
                session=sampler_args["session"],
                burn_in_steps=sampler_args["burn_in_steps"]
            )
        elif sampling_method == SamplingMethod.SGLD:
            sampler = SGLDSampler(
                batch_generator=sampler_args["batch_generator"],
                seed=sampler_args["seed"],
                cost_fun=sampler_args["cost_fun"],
                params=sampler_args["params"],
                epsilon=sampler_args["epsilon"],
                scale_grad=sampler_args["scale_grad"],
                session=sampler_args["session"],
                burn_in_steps=sampler_args["burn_in_steps"]
            )
        elif sampling_method == SamplingMethod.RelativisticSGHMC:
            raise NotImplementedError()
        else:
            raise ValueError()

        return sampler


def get_default_net(inputs, seed=None):
    from tensorflow.contrib.layers import variance_scaling_initializer as HeNormal
    fc_layer_1 = tf.layers.dense(
        inputs, units=50, activation=tf.tanh,
        kernel_initializer=HeNormal(factor=1.0, dtype=tf.float64, seed=seed),
        bias_initializer=tf.zeros_initializer(dtype=tf.float64),
        name="fc_layer_1"
    )

    fc_layer_2 = tf.layers.dense(
        fc_layer_1, units=50, activation=tf.tanh,
        kernel_initializer=HeNormal(factor=1.0, dtype=tf.float64, seed=seed),
        bias_initializer=tf.zeros_initializer(dtype=tf.float64),
        name="fc_layer_2"
    )

    fc_layer_3 = tf.layers.dense(
        fc_layer_2, units=50, activation=tf.tanh,
        kernel_initializer=HeNormal(factor=1.0, dtype=tf.float64, seed=seed),
        bias_initializer=tf.zeros_initializer(dtype=tf.float64),
        name="fc_layer_3"
    )

    layer_4 = tf.layers.dense(
        fc_layer_3, units=1, activation=None,  # linear activation
        kernel_initializer=HeNormal(factor=1.0, dtype=tf.float64, seed=seed),
        bias_initializer=tf.zeros_initializer(dtype=tf.float64),
        name="fc_layer_4"
    )

    output_bias = tf.Variable(
        [[np.log(1e-3)]], dtype=tf.float64,
        name="output_bias"
    )

    output = tf.concat(
        [layer_4, tf.ones_like(layer_4, dtype=tf.float64) * output_bias],
        axis=1,
        name="Network_Output"
    )

    return output


#  Priors {{{ #

class LogVariancePrior(object):
    """ Prior on the log predicted variance."""

    def __init__(self, mean=10e-3, var=2):
        """ Initialize prior for a given `mean` and `variance`.

        Parameters
        ----------
        mean : float, optional
        Actual mean on a linear scale. Default value is `10e-3`.

        var : float, optional
        Variance on a log scale. Default value is `2`.

        """

        self.mean = tf.constant(mean, name="log_variance_prior_mean", dtype=tf.float64)
        self.var = tf.constant(var, name="log_variance_prior_var", dtype=tf.float64)

    def log_like(self, log_var):
        """ Compute the log likelihood of this prior for a given input.

        Parameters
        ----------
        log_var: tensorflow.Tensor

        Returns
        -------
        log_like_output: tensorflow.Tensor

        """

        return tf.reduce_mean(tf.reduce_sum(
            safe_divide(-tf.square(log_var - tf.log(self.mean)), (2. * self.var)) - 0.5 * tf.log(
                self.var), axis=1), name="variance_prior_log_like")


class WeightPrior(object):
    """ Prior on the weights."""
    def __init__(self):
        """ Initialize weight prior with weight decay initialized to `1.` """
        self.Wdecay = tf.constant(1., name="wdecay", dtype=tf.float64)

    def log_like(self, params):
        """ Compute the log log likelihood of this prior for a given input.

        Parameters
        ----------
        params : list of tensorflow.Variable objects

        Returns
        -------
        log_like: tensorflow.Tensor

        """
        ll = tf.convert_to_tensor(0., name="ll", dtype=tf.float64)
        n_params = tf.convert_to_tensor(0., name="n_params", dtype=tf.float64)

        for p in params:
            ll += tf.reduce_sum(-self.Wdecay * 0.5 * tf.square(p))
            n_params += tf.cast(tf.reduce_prod(tf.to_float(p.shape)), dtype=tf.float64)
        return safe_divide(ll, n_params, name="weight_prior_log_like")

#  }}} Priors #


#  }}}  Imports #


class BayesianNeuralNetwork(object):
    def __init__(self, sampling_method=SamplingMethod.SGHMC,
                 n_nets=100, learning_rate=1e-3, mdecay=5e-2,
                 n_iters=50000, batch_size=20, burn_in_steps=1000,
                 sample_steps=100, normalize_input=True, normalize_output=True,
                 get_net=get_default_net, batch_generator=generate_batches,
                 seed=None, session=None):
        """
        Bayesian Neural Networks use Bayesian methods to estimate the posterior
        distribution of a neural network's weights. This allows to also
        predict uncertainties for test points and thus makes Bayesian Neural
        Networks suitable for Bayesian optimization.

        This module uses stochastic gradient MCMC methods to sample
        from the posterior distribution.

        See [1] for more details.

        [1] J. T. Springenberg, A. Klein, S. Falkner, F. Hutter
            Bayesian Optimization with Robust Bayesian Neural Networks.
            In Advances in Neural Information Processing Systems 29 (2016).

        Parameters
        ----------
        sampling_method : SamplingMethod, optional
            Method used to sample networks for this BNN.
            Defaults to `SamplingMethod.SGHMC`.

        n_nets: int, optional
            Number of nets to sample during training (and use to predict).
            Defaults to `100`.

        learning_rate: float, optional
            Learning rate to use during sampling.
            Defaults to `1e-3`.

        mdecay: float, optional
            Momentum decay per time-step (parameter for SGHMCSampler).
            Defaults to `0.05`.

        n_iters: int, optional
            Total number of iterations of the sampler to perform.
            Defaults to `50000`

        batch_size: int, optional
            Number of datapoints to include in each minibatch.
            Defaults to `20` datapoints per minibatch.

        burn_in_steps: int, optional
            Number of burn-in steps to perform
            Defaults to `1000`.

        sample_steps: int, optional
            Number of sample steps to perform.
            Defaults to `100`.

        normalize_input: bool, optional
            Specifies whether or not input data should be normalized.
            Defaults to `True`

        normalize_output: bool, optional
            Specifies whether or not outputs should be normalized.
            Defaults to `True`

        get_net: callable, optional
            Callable that returns a network specification.
            Expected inputs are a `tensorflow.Placeholder` object that
            serves as feedable input to the network and an integer random seed.
            Expected return value is the networks final output.
            Defaults to `get_default_net`.

        batch_generator: callable, optional
            TODO: DOKU
            NOTE: Generator callable with signature like generate_batches that
            yields feedable dicts of minibatches.

        seed: int, optional
            Random seed to use in this BNN.
            Defaults to `None`.

        session: tensorflow.Session, optional
            A `tensorflow.Session` object used to delegate computations
            performed in this network over to `tensorflow`.
            Defaults to `None` which indicates we should start a fresh
            `tensorflow.Session`.

        """

        # XXX: Raise readable errors for all sanitizations when they fail
        # Sanitize inputs
        assert(isinstance(n_nets, int))
        assert(isinstance(n_iters, int))
        assert(isinstance(burn_in_steps, int))
        assert(isinstance(sample_steps, int))
        assert(isinstance(batch_size, int))

        assert(n_nets > 0)
        assert(n_iters > 0)
        assert(burn_in_steps >= 0)
        assert(sample_steps > 0)
        assert(batch_size > 0)

        assert(hasattr(get_net, "__call__"))
        assert(hasattr(batch_generator, "__call__"))

        if not SamplingMethod.is_supported(sampling_method):
            raise ValueError(
                "'BayesianNeuralNetwork.__init__' received unsupported input "
                "for parameter 'sampling_method'. Input was: {input}.\n"
                "Supported sampling methods are enumerated in "
                "'SamplingMethod' enum type.".format(input=sampling_method)
            )

        self.sampling_method = sampling_method

        self.get_net = get_net
        self.batch_generator = batch_generator

        self.normalize_input = normalize_input
        self.normalize_output = normalize_output

        self.n_nets = n_nets
        self.n_iters = n_iters

        self.batch_size = batch_size

        self.learning_rate = learning_rate
        self.mdecay = mdecay

        self.burn_in_steps = burn_in_steps
        self.sample_steps = sample_steps

        self.samples = deque(maxlen=n_nets)

        self.seed = seed

        self.session = session

        if not self.session:
            self.session = tf.Session()

        self.variance_prior = LogVariancePrior(mean=1e-6, var=0.01)
        self.weight_prior = WeightPrior()

    def negative_log_likelihood(self, X, Y):
        """ Compute the negative log likelihood of the
            current network parameters with respect to inputs `X` with
            labels `Y`.

        Parameters
        ----------
        X : tensorflow.Placeholder
            Placeholder for input datapoints.

        Y : tensorflow.Placeholder
            Placeholder for input labels.

        Returns
        -------
        neg_log_like: tensorflow.Tensor
            Negative log likelihood of the current network parameters with
            respect to inputs `X` with labels `Y`.


        mse: tensorflow.Tensor
            Mean squared error of the current network parameters
            with respect to inputs `X` with labels `Y`.

        """

        self.net_output = self.get_net(inputs=X, seed=self.seed)

        f_mean = tf.reshape(self.net_output[:, 0], shape=(-1, 1))
        f_log_var = tf.reshape(self.net_output[:, 1], shape=(-1, 1))

        f_var_inv = 1. / (tf.exp(f_log_var) + 1e-16)

        mse = tf.square(Y - f_mean)

        log_like = tf.reduce_sum(
            tf.reduce_sum(-mse * (0.5 * f_var_inv) - 0.5 * f_log_var, axis=1)
        )

        # scale by batch size to make this work nicely with the updaters above
        log_like = log_like / tf.constant(self.batch_size, dtype=tf.float64)

        # scale the priors by the dataset size for the same reason
        n_examples = tf.constant(self.X.shape[0], tf.float64, name="n_examples")

        # prior for the variance
        log_like += self.variance_prior.log_like(f_log_var) / n_examples

        # prior for the weights
        log_like += (self.weight_prior.log_like(tf.trainable_variables()) /
                     n_examples)

        return -log_like, tf.reduce_mean(mse)

    @BaseModel._check_shapes_predict
    def train(self, X, y, *args, **kwargs):
        """ Train our Bayesian Neural Network using input datapoints `X`
            with corresponding labels `y`.

        Parameters
        ----------
        X : numpy.ndarray (N, D)
            Input training datapoints.

        y : numpy.ndarray (N,)
            Input training labels.
        """
        # XXX: Some changes are necessary to allow multiple successive calls
        # to train: Proposal = clear the whole preexisting graph?
        # Advantages over only setting up once is that we can use the same object
        # on different function successively which is a lot more flexible
        # XXX: We might also want to move session construction here then

        start_time = time()
        self.is_trained = False

        self.X, self.y = X, y

        if self.normalize_input:
            self.X, self.x_mean, self.x_std = zero_mean_unit_var_normalization(self.X)

        if self.normalize_output:
            self.y, self.y_mean, self.y_std = zero_mean_unit_var_normalization(self.y)

        n_datapoints, n_inputs = X.shape

        # set up placeholders for data minibatches
        self.X_Minibatch = tf.placeholder(shape=(None, n_inputs),
                                          dtype=tf.float64,
                                          name="X_Minibatch")
        self.Y_Minibatch = tf.placeholder(dtype=tf.float64, name="Y_Minibatch")

        # set up tensors for negative log likelihood and mean squared error
        Nll, Mse = self.negative_log_likelihood(
            X=self.X_Minibatch, Y=self.Y_Minibatch
        )

        self.network_params = tf.trainable_variables()

        # remove any leftover samples from previous "train" calls
        self.samples.clear()

        self.sampler = SamplingMethod.get_sampler(
            self.sampling_method,
            batch_generator=self.batch_generator(
                x=self.X, x_placeholder=self.X_Minibatch,
                y=self.y, y_placeholder=self.Y_Minibatch,
                batch_size=self.batch_size,
                seed=self.seed
            ),
            seed=self.seed,
            cost_fun=lambda *_: Nll,  # BNN costs do not need params as input
            params=self.network_params,
            epsilon=self.learning_rate,
            mdecay=self.mdecay,
            scale_grad=n_datapoints,
            session=self.session,
            burn_in_steps=self.burn_in_steps
        )

        self.session.run(tf.global_variables_initializer())

        logging.info("Starting sampling")

        def log_full_training_error(iteration_index, is_sampling: bool):
            """ Compute the error of our last sampled network parameters
                on the full training dataset and use `logging.info` to
                log it. The boolean flag `sampling` is used to determine
                whether we are already collecting sampled networks, in which
                case additional info is logged using `logging.info`.

            Parameters
            ----------
            is_sampling : bool
                Boolean flag that specifies if we are already
                collecting samples or if we are still doing burn-in steps.
                If set to `True` we will also log the total number
                of samples collected thus far.

            """
            total_nll, total_mse = self.session.run(
                [Nll, Mse], feed_dict={
                    self.X_Minibatch: self.X,
                    self.Y_Minibatch: self.y.reshape(-1, 1)
                }
            )
            t = time() - start_time
            if is_sampling:
                logging.info("Iter {:8d} : NLL = {:.4e} MSE = {:.4e} "
                             "Time = {:5.2f}".format(iteration_index,
                                                     float(total_nll),
                                                     float(total_mse),
                                                     t))
            else:
                logging.info("Iter {:8d} : NLL = {:.4e} MSE = {:.4e} "
                             "Samples = {} Time = {:5.2f}".format(
                                 iteration_index, float(total_nll),
                                 float(total_mse), len(self.samples), t))

        logging_intervals = {"burn-in": 512, "sampling": self.sample_steps}

        sample_chain = islice(self.sampler, self.n_iters)

        for i, (parameter_values, _) in enumerate(sample_chain):
            burning_in = i <= self.burn_in_steps

            if burning_in and i % logging_intervals["burn-in"] == 0:
                log_full_training_error(iteration_index=i, is_sampling=False)

            if not burning_in and i % logging_intervals["sampling"] == 0:
                log_full_training_error(iteration_index=i, is_sampling=True)

                # collect sample
                self.samples.append(parameter_values)

                if len(self.samples) == self.n_nets:
                    # collected enough sample networks, stop iterating
                    break

        """
        for i in range(self.n_iters):
            # boolean that tracks whether our sampler is doing burn-in steps
            burning_in = i <= self.burn_in_steps

            param_values, _ = next(self.sampler)

            if burning_in and i % logging_intervals["burn-in"] == 0:
                log_full_training_error(iteration_index=i, is_sampling=False)

            if not burning_in and i % logging_intervals["sampling"] == 0:
                log_full_training_error(iteration_index=i, is_sampling=True)

                # collect sample
                self.samples.append(param_values)

                if len(self.samples) == self.n_nets:
                    # collected enough sample networks, stop iterating
                    break
        """

        self.is_trained = True

    def compute_network_output(self, params, input_data):
        """ Compute and return the output of the network when
            parameterized with `params` on `input_data`.

        Parameters
        ----------
        params : list of ndarray objects
            List of parameter values (ndarray)
            for each tensorflow.Variable parameter of our network.

        input_data : ndarray (N, D)
            Input points to compute the network output for.

        Returns
        -------
        network_output: ndarray (N,)
            Output of the network parameterized with `params`
            on the given `input_data` points.
        """

        feed_dict = dict(zip(self.network_params, params))
        feed_dict[self.X_Minibatch] = input_data
        return self.session.run(self.net_output, feed_dict=feed_dict)

    @BaseModel._check_shapes_predict
    def predict(self, X_test, return_individual_predictions=False, *args, **kwargs):
        """
        Returns the predictive mean and variance of the objective function at
        the given test points.

        Parameters
        ----------
        X_test: np.ndarray (N, D)
            Input test datapoints.

        return_individual_predictions: bool
            If set to `True` than the individual predictions of
            all samples are returned.

        Returns
        ----------
        mean: np.array(N,)
            predictive mean

        variance: np.array(N,)
            predictive variance

        """

        if not self.is_trained:
            logging.error("Model is not trained!")
            return

        # Normalize input
        if self.normalize_input:
            X_, _, _ = zero_mean_unit_var_normalization(
                X_test, self.x_mean, self.x_std
            )
        else:
            X_ = X_test

        f_out = []
        theta_noise = []

        for sample in self.samples:
            out = self.compute_network_output(params=sample, input_data=X_)

            f_out.append(out[:, 0])
            theta_noise.append(np.exp(out[:, 1]))

        f_out = np.asarray(f_out)
        theta_noise = np.asarray(theta_noise)

        if return_individual_predictions:
            if self.normalize_output:
                f_out = zero_mean_unit_var_unnormalization(
                    f_out, self.y_mean, self.y_std
                )
                theta_noise *= self.y_std**2
            return f_out, theta_noise

        m = np.mean(f_out, axis=0)
        # Total variance
        # v = np.mean(f_out ** 2 + theta_noise, axis=0) - m ** 2
        v = np.mean((f_out - m) ** 2, axis=0)

        if self.normalize_output:
            m = zero_mean_unit_var_unnormalization(m, self.y_mean, self.y_std)
            v *= self.y_std ** 2

        return m, v
