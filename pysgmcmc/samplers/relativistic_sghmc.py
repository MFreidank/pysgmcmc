# vim:foldmethod=marker
import tensorflow as tf
from pysgmcmc.samplers.base_classes import MCMCSampler
from pysgmcmc.stepsize_schedules import ConstantStepsizeSchedule

from pysgmcmc.tensor_utils import (
    vectorize, unvectorize
)

from arspy.ars import adaptive_rejection_sampling


class RelativisticSGHMCSampler(MCMCSampler):
    """ Relativistic Stochastic Gradient Hamiltonian Monte-Carlo Sampler.

        See [1] for more details on Relativistic SGHMC.

        [1] X. Lu, V. Perrone, L. Hasenclever, Y. W. Teh, S. J. Vollmer
            In Proceedings of the 20 th International Conference on Artificial Intelligence and Statistics (AISTATS) 2017\n
            `Relativistic Monte Carlo <http://proceedings.mlr.press/v54/lu17b/lu17b.pdf>`_

    """

    def __init__(self, params, cost_fun, batch_generator=None,
                 stepsize_schedule=ConstantStepsizeSchedule(0.001),
                 mass=1.0, speed_of_light=1.0, D=1.0, Bhat=0.0,
                 session=tf.get_default_session(), dtype=tf.float64, seed=None):
        """ Initialize the sampler parameters and set up a tensorflow.Graph
            for later queries.

        Parameters
        ----------
        params : list of tensorflow.Variable objects
            Target parameters for which we want to sample new values.

        Cost : tensorflow.Tensor
            1-d Cost tensor that depends on `params`.
            Frequently denoted as U(theta) in literature.

        batch_generator : BatchGenerator, optional
            Iterable which returns dictionaries to feed into
            tensorflow.Session.run() calls to evaluate the cost function.
            Defaults to `None` which indicates that no batches shall be fed.

        stepsize_schedule : pysgmcmc.stepsize_schedules.StepsizeSchedule
            Iterator class that produces a stream of stepsize values that
            we can use in our samplers.
            See also: `pysgmcmc.stepsize_schedules`

        mass : float, optional
            mass constant.
            Defaults to `1.0`.

        speed_of_light : float, optional
            "Speed of light" constant. TODO EXTEND DOKU
            Defaults to `1.0`.

        D : float, optional
            Diffusion constant.
            Defaults to `1.0`.

        Bhat : float, optional
            TODO: Documentation

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
        pysgmcmc.sampling.MCMCSampler:
            Base class for `RelativisticSGHMCSampler` that specifies how
            actual sampling is performed (using iterator protocol,
            e.g. `next(sampler)`).

        """

        # Set up MCMCSampler base class:
        # initialize member variables common to all samplers
        # and run initializers for all uninitialized variables in `params`
        # (to avoid errors in the graph definitions below).
        super().__init__(
            params=params, cost_fun=cost_fun, batch_generator=batch_generator,
            stepsize_schedule=stepsize_schedule,
            seed=seed, dtype=dtype, session=session
        )

        # Use `-self.Cost` since the rest of the implementation expects
        # a log likelihood (instead of the *negative* log likelihood that
        # we normally use as costs)
        grads = [vectorize(gradient) for gradient in tf.gradients(-self.Cost, params)]

        D = tf.constant(D, dtype=dtype)
        Bhat = tf.constant(Bhat, dtype=dtype)

        momentum = [
            tf.Variable(momentum_sample, dtype=dtype)
            for momentum_sample in _sample_relativistic_momentum(
                m=mass, c=speed_of_light, n_params=len(self.params)
            )
        ]

        # In internal implementation, stick to mathematical formulas.
        # For users, prefer readability.
        m = tf.constant(mass, dtype=dtype)
        c = tf.constant(speed_of_light, dtype=dtype)

        for i, (Param, Grad) in enumerate(zip(params, grads)):
            Vectorized_Param = self.vectorized_params[i]

            p_grad = self.Epsilon * momentum[i] / (m * tf.sqrt(momentum[i] * momentum[i] / (tf.square(m) * tf.square(c)) + 1))

            n = tf.sqrt(self.Epsilon * (2 * D - self.Epsilon * Bhat)) * tf.random_normal(shape=Vectorized_Param.shape, dtype=dtype)
            Momentum_t = tf.assign_add(
                momentum[i],
                tf.reshape(self.Epsilon * Grad + n - D * p_grad, momentum[i].shape)
            )

            p_grad_new = self.Epsilon * Momentum_t / (m * tf.sqrt(Momentum_t * Momentum_t / (tf.square(m) * tf.square(c)) + 1))
            Vectorized_Theta_t = tf.assign_add(
                Vectorized_Param,
                tf.reshape(p_grad_new, Vectorized_Param.shape)
            )

            self.Theta_t[i] = tf.assign(
                Param,
                unvectorize(Vectorized_Theta_t, original_shape=Param.shape)
            )


def _sample_relativistic_momentum(m, c, n_params,
                                  bounds=(float("-inf"), float("inf"))):
    """
    Use adaptive rejection sampling (here: provided by external library `ARSpy`)
    to sample initial values for relativistic momentum `p`.
    The relativistic momentum variable in Relativistic MCMC has (marginal)
    distribution
    .. math:: \\propto e^{-K(p)}
    where :math:`K(p)` is the relativistic kinetic energy.
    This distribution is a multivariate generalisation of the symmetric
    hyperbolic distribution, which cannot easily be sampled directly.
    Therefore we resort to *adaptive rejection sampling* to generate our samples
    and initialize our momentum terms properly.

    See `the paper "Relativistic Monte Carlo" <http://proceedings.mlr.press/v54/lu17b/lu17b.pdf/#page=2>`_ for more information on Relativistic Hamiltonian Dynamics.

    See `Generalized hyperbolic distribution <https://en.wikipedia.org/wiki/Generalised_hyperbolic_distribution>`_ for more information on our target distribution.

    Parameters
    ----------
    m : float
        Mass constant used for sampling.

    c : float
        Speed of light constant used for sampling.

    n_params : int
        Number of target parameters of the target log pdf to sample from.

    bounds : Tuple[float, float], optional
        Adaptive rejection sampling bounds to use during sampling.
        Defaults to `(float("-inf"), float("inf"))`, i.e. unbounded
        adaptive rejection sampling.

    Returns
    ----------
    momentum_samples : list
        Samples used to initialize our samplers momentum variables.

    Examples
    ----------

    Drawing 10 momentum values for 10 target parameters via (unbounded)
    adaptive rejection sampling:

    >>> n_params = 10
    >>> momentum_values = _sample_relativistic_momentum(m=1.0, c=1.0, n_params=n_params)
    >>> len(momentum_values) == n_params
    True

    See also
    ----------
    `ARSpy`: Our external dependency that handles adaptive rejection sampling.
             Available `here <https://github.com/MFreidank/pyars>`_.

    """
    # XXX: Remove when more is supported, currently only floats for mass
    # and c are.
    assert isinstance(m, float)
    assert isinstance(c, float)

    def generate_relativistic_logpdf(m, c):
        def relativistic_log_pdf(p):
            """
            Logarithm of pdf of (multivariate) generalized
            hyperbolic distribution.
            """
            from numpy import sqrt
            return -m * c ** 2 * sqrt(p ** 2 / (m ** 2 * c ** 2) + 1.)
        return relativistic_log_pdf

    momentum_log_pdf = generate_relativistic_logpdf(m=m, c=c)
    return adaptive_rejection_sampling(
        logpdf=momentum_log_pdf, a=-10.0, b=10.0,
        domain=bounds, n_samples=n_params
    )
