from abc import ABCMeta, abstractmethod
import numpy as np
import tensorflow as tf


class StepsizeSchedule(object):
    """ Generic base class for all stepsize schedules. """
    __metaclass__ = ABCMeta

    def __init__(self, initial_value, initialize_from_heuristic=False):
        assert isinstance(initialize_from_heuristic, bool)

        self.stepsize = initial_value
        self.initialize_from_heuristic = initialize_from_heuristic

    @abstractmethod
    def __next__(self):
        """
        Compute and return the next stepsize according to this schedule.

        Returns
        ----------
        next_stepsize : float
            Next stepsize to use according to this schedule.
        """

    def __iter__(self):
        return self

    @abstractmethod
    def update(self, *args, **kwargs):
        """
        Update this schedule with new information. What information
        will be relevant depends on the type of schedule used.
        Information may e.g. include cost values for the last step size
        used, effective sample sizes of a sampler, values of other
        hyperparameters etc.

        """

    def find_reasonable_epsilon(self, sampler):
        """ Initialize using a heuristic introduced in the original NUTS paper.

            In a nutshell, this heuristic repeatedly doubles or halves the stepsize
            until the acceptance probability of a Langevin proposal for `sampler`
            with this stepsize crosses `0.5`.

            See [1] for more details on this heuristic procedure.\n

            [1] M. D. Hoffman, A. Gelman
                In Journal of Machine Learning Research 15 (2014).\n

                `The No-U-Turn Sampler: Adaptively Setting Path Lengths in Hamiltonian Monte Carlo. <http://www.stat.columbia.edu/~gelman/research/published/nuts.pdf>`_


        Parameters
        ----------
        sampler : pysgmcmc.samplers.base_classes.MCMCSampler
            Sampler instance to perform leapfrog steps for.
            These Leapfrog steps are used to compute our Langevin proposal.

        Returns
        ----------
        reasonable_epsilon : float
            First stepsize for which langevin proposal of `sampler` crossed `0.5`.


        Examples
        ----------

        Computing a reasonable initial stepsize for one of our simple
        gaussian mixtures:

        >>> import tensorflow as tf
        >>> import numpy as np
        >>> from itertools import islice
        >>> from pysgmcmc.samplers.sghmc import SGHMCSampler
        >>> from pysgmcmc.stepsize_schedules import ConstantStepsizeSchedule
        >>> from pysgmcmc.diagnostics.objective_functions import (
        ...     gmm3_log_likelihood as ll, to_negative_log_likelihood)
        >>> session = tf.Session()
        >>> x = tf.Variable(1.0)
        >>> n_burn_in, n_samples = 1000, 2000
        >>> sampler = SGHMCSampler(
        ... params=[x], burn_in_steps=n_burn_in,
        ... cost_fun=to_negative_log_likelihood(ll), seed=1,
        ... session=session, dtype=tf.float32)
        >>> session.run(tf.global_variables_initializer())
        >>> schedule = ConstantStepsizeSchedule(initial_value=1.0)
        >>> epsilon = schedule.find_reasonable_epsilon(sampler)
        >>> session.close()
        >>> epsilon
        0.5

        Computing a reasonable initial stepsize for our 2d-banana function works
        in exactly the same way:

        >>> import tensorflow as tf
        >>> import numpy as np
        >>> from itertools import islice
        >>> from pysgmcmc.samplers.sghmc import SGHMCSampler
        >>> from pysgmcmc.stepsize_schedules import ConstantStepsizeSchedule
        >>> from pysgmcmc.diagnostics.objective_functions import (
        ...     banana_log_likelihood as ll, to_negative_log_likelihood)
        >>> session = tf.Session()
        >>> x = tf.Variable(1.0)
        >>> y = tf.Variable(1.0)
        >>> n_burn_in, n_samples = 1000, 2000
        >>> sampler = SGHMCSampler(
        ... params=[x, y], burn_in_steps=n_burn_in,
        ... cost_fun=to_negative_log_likelihood(ll), seed=1,
        ... session=session, dtype=tf.float32)
        >>> session.run(tf.global_variables_initializer())
        >>> schedule = ConstantStepsizeSchedule(initial_value=1.0)
        >>> epsilon = schedule.find_reasonable_epsilon(sampler)
        >>> session.close()
        >>> epsilon
        0.25

        """
        epsilon = 1.

        theta, costs, r = sampler.session.run(
            [sampler.params, sampler.cost, sampler.momentum]
        )

        theta_, costs_, r_ = sampler.leapfrog(
            feed_dict={sampler.epsilon: epsilon}
        )

        def p(r, costs):
            log_likelihood = -np.squeeze(costs)
            r = np.squeeze(r)
            return np.exp(log_likelihood - 0.5 * np.dot(np.transpose(r), r))

        # Compute old and new likelihood: p(theta, r)
        p_0 = p(r=r, costs=costs)
        p_ = p(r=r_, costs=costs_)

        a = 2. * ((p_ / p_0) > 0.5) - 1.

        while ((p_ / p_0) ** a) > 2 ** -a:
            epsilon *= (2 ** a)

            # Reset all sampler parameters to their initial values
            sampler.session.run(tf.global_variables_initializer())

            theta_, costs_, r_ = sampler.leapfrog(
                feed_dict={sampler.epsilon: epsilon}
            )

            p_ = p(costs=costs_, r=r_)

        self.stepsize = epsilon

        return epsilon


class ConstantStepsizeSchedule(StepsizeSchedule):
    """ Trivial schedule that keeps the stepsize at a constant value.  """

    def __next__(self):
        """
        Calling `next(schedule)` on a constant stepsize schedule
        will always return the schedules initial value.

        Returns
        ----------
        constant_value : float
            Constant value associated with this schedule.

        Examples
        ----------
        Proof of concept:

        >>> schedule = ConstantStepsizeSchedule(0.01)
        >>> schedule.stepsize
        0.01
        >>> next(schedule)
        0.01
        >>> from itertools import islice
        >>> list(islice(schedule, 4))
        [0.01, 0.01, 0.01, 0.01]

        """
        return self.stepsize

    def __str__(self):
        """ Pretty string representation of `ConstantStepsizeSchedule`.

        Returns
        ----------
        schedule_str : string
            String representation of this schedule.

        Examples
        ----------
        Proof of concept:

        >>> schedule = ConstantStepsizeSchedule(0.01)
        >>> str(schedule)
        'ConstantStepsizeSchedule(stepsize=0.01)'

        >>> schedule = ConstantStepsizeSchedule(0.1)
        >>> str(schedule)
        'ConstantStepsizeSchedule(stepsize=0.1)'

        """
        return "ConstantStepsizeSchedule(stepsize={})".format(self.stepsize)

    def update(self, *args, **kwargs):
        """ Updating a constant stepsize schedule is a no-op. """
        pass


def DualAveragingStepsizeSchedule(StepsizeSchedule):
    """ Stepsize schedule based on dual averaging."""

    def __init__(self, n_intermediate_leapfrogs=1,
                 desired_acceptance_rate=0.651,
                 n_adaptation_iterations=float("inf"),
                 gamma=0.05,
                 t_0=10., initial_value=1., initialize_from_heuristic=True):
        # n_adaptation iterations is called m_adapt in nuts paper
        # n_intermediate_leapfrogs is called L_m in nuts paper
        # desired_acceptance_rate is called delta in nuts paper
        super().__init__(
            initial_value=initial_value,
            initialize_from_heuristic=initialize_from_heuristic
        )

        self.stepsize_bar = 1.

        self.n_iterations = 0
        self.n_adaptation_iterations = n_adaptation_iterations
        self.n_intermediate_leapfrogs = n_intermediate_leapfrogs

        self.last_r, self.last_costs = None, None

        self.statistics = 0.
        self.t_0 = t_0
        self.desired_acceptance_rate = desired_acceptance_rate

        self.gamma = gamma
        # XXX: How to set up mu properly? We run into trouble since
        # it should be set based on results of "find_reasonable_epsilon"

    @property
    def should_adapt_stepsize(self):
        if self.n_iterations == 0:
            return False
        return self.n_iterations % self.n_intermediate_leapfrogs == 0

    def update(self, momentum, costs, *args, **kwargs):
        if self.n_iterations > self.n_adaptation_iterations:
            self.stepsize = self.stepsize_bar
            return

        # XXX: Move somewhere appropriate,
        # then call from here and up in `find_reasonable_epsilon`
        def p(r, costs):
            log_likelihood = -np.squeeze(costs)
            r_squeezed = np.squeeze(r)
            return np.exp(
                log_likelihood - 0.5 * np.dot(np.transpose(r_squeezed), r_squeezed)
            )

        if self.should_adapt_stepsize:
            # acceptance rate is called alpha in nuts paper
            acceptance_rate = np.min(
                1.,
                p(r=momentum, costs=costs) / p(momentum=self.last_r, costs=self.last_costs)
            )

            acceptance_difference = self.desired_acceptance_rate - acceptance_rate
            moving_average_coefficient = (1. / (self.n_iterations + self._t0))

            # statistics is called H_m in nuts paper
            self.statistics = (
                (1 - moving_average_coefficient) * self.statistics +
                moving_average_coefficient * acceptance_difference
            )

            self.stepsize = np.exp(
                self.mu - (np.sqrt(self.n_iterations) / self.gamma) * self.statistics
            )

            moving_average_coefficient = self.n_iterations ** -self.kappa

            self.stepsize_bar = np.exp(
                moving_average_coefficient * np.log(self.stepsize) +
                (1. - moving_average_coefficient) * np.log(self.stepsize_bar)
            )

            self.last_momentum = momentum
            self.last_costs = costs
