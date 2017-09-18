# vim:foldmethod=marker
import tensorflow as tf
from pysgmcmc.sampling import MCMCSampler

from pysgmcmc.tensor_utils import (
    vectorize, unvectorize
)


class RelativisticSGHMCSampler(MCMCSampler):
    """ Relativistic Stochastic Gradient Hamiltonian Monte-Carlo Sampler.

        See [1] for more details on Relativistic SGHMC.

        [1] X. Lu, V. Perrone, L. Hasenclever, Y. W. Teh, S. J. Vollmer
            In Proceedings of the 20 th International Conference on Artifi-
cial Intelligence and Statistics (AISTATS) 2017\n

            `Relativistic Monte Carlo <proceedings.mlr.press/v54/lu17b/lu17b.pdf>`_
    """
    def __init__(self, params, cost_fun, momentum=[0.0], batch_generator=None,
                 epsilon=0.001, mass=1.0, c=1.0, D=1.0, Bhat=0.0,
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

        momentum : float or List[float], optional
            Initial values for the momentum of the sampler.
            Defaults to `0.0`.

        batch_generator : BatchGenerator, optional
            Iterable which returns dictionaries to feed into
            tensorflow.Session.run() calls to evaluate the cost function.
            Defaults to `None` which indicates that no batches shall be fed.

        epsilon : float, optional
            Value that is used as learning rate parameter for the sampler,
            also denoted as discretization parameter in literature.
            Defaults to `0.001`.

        mass : float, optional
            mass constant.
            Defaults to `1.0`.

        c : float, optional
            "Speed of light" constant.
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
            seed=seed, dtype=dtype, session=session
        )

        grads = [vectorize(gradient) for gradient in tf.gradients(self.Cost, params)]

        stepsize = tf.constant(epsilon)
        m = tf.constant(mass)
        c = tf.constant(c)
        D = tf.constant(D)
        Bhat = tf.constant(Bhat)
        momentum = [tf.Variable(momentum_val) for momentum_val in momentum]

        for i, (Param, Grad) in enumerate(zip(params, grads)):
            Vectorized_Param = self.vectorized_params[i]

            p_grad = stepsize * momentum[i] / (m * tf.sqrt(momentum[i] * momentum[i] / (tf.square(m) * tf.square(c)) + 1))

            n = tf.sqrt(stepsize * (2 * D - stepsize * Bhat)) * tf.random_normal(shape=Vectorized_Param.shape)
            Momentum_t = tf.assign_add(
                momentum[i],
                tf.reshape(stepsize * Grad + n - D * p_grad, momentum[i].shape)
            )

            p_grad_new = stepsize * Momentum_t / (m * tf.sqrt(Momentum_t * Momentum_t / (tf.square(m) * tf.square(c)) + 1))
            Vectorized_Theta_t = tf.assign_add(
                Vectorized_Param,
                tf.reshape(p_grad_new, Vectorized_Param.shape)
            )

            self.Theta_t[i] = tf.assign(
                Param,
                unvectorize(Vectorized_Theta_t, original_shape=Param.shape)
            )
