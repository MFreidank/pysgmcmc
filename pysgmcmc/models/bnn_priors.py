""" Priors on weights and log predicted variance used in BNNs.  """
import tensorflow as tf
from pysgmcmc.tensor_utils import safe_divide


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
