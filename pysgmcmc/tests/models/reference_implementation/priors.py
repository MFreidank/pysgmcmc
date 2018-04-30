import numpy as np
import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams


class ReferenceLogVariancePrior(object):
    def __init__(self, mean, var=2):
        """
        Prior on the log predicted variance
        :param mean: Actual mean on a linear scale: Default 10E-3
        :param var: Variance on a log scale: Default 2
        """

        self.mean = mean
        self.var = var

    def prepare_for_train(self, n_examples):
        self.n_examples = theano.shared(np.float32(n_examples))

    def update_for_train(self, n_examples):
        self.n_examples.set_value(np.float32(n_examples))

    def log_like(self, log_var):
        return T.mean(T.sum(
            -T.square(log_var - T.log(self.mean)) / (2 * self.var) - 0.5 * T.log(
                self.var), axis=1))  # / self.n_examples


class ReferenceWeightPrior(object):
    def __init__(self, rng=None, alpha=1, beta=10000.):
        if rng:
            self._srng = rng
        else:
            self._srng = RandomStreams(np.random.randint(1, 2147462579))

        self.alpha_prior = alpha
        self.beta_prior = beta
        self.wdecay = theano.shared(np.float32(1.))

    def get_decay(self):
        return self.wdecay

    def prepare_for_train(self, params, n_data):
        self.n_data = n_data
        return self.wdecay

    def update_for_train(self, n_data):
        self.n_data = n_data
        self.wdecay.set_value(1.)

    def log_like(self, params):
        ll = 0.
        n_params = 0
        # NOTE: we are dropping all constants here
        for p in params:
            ll += T.sum(-self.wdecay * 0.5 * T.square(p))
            n_params += T.prod(p.shape)
        return ll / n_params

    def update(self, params):
        W_sum = 0
        W_size = 0
        for p in params:
            W = p.get_value()
            W_sum += np.sum(np.square(W))
            W_size += np.prod(W.shape)
        alpha = self.alpha_prior + 0.5 * W_size
        beta = self.beta_prior + 0.5 * W_sum
        p_wd = np.random.gamma(alpha, 1. / (beta + 1e-4))
        # wd is the next weight decay
        wd = p_wd
        # the scaling with n_data above is now done in
        # the log likeliehood (where it should be!)
        self.wdecay.set_value(np.float32(wd))
