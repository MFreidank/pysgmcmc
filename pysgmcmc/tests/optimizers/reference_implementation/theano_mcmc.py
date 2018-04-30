import numpy as np
import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from .utils import sharedX



class SGHMCSampler(object):

    def __init__(self, rng=None, precondition=False, ignore_burn_in=False):
        if rng:
            self._srng = rng
        else:
            self._srng = RandomStreams(np.random.randint(1, 2147462579))
        self.precondition = precondition
        self.prepared = False
        self.ignore_burn_in = ignore_burn_in
        self.steps_burn_in = 0
        self.requires_burn_in = self.precondition
        self.optim_params = []
        self.initial_values = []

    def _store_initial_values(self, *params):
        self.optim_params = []
        self.initial_values = []
        for param in params:
            self.optim_params.append(param)
            self.initial_values.append(param.get_value())

    def prepare_updates(self, cost, params, epsilon, mdecay=0.05, inputs=[], scale_grad=1.,
                        A=None, **kwargs):
        self.updates = []
        self.burn_in_updates = []
        grads = T.grad(cost, params)
        self.params = params
        self.cost = cost
        self.count = sharedX(0)
        self.epsilon = sharedX(np.float32(epsilon))
        self.mdecay = sharedX(np.float32(mdecay ))
        self.inputs = inputs
        self.scale_grad = theano.shared(np.float32(scale_grad))
        if A is not None:
            # calculate mdecay based on A
            #raise NotImplementedError("TODO")
            eps_scaled = epsilon / np.sqrt(scale_grad)
            new_mdecay = A * eps_scaled
            self.mdecay.set_value(np.float32(new_mdecay))
            print("You specified A of {} -> changing mdecay to {}".format(A, mdecay))

        for theta,grad in zip(params, grads):
            xi = sharedX(theta.get_value() * 0. + 1, broadcastable=theta.broadcastable)
            g = sharedX(theta.get_value() * 0. + 1, broadcastable=theta.broadcastable)
            g2 = sharedX(theta.get_value() * 0. + 1, broadcastable=theta.broadcastable)
            p = sharedX(theta.get_value() * 0., broadcastable=theta.broadcastable)
            r_t = 1. / (xi + 1.)
            self._store_initial_values(xi, g, g2, p)
            if self.precondition:
                g_t = (1. - r_t) * g + r_t * grad
                g2_t = (1. - r_t) * g2 + r_t * grad**2
                xi_t = 1. + xi * (1. - g * g / (g2 + 1e-16))
                Minv = 1. / (T.sqrt(g2 + 1e-16) + 1e-16)
                self.burn_in_updates.append((g, g_t))
                self.burn_in_updates.append((g2, g2_t))
                self.burn_in_updates.append((xi, xi_t))
                noise = 0.
            else:
                Minv = 1.
                noise = 0.
            self.epsilon_scaled = self.epsilon / T.sqrt(self.scale_grad)
            noise_scale = 2. * self.epsilon_scaled ** 2 * self.mdecay * Minv - 2. * self.epsilon_scaled ** 3 * T.square(Minv) * noise
            sigma = T.sqrt(T.maximum(noise_scale, 1e-16))
            sample_t = self._srng.normal(size=theta.shape) * sigma
            p_t = p - self.epsilon**2 * Minv * grad - self.mdecay * p + sample_t
            theta_t = theta + p_t
            self.updates.append((theta, theta_t))
            self.updates.append((p, p_t))
        self.prepared = True
        if self.ignore_burn_in:
            self.updates += self.burn_in_updates
            return self.updates
        else:
            return self.updates, self.burn_in_updates

    def step(self, *inp):
        if not self.prepared:
            raise RuntimeError("You called step() without a prior call to prepare_updates()")
        if not hasattr(self, "step_fun"):
            print("... compiling theano function")

            self.step_fun = theano.function(self.inputs, self.cost, updates=self.updates)
        if not self.ignore_burn_in and self.steps_burn_in < 1 and self.requires_burn_in:
            raise RuntimeError("Your sampler requires a burn_in please run step_burn_in() for a few steps")
        nll = self.step_fun(*inp)
        return self.params, nll

    def step_burn_in(self, *inp):
        if not self.prepared:
            raise RuntimeError("You called step_burn_in() without a prior call to prepare_updates()")
        if not hasattr(self, "step_fun_burn_in"):
            print("... compiling theano function")
            if self.ignore_burn_in:
                self.step_fun_burn_in = theano.function(self.inputs, self.cost, updates=self.updates)
            else:
                self.step_fun_burn_in = theano.function(self.inputs, self.cost, updates=self.updates + self.burn_in_updates)

        nll = self.step_fun_burn_in(*inp)
        self.steps_burn_in += 1
        return self.params, nll

    def reset(self, n_samples, epsilon, reset_opt_params=False, **kwargs):
        if self.prepared:
            self.epsilon.set_value(np.float32(epsilon))
            self.scale_grad.set_value(np.float32(n_samples))
            if hasattr(self, "mdecay"):
                if "mdecay" in kwargs:
                    self.mdecay.set_value(np.float32(kwargs["mdecay"]))
                elif "A" in kwargs:
                    eps_scaled = self.epsilon.get_value() / np.sqrt(n_samples)
                    new_mdecay = A * eps_scaled
                    self.mdecay.set_value(np.float32(new_mdecay))
            if reset_opt_params:
                for param,value in zip(self.optim_params, self.initial_values):
                    param.set_value(value)
        else:
            raise RuntimeError("reset called before prepare")

class SGLDSampler(SGHMCSampler):

    def __init__(self, rng=None, precondition=False):
        super(SGLDSampler, self).__init__(rng=rng, precondition=precondition)

    def prepare_updates(self, cost, params, epsilon, A=1., inputs=[], scale_grad=1., **kwargs):
        self.updates = []
        self.burn_in_updates = []
        grads = T.grad(cost, params)
        self.params = params
        self.cost = cost
        self.count = sharedX(0)
        self.epsilon = sharedX(np.float32(epsilon))
        self.A = T.cast(A, theano.config.floatX)
        self.inputs = inputs
        for theta,grad in zip(params, grads):
            xi = sharedX(theta.get_value() * 0. + 1, broadcastable=theta.broadcastable)
            g = sharedX(theta.get_value() * 0. + 1, broadcastable=theta.broadcastable)
            g2 = sharedX(theta.get_value() * 0. + 1, broadcastable=theta.broadcastable)
            r_t = 1. / (xi + 1.)
            self._store_initial_values(xi, g, g2)
            if self.precondition:
                g_t = (1. - r_t) * g + r_t * grad
                g2_t = (1. - r_t) * g2 + r_t * grad**2
                xi_t = 1 + xi * (1 - g * g / (g2 + 1e-16))
                Minv = 1. / (T.sqrt(g2 + 1e-16) + 1e-16)
                self.burn_in_updates.append((g, g_t))
                self.burn_in_updates.append((g2, g2_t))
                self.burn_in_updates.append((xi, xi_t))
                noise = 0.
            else:
                Minv = 1.
                noise = 0.
            sigma = T.sqrt(2. * self.epsilon * (Minv * (self.A - noise)) / T.cast(scale_grad, dtype=theano.config.floatX))
            sample_t = self._srng.normal(size=theta.shape) * sigma
            theta_t = theta - self.epsilon * Minv * self.A * grad + sample_t
            self.updates.append((theta, theta_t))
        self.prepared = True
        if self.ignore_burn_in:
            return self.updates + self.burn_in_updates
        else:
            return self.updates, self.burn_in_updates
