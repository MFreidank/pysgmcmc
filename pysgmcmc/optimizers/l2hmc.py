import typing
import tensorflow as tf
import numpy as np
from keras import backend as K
from keras.optimizers import Optimizer, Adam
from pysgmcmc.keras_utils import (
    INTEGER_DTYPE, FLOAT_DTYPE,
    keras_control_dependencies,
    n_dimensions, to_vector, updates_for, while_loop
)
from pysgmcmc.l2hmc_layers import (
    Sequential, Zip, Linear, Parallel, ScaleTanh
)
from pysgmcmc.custom_typing import KerasTensor, KerasVariable
# NOTE: Open questions:
# XXX: How can we reformat l2HMC algorithm to fit into an optimizer based scheme
# e.g. can we split updates up such that we can work on a given `loss` tensor without
# needing access to our energy function inside the optimizer?


class Dynamics(object):
    def __init__(self, x_dim, energy_function, T=25, eps=0.1,
                 net_factory=None, eps_trainable=True, use_temperature=False):

        self.x_dim = x_dim
        self.use_temperature = use_temperature

        alpha = K.variable(K.log(K.constant(eps), name="alpha"))

        self.eps = K.exp(alpha)
        # XXX: We should try to avoid needing access to the energy function inside.
        self._fn = energy_function
        self.T = T

        self._init_mask()

        # TODO: Turn into keras networks
        self.XNet = net_factory(x_dim, scope='XNet', factor=2.0)
        self.VNet = net_factory(x_dim, scope='VNet', factor=1.0)

    def _init_mask(self):
        mask_per_step = []

        for t in range(self.T):
            ind = np.random.permutation(np.arange(self.x_dim))[:int(self.x_dim / 2)]
            m = np.zeros((self.x_dim,))
            m[ind] = 1
            mask_per_step.append(m)

        self.mask = K.constant(np.stack(mask_per_step), dtype=FLOAT_DTYPE)

    def _format_time(self, step, tile):
        raise NotImplementedError()

    def _forward_step(self, x, v, step):
        num_target_params, *_ = K.int_shape(x)
        t = self._format_time(step, tile=num_target_params)

        grad1 = self.grad_energy(x)  # XXX This is actually just K.gradients(loss, params) up to temperature

        S1 = self.VNet([x, grad1, t, None])

        sv1, tv1, fv1 = 0.5 * self.eps * S1[0], S1[1], self.eps * S1[2]

        v_h = v * K.exp(sv1) + 0.5 * self.eps * ((-K.exp(fv1) * grad1) + tv1)

        m, mb = self._get_mask(step)

        X1 = self.XNet([v_h, m * x, t, None])

        sx1, tx1, fx1 = (self.eps * X1[0], X1[1], self.eps * X1[2])
        y = m * x + mb * (x * K.exp(sx1) + self.eps * (K.exp(fx1) * v_h + tx1))

        X2 = self.XNet([v_h, mb * y, t, None])

        sx2, tx2, fx2 = (self.eps * X2[0], X2[1], self.eps * X2[2])

        x_o = mb * y + m * (y * K.exp(sx2) + self.eps * (K.exp(fx2) * v_h + tx2))

        S2 = self.VNet([x_o, self.grad_energy(x_o), t, None])
        sv2, tv2, fv2 = (0.5 * self.eps * S2[0], S2[1], self.eps * S2[2])

        grad2 = self.grad_energy(x_o)
        v_o = v_h * K.exp(sv2) + 0.5 * self.eps * (-K.exp(fv2) * grad2 + tv2)

        log_jac_contrib = K.sum(sv1 + sv2 + mb * sx1 + m * sx2, axis=1)

        return x_o, v_o, log_jac_contrib

    def forward(self, x):
        # performs self.T _forward_step iterations
        v = K.random_normal(K.int_shape(x))

        num_target_params, *_ = K.int_shape(x)
        t = K.constant(0.)
        j = K.zeros((num_target_params,))

        def body(x, v, t, j):
            new_x, new_v, log_j = self._forward_step(x, v, t)
            return new_x, new_v, t + 1, j + log_j

        def condition(x, v, t, j):
            return K.less(t, self.T)

        X, V, t, log_jac_ = while_loop(
            condition=condition, body=body, loop_variables=(x, v, t, j)
        )

        return X, V, self.p_accept(x, v, X, V, log_jac_)

    def _backward_step(self, x_o, v_o, step):
        num_target_params, *_ = K.int_shape(x_o)
        t = self._format_time(step, tile=num_target_params)

        grad1 = self.grad_energy(x_o)

        S1 = self.VNet([x_o, grad1, t, None])
        sv2, tv2, fv2 = (-0.5 * self.eps * S1[0], S1[1], self.eps * S1[2])

        v_h = v_o - 0.5 * self.eps * ((-K.exp(fv2) * grad1) + tv2) * K.exp(sv2)

        m, mb = self._get_mask(step)

        X1 = self.XNet([v_h, mb * x_o, t, None])
        sx2, tx2, fx2 = (-self.eps * X1[0], X1[1], self.eps * X1[2])

        y = mb * x_o + m * K.exp(sx2) * (x_o - self.eps * (K.exp(fx2) * v_h + tx2))

        X2 = self.XNet([v_h, m * y, t, None])
        sx1, tx1, fx1 = (-self.eps * X2[0], X2[1], self.eps * X2[2])

        # New sample `x`
        x = m * y + mb * K.exp(sx1) * (y - self.eps * (K.exp(fx1) * v_h + tx1))

        # Use new sample to update v => this could happen at begin of next iteration?
        grad2 = self.grad_energy(x)

        S2 = self.VNet([x, grad2, t, None])
        sv1, tv1, fv1 = (-0.5 * self.eps * S2[0], S2[1], self.eps * S2[2])

        v = K.exp(sv1) * (v_h - 0.5 * self.eps * (-(K.exp(fv1) * grad2) + tv1))

        return x, v, K.sum(sv1 + sv2 + mb * sx1 + m * sx2, axis=1)

    def backward(self, x):
        # performs self.T many _backward_step iterations, updating x, v and log_j
        # x = sample, v = momentum

        # replace with stateful variable for v and update `params` instead of x?
        # then simply have a node that usually keeps v at current value and assigns a new
        # random normal every self.T iterations.
        v = K.random_normal(K.int_shape(x))

        num_target_params, *_ = K.int_shape(x)
        t = K.constant(0.)
        j = K.zeros((num_target_params,))

        def body(x, v, t, j):
            new_x, new_v, log_j = self._backward_step(x, v, self.T - t - 1)
            return new_x, new_v, t + 1, j + log_j

        def condition(x, v, t, j):
            return K.less(t, self.T)

        X, V, t, log_jac_ = while_loop(
            condition=condition, body=body, loop_variables=(x, v, t, j)
        )

        return X, V, self.p_accept(x, v, X, V, log_jac_)

    def kinetic(self, v):
        return 0.5 * K.sum(K.square(v), axis=1)

    def hamiltonian(self, x, v):
        return self.energy(x) + self.kinetic(v)

    def energy(self, x):
        if self.use_temperature:
            T = self.temperature
        else:
            T = K.constant(1.0, dtype=FLOAT_DTYPE)

        return self._fn(x) / T

    def grad_energy(self, x):
        # XXX: Does this work?
        return K.gradients(self.energy(x), x)[0]

    def p_accept(self, x0, v0, x1, v1, log_jac):
        e_new = self.hamiltonian(x1, v1)
        e_old = self.hamiltonian(x0, v0)

        v = e_old - e_new + log_jac
        p = K.exp(K.minimum(v, 0.0))

        return tf.where(tf.is_finite(p), p, K.zeros_like(p))


class L2HMC(Optimizer):
    """ L2HMC-based optimizer that uses a learned hmc update rule to train a BNN. """

    def __init__(self, energy_function, hyperoptimizer=Adam(), **kwargs):
        super().__init__(**kwargs)
        self.hyperoptimizer = hyperoptimizer
        self.energy_function = energy_function

    # TODO: Factor out into keras utils
    def _keras_accept(x, Lx, px):
        import tensorflow as tf
        mask = (px - K.random_uniform(K.int_shape(px)) >= 0.)
        return tf.where(mask, Lx, x)

    def _propose(self, x, dynamics, do_mh_step=True):
        # sample mask for forward/backward
        # mask = tf.cast(tf.random_uniform((tf.shape(x)[0], 1), maxval=2, dtype=tf.int32), TF_FLOAT)
        num_target_params, *_ = K.int_shape(x)
        mask = K.cast(
            K.random_uniform((num_target_params, 1), maxval=2, dtype=INTEGER_DTYPE),
            FLOAT_DTYPE
        )

        Lx1, Lv1, px1 = dynamics.forward(x)
        Lx2, Lv2, px2 = dynamics.backward(x)

        Lx = mask * Lx1 + (1 - mask) * Lx2

        Lv = None

        px = K.squeeze(mask, axis=1) * px1 + K.squeeze(1 - mask, axis=1) * px2

        outputs = []

        if do_mh_step:
            outputs.append(self.tf_accept(x, Lx, px))

        return Lx, Lv, px, outputs

    def hyperloss(self, x, z, Lx, Lz, px, pz, scale=0.1):
        v1 = (K.sum(K.square(x - Lx), axis=1) * px) + 1e-4
        v2 = (K.sum(K.square(z - Lz), axis=1) * pz) + 1e-4

        return (scale * (K.mean(1.0 / v1) + K.mean(1.0 / v2))) + ((-K.mean(v1) - K.mean(v2)) / scale)

    def architecture(self):
        # XXX: Fix up layers, we can first take them from l2hmc repo and later
        # turn them into actual keras layers and have "net" as a keras network
        def network(x_dim, scope, factor):
            with K.name_scope(scope):
                net = Sequential([
                    Zip([
                        Linear(x_dim, 10, scope='embed_1', factor=1.0 / 3),
                        Linear(x_dim, 10, scope='embed_2', factor=factor * 1.0 / 3),
                        Linear(2, 10, scope='embed_3', factor=1.0 / 3),
                        lambda _: 0.,
                    ]),
                    sum,
                    tf.nn.relu,
                    Linear(10, 10, scope='linear_1'),
                    tf.nn.relu,
                    Parallel([
                        Sequential([
                            Linear(10, x_dim, scope='linear_s', factor=0.001),
                            ScaleTanh(x_dim, scope='scale_s')
                        ]),
                        Linear(10, x_dim, scope='linear_t', factor=0.001),
                        Sequential([
                            Linear(10, x_dim, scope='linear_f', factor=0.001),
                            ScaleTanh(x_dim, scope='scale_f'),
                        ])
                    ])
                ])
                return net
        return network

    def get_updates(self, loss, params):

        x = to_vector(params)
        num_target_params, *_ = K.int_shape(x)

        z = K.random_normal_variable(shape=x.shape, mean=0., scale=1.)

        dynamics = Dynamics(
            x_dim=num_target_params,
            T=10, eps=0.1, net_factory=self.architecture,
            energy_function=self.energy_function
        )

        Lx, _, px, output = self.propose(x, dynamics, do_mh_step=True)
        Lz, _, pz, _ = self.propose(z, dynamics, do_mh_step=False)

        hyperloss = self.hyperloss(x=x, z=z, Lx=Lx, Lz=Lz, px=px, pz=pz)

        # XXX: I assume I need to pass all variables of our architecture net
        # and z here, hard to say though, it might be worth investigating what really happens
        # in the L2HMC code in the respective optimizer op.
        hyperupdates = self.hyperoptimizer(loss=hyperloss, params=[x, z])

        # NOTE: hyperupdates assigns only to latent variables of architecture net
        # and z. These updates need to occur, but we need to evaluate the net
        # with the new parameters to obtain a new sample to assign to `params!`

        # NOTE 2: It might be significantly easier to support this kind of
        # method in our old methodology (on master branch).

        return hyperupdates
