# vim:foldmethod=marker
import typing
import tensorflow as tf
from keras import backend as K
from keras.optimizers import Optimizer
from pysgmcmc.custom_typing import KerasTensor, KerasVariable

from collections import OrderedDict
import sympy
import warnings
import functools


original_random_normal = K.random_normal


def to_hyperoptimizer(optimizer):
    def hyperupdate(optimizer, grads, params):
        optimizer.get_gradients = lambda *args, **kwargs: grads
        return optimizer.get_updates(loss=None, params=params)
    optimizer.hyperupdate = lambda grads, params: hyperupdate(optimizer, grads, params)
    return optimizer


def sympy_compatibility(tensor_constructor):
    """ Makes `tensor_constructor` compatible with hypergradient differentiation based on sympy.
        Concretely, this is a decorator that adds a field `sympy_name` to
        the tensor constructed by `tensor_constructor`.
        Using this field we can easily generate a `sympy.symbol` for a tensor
        we constructed with `tensor_constructor`.
        The `sympy_name` is taken from the passed in `name` argument to `tensor_constructor`.

    Parameters
    ----------
    tensor_constructor : TODO
        A function that constructs tensors, e.g. `K.ones`.

    Returns
    ----------
    compatible_tensor_constructor : TODO
        A function that calls `tensor_constructor` and then adds a string field
        `sympy_name` to the constructed tensor.

    """

    @functools.wraps(tensor_constructor)
    def construct(*args, name=None, **kwargs):
        if tensor_constructor != original_random_normal:
            tensor = tensor_constructor(name=name, *args, **kwargs)
        else:
            tensor = tensor_constructor(*args, **kwargs)
        tensor.sympy_name = name
        return tensor
    return construct

K.constant = sympy_compatibility(K.constant)
K.random_normal = sympy_compatibility(K.random_normal)
K.ones = sympy_compatibility(K.ones)
K.zeros = sympy_compatibility(K.zeros)


def heaviside(x):
    return K.switch(x < 0, K.zeros_like(x, dtype=K.floatx()), K.ones_like(x, dtype=K.floatx()))


def maximum(a, b):
    return K.maximum(K.cast(a, K.floatx()), K.cast(b, K.floatx()))


# XXX COnstruct sympy graph exactly once and pass different things in
def sympy_derivative(with_respect_to: str, tensor_names: typing.List[str]):
    print(tensor_names)

    assert with_respect_to in tensor_names

    symbols = OrderedDict((
        (name, sympy.symbols(name))
        for name in tensor_names
    ))

    g2, p = symbols["g2"], symbols["p"]
    epsilon, mdecay, noise = symbols["epsilon"], symbols["mdecay"], symbols["noise"]
    grad, scale_grad = symbols["grad"], symbols["scale_grad"]
    random_sample = symbols["random_sample"]

    Minv = 1. / (sympy.sqrt(g2 + K.epsilon()) + K.epsilon())

    epsilon_scaled = epsilon / sympy.sqrt(scale_grad)
    noise_scale = 2. * (epsilon_scaled ** 2) * mdecay * Minv - 2. * epsilon_scaled ** 3 * (Minv ** 2) * noise
    sigma = sympy.sqrt(sympy.Max(noise_scale, K.epsilon()))

    sample_t = random_sample * sigma
    p_t = p - (epsilon ** 2) * Minv * grad - mdecay * p + sample_t

    derivative = sympy.diff(
        p_t, symbols[with_respect_to]
    )
    print(derivative)

    # Callable derivative function that computes d_theta d_`with respect to`
    return sympy.lambdify(
        args=tuple(symbols.values()),
        expr=derivative,
        modules={"sqrt": K.sqrt, "Heaviside": heaviside, "Max": maximum}
    )


class SGHMCHD(Optimizer):
    def __init__(self,
                 lr: float=0.01,
                 mdecay: float=0.05,
                 noise: float=0.,
                 burn_in_steps: int=3000,
                 scale_grad: float=1.0,
                 hyperloss=None,
                 seed: int=None,
                 **kwargs) -> None:
        print("using new sghmchd")
        super(SGHMCHD, self).__init__(**kwargs)
        self.seed = seed
        self.initial_lr = lr
        self.initial_mdecay = mdecay
        self.burn_in_steps = K.constant(burn_in_steps, dtype="int64")
        self.scale_grad = K.constant(scale_grad, dtype=K.floatx())
        self.scale_grad_val = scale_grad
        self.initial_noise = noise

        self.hyperloss = hyperloss

        self.iterations = K.variable(0, dtype="int64", name="iterations")

    def burning_in(self):
        return self.iterations < self.burn_in_steps

    def noise_sample(self, shape):
        return K.random_normal(shape=shape, seed=self.seed)

    def get_updates(self, loss, params):
        print(loss, loss.shape)
        grads = self.get_gradients(loss, params)

        self.updates = [K.update_add(self.iterations, 1)]

        derivative_function = sympy_derivative(
            with_respect_to="epsilon", tensor_names=(
                "g2", "p", "epsilon", "mdecay", "noise",
                "grad", "scale_grad", "random_sample"
            )
        )

        from keras.optimizers import Adamax
        import numpy as np
        hyperoptimizers = {
            theta.name: to_hyperoptimizer(Adamax(lr=0.002 / np.sqrt(self.scale_grad_val))) for theta in params
        }

        for (theta, grad) in zip(params, grads):
            epsilon = K.variable(
                K.constant(self.initial_lr, shape=theta.shape, name="epsilon")
            )
            if K.backend() == "tensorflow":
                import tensorflow as tf
                tf.summary.histogram("epsilon", epsilon)
            mdecay = K.constant(self.initial_mdecay, shape=theta.shape, name="mdecay")
            noise = K.constant(self.initial_noise, shape=theta.shape, name="noise")
            xi = K.ones(theta.shape, name="xi")
            g = K.ones(theta.shape, name="g")
            g2 = K.ones(theta.shape, name="g2")
            p = K.zeros(theta.shape, name="p")

            #  Hypergradient Update {{{ #
            dxdh = K.zeros(theta.shape, name="dxdh")

            random_sample = self.noise_sample(shape=theta.shape)

            dxdh_t = derivative_function(
                g2, p, epsilon, mdecay, noise,
                grad, self.scale_grad, random_sample
            )

            try:
                dxdlr = theta.hypergradient[0]
            except AttributeError:
                warnings.warn(
                    "No hyperloss given, but SGHMCHD is used as optimizer. "
                    "Falling back to standard SGHMC behaviour."
                )
                # if no hyperloss was set, do not change the stepsize.
                dxdlr = K.zeros_like(grad)

            hypergradient = dxdlr * dxdh

            hyperoptimizer = hyperoptimizers[theta.name]

            self.updates.extend([
                hyperoptimizer.hyperupdate(grads=(hypergradient,), params=(epsilon,))
            ])
            #  }}} Hypergradient Update #

            r_t = 1. / (xi + 1.)

            g_t = (1. - r_t) * g + r_t * grad
            g2_t = (1. - r_t) * g2 + r_t * grad**2
            xi_t = 1. + xi * (1. - g * g / (g2 + K.epsilon()))
            Minv = 1. / (K.sqrt(g2 + K.epsilon()) + K.epsilon())

            burning_in = self.iterations < self.burn_in_steps

            self.updates.extend([
                (g, K.switch(burning_in, g_t, K.identity(g))),
                (g2, K.switch(burning_in, g2_t, K.identity(g2))),
                (xi, K.switch(burning_in, xi_t, K.identity(xi))),
                (dxdh, dxdh_t),
            ])

            epsilon_scaled = epsilon / K.sqrt(self.scale_grad)
            noise_scale = 2. * K.square(epsilon_scaled) * mdecay * Minv - 2. * epsilon_scaled ** 3 * K.square(Minv) * noise
            sigma = K.sqrt(K.maximum(noise_scale, K.epsilon()))
            sample_t = random_sample * sigma
            p_t = p - K.square(epsilon) * Minv * grad - mdecay * p + sample_t
            theta_t = theta + p_t
            self.updates.extend([(theta, theta_t), (p, p_t)])

        return self.updates
