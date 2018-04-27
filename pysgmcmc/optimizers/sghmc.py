import numpy as np
import torch
from torch.optim import Optimizer


class SGHMC(Optimizer):
    def __init__(self, params, lr=0.01, mdecay=0.05,
                 scale_grad=1.0, burn_in_steps=3000):

        parameters = tuple(params)
        self.parameter_sizes = [
            np.prod(list(map(int, param.shape))) for param in parameters
        ]

        lr_scaled = lr / np.sqrt(scale_grad)

        defaults = dict(lr=lr, lr_scaled=lr_scaled, mdecay=mdecay, scale_grad=scale_grad,
                        burn_in_steps=burn_in_steps)
        super(SGHMC, self).__init__(parameters, defaults)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        # XXX: Debug this and models/bayesian_neural_network until it works
        loss = None
        if closure is not None:
            loss = closure()

        group, = self.param_groups
        parameters = group["params"]
        grads = tuple(
            torch.reshape(parameter.grad.data, (parameter_size, 1))
            for parameter, parameter_size in
            zip(parameters, self.parameter_sizes)
        )
        gradient = torch.cat(grads)
        x = torch.cat(
            tuple(
                torch.reshape(parameter, (parameter_size, 1))
                for (parameter, parameter_size) in
                zip(parameters, self.parameter_sizes)
            )
        )

        if len(self.state) == 0:
            self.state["tau"] = torch.ones_like(x)
            self.state["g"] = torch.ones_like(x)
            self.state["v_hat"] = torch.ones_like(x)
            self.state["momentum"] = torch.zeros_like(x)

        tau, g = self.state["tau"], self.state["g"]
        v_hat, momentum = self.state["v_hat"], self.state["momentum"]

        group, = self.param_groups
        mdecay = group["mdecay"]
        stepsize, stepsize_scaled = group["lr"], group["lr_scaled"]

        noise = 0

        r_t = 1. / (tau + 1.)
        tau_t = 1. + tau - tau * (g * g * tau / (v_hat + 1e-6))

        minv_t = 1. / (torch.sqrt(v_hat + 1e-6) + 1e-6)

        g_t = g - r_t * g + r_t * gradient

        v_hat_t = v_hat - r_t * v_hat + r_t * gradient ** 2

        noise_scale = (
            2. * stepsize_scaled ** 2. *
            mdecay * minv_t - 2. * stepsize_scaled ** 3. *
            minv_t ** 2 * noise - stepsize_scaled ** 4
        )

        sigma = torch.sqrt(torch.max(noise_scale, torch.Tensor([1e-16])))

        sample = sigma * torch.normal(torch.zeros_like(gradient), torch.ones_like(gradient))

        v_t = momentum - -stepsize ** 2 * minv_t * gradient - mdecay * momentum + sample

        x_t = x + v_t

        self.state["tau"] = tau_t
        self.state["g"] = g_t
        self.state["v_hat"] = v_hat_t
        self.state["v"] = v_t

        for parameter, sampled_values in zip(parameters, torch.split(x_t, self.parameter_sizes)):
            parameter.data.copy_(torch.reshape(sampled_values, parameter.shape))

        return loss
