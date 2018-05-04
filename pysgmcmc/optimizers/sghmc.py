# vim: foldmethod=marker
import numpy as np
import torch
from torch.optim import Optimizer

# XXX: Something is still buggy, maybe try fixing up a test vs theano_mcmc?


class SGHMC(Optimizer):
    name = "SGHMC"

    def __init__(self,
                 params,
                 lr: float=0.01,
                 num_burn_in_steps: int=3000,
                 mdecay: float=0.05,
                 scale_grad: float=1.) -> None:
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if num_burn_in_steps < 0:
            raise ValueError("Invalid num_burn_in_steps: {}".format(num_burn_in_steps))


        defaults = dict(
            lr=lr, scale_grad=scale_grad,
            num_burn_in_steps=num_burn_in_steps,
            mdecay=mdecay,
            noise=0.
        )
        super().__init__(params, defaults)

    def step(self, closure=None):
        loss = None

        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for parameter in group["params"]:

                if parameter.grad is None:
                    continue


                state = self.state[parameter]

                #  State initialization {{{ #

                if len(state) == 0:
                    state["iteration"] = 0
                    state["tau"] = torch.ones_like(parameter)
                    state["g"] = torch.ones_like(parameter)
                    state["v_hat"] = torch.ones_like(parameter)
                    state["momentum"] = torch.zeros_like(parameter)
                #  }}} State initialization #

                state["iteration"] += 1

                #  Readability {{{ #
                mdecay, lr = group["mdecay"], group["lr"]
                noise = group["noise"]
                scale_grad = torch.tensor(group["scale_grad"]).float()

                tau, g, v_hat = state["tau"], state["g"], state["v_hat"]
                momentum = state["momentum"]

                gradient = parameter.grad.data
                #  }}} Readability #

                r_t = 1. / (tau + 1.)
                minv_t = 1. / torch.sqrt(v_hat)

                #  Burn-in updates {{{ #
                if state["iteration"] <= group["num_burn_in_steps"]:
                    # Update state
                    tau.add_(1. - tau * (g * g / v_hat))
                    g.add_(-g * r_t + r_t * gradient)
                    v_hat.add_(-v_hat * r_t + r_t * (gradient ** 2))
                #  }}} Burn-in updates #

                lr_scaled = lr / torch.sqrt(scale_grad)

                #  Draw random sample {{{ #

                noise_scale = (
                    2. * (lr_scaled ** 2) * mdecay * minv_t -
                    2. * (lr_scaled ** 3) * (minv_t ** 2) * noise -
                    lr_scaled ** 4
                )

                sigma = torch.sqrt(torch.clamp(noise_scale, min=1e-16))

                sample_t = torch.normal(mean=torch.tensor(0.)) * sigma
                #  }}} Draw random sample #


                #  SGHMC Update {{{ #
                momentum.add_(
                    - (lr ** 2) * minv_t * gradient - mdecay * momentum + sample_t
                )

                parameter.data.add_(momentum)
                #  }}} SGHMC Update #

        return loss
