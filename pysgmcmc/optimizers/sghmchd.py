# vim: foldmethod=marker
from collections import namedtuple, OrderedDict
import sympy
import torch
import numpy as np
from torch.optim import Optimizer


SympyGraph = namedtuple("SympyGraph", ["update_rule", "symbols"])


class SGHMCHD(Optimizer):
    name = "SGHMCHD"

    def __init__(self,
                 params,
                 hypergradients_for=("lr",),
                 lr: float=1e-2,
                 num_burn_in_steps: int=3000,
                 mdecay: float=0.05,
                 scale_grad: float=1.) -> None:
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if num_burn_in_steps < 0:
            raise ValueError("Invalid num_burn_in_steps: {}".format(num_burn_in_steps))

        def sympy_graph(burn_in):

            #  Symbolic Graph for burn-in update {{{ #
            tau_sympy = sympy.symbols("tau")
            v_hat_sympy = sympy.symbols("v_hat")
            momentum_sympy = sympy.symbols("momentum")

            lr_sympy = sympy.symbols("lr")
            mdecay_sympy = sympy.symbols("mdecay")
            noise_sympy = sympy.symbols("noise")

            gradient_sympy = sympy.symbols("gradient")

            random_sample_sympy = sympy.symbols("random_sample")
            symbols = OrderedDict((
                ("tau", tau_sympy),
                ("v_hat", v_hat_sympy),
                ("momentum", momentum_sympy),
                ("lr", lr_sympy),
                ("mdecay", mdecay_sympy),
                ("noise", noise_sympy),
                ("gradient", gradient_sympy),
                ("random_sample", random_sample_sympy)
            ))

            #  }}} Readability #

            r_t = 1. / (tau_sympy + 1.)

            #  Burn-in updates {{{ #
            if burn_in:
                v_hat_t_sympy = v_hat_sympy - v_hat_sympy * r_t + r_t * (gradient_sympy ** 2)
            else:
                v_hat_t_sympy = v_hat_sympy
            #  }}} Burn-in updates #

            minv_t_sympy = 1. / sympy.sqrt(v_hat_t_sympy)

            lr_scaled_sympy = lr_sympy / sympy.sqrt(scale_grad)

            #  Draw random sample {{{ #

            noise_scale_sympy = (
                2. * (lr_scaled_sympy ** 2) * mdecay_sympy * minv_t_sympy -
                2. * (lr_scaled_sympy ** 3) * (minv_t_sympy ** 2) * noise_sympy -
                (lr_scaled_sympy ** 4)
            )

            # sigma_sympy = sympy.sqrt(sympy.Max(noise_scale_sympy, 1e-16))

            # sample_t = torch.normal(mean=0., std=torch.tensor(1.)) * sigma
            sample_t_sympy = random_sample_sympy * noise_scale_sympy
            #  }}} Draw random sample #

            #  SGHMC Update {{{ #
            momentum_t_sympy = (
                momentum_sympy - (lr_sympy ** 2) * minv_t_sympy * gradient_sympy -
                mdecay_sympy * momentum_sympy + sample_t_sympy
            )

            return SympyGraph(update_rule=momentum_t_sympy, symbols=symbols)

        #  }}} SGHMC Update #

        burn_in_graph = sympy_graph(burn_in=True)
        sampling_graph = sympy_graph(burn_in=False)

        from collections import defaultdict
        self.derivatives = defaultdict(dict)

        for tensorname in hypergradients_for:
            self.derivatives["burn-in"][tensorname] = sympy.lambdify(
                args=burn_in_graph.symbols.values(),
                expr=sympy.diff(
                    burn_in_graph.update_rule,
                    burn_in_graph.symbols[tensorname]
                ),
                modules={"sqrt": torch.sqrt}
            )

            self.derivatives["sampling"][tensorname] = sympy.lambdify(
                args=sampling_graph.symbols.values(),
                expr=sympy.diff(
                    sampling_graph.update_rule,
                    sampling_graph.symbols[tensorname]
                ),
                modules={"sqrt": torch.sqrt}
            )

        #  }}} Symbolic Graph for burn-in update #

        defaults = dict(
            lr=lr, scale_grad=float(scale_grad),
            num_burn_in_steps=num_burn_in_steps,
            mdecay=mdecay,
            symbolic_graphs={
                "burn_in": sympy_graph(burn_in=True), "sampling": sympy_graph(burn_in=False)
            },
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
                    state["lr"] = torch.tensor(
                        torch.ones_like(parameter) * group["lr"], requires_grad=True
                    )

                #  }}} State initialization #

                state["iteration"] += 1

                #  Readability {{{ #
                mdecay, noise, lr = group["mdecay"], group["noise"], state["lr"]
                scale_grad = torch.tensor(group["scale_grad"])

                tau, g, v_hat = state["tau"], state["g"], state["v_hat"]
                momentum = state["momentum"]

                gradient = parameter.grad.data
                #  }}} Readability #

                r_t = 1. / (tau + 1.)

                random_sample = torch.normal(mean=0., std=torch.tensor(1.))
                torch_tensors = OrderedDict((
                    ("tau", tau),
                    ("v_hat", v_hat),
                    ("momentum", momentum),
                    ("lr", lr),
                    ("mdecay", mdecay),
                    ("noise", noise),
                    ("gradient", gradient),
                    ("random_sample", random_sample)
                ))

                if state["iteration"] <= group["num_burn_in_steps"]:
                    derivatives = self.derivatives["burn-in"]
                else:
                    derivatives = self.derivatives["sampling"]

                dxdlr = derivatives["lr"](*torch_tensors.values())
                # dxdlr_t = torch.reshape(dxdlr, (np.prod(dxdlr.shape),))
                # gradient_t = torch.reshape(gradient, (np.prod(dxdlr.shape),))
                # dfdlr = torch.dot(dxdlr_t, gradient_t)
                dfdlr = dxdlr * gradient

                from torch.optim import Adam
                if "hyperoptimizer" not in state:
                    state["hyperoptimizer"] = Adam(
                        params=(state["lr"],),
                        lr=1e-5,
                    )
                state["hyperoptimizer"].zero_grad()
                state["lr"].grad = dfdlr
                state["lr"].grad.data = dfdlr

                state["hyperoptimizer"].step()
                # print(state["lr"])

                #  Burn-in updates {{{ #
                if state["iteration"] <= group["num_burn_in_steps"]:
                    # Update state
                    tau.add_(1. - tau * (g * g / v_hat))
                    g.add_(-g * r_t + r_t * gradient)
                    v_hat.add_(-v_hat * r_t + r_t * (gradient ** 2))

                    derivatives = self.derivatives["burn-in"]
                else:
                    derivatives = self.derivatives["sampling"]

                # XXX: Get this to work efficiently
                """
                sympy_tensors = None
                torch_tensors = None
                group["derivatives"]["lr"] = sympy.lambdify(
                    args=sympy_tensors,
                    expr=sympy.diff(
                        symbolic_graph.update_rule, symbolic_graph.symbols["lr"]
                    ),
                    modules={
                        "sqrt": torch.sqrt,
                        # "Max": lambda a, b: torch.clamp(b, min=a),
                    }
                )(*torch_tensors)
                """

                #  }}} Burn-in updates #

                minv_t = 1. / torch.sqrt(v_hat)

                lr_scaled = lr / torch.sqrt(scale_grad)

                #  Draw random sample {{{ #

                noise_scale = (
                    2. * (lr_scaled ** 2) * mdecay * minv_t -
                    2. * (lr_scaled ** 3) * (minv_t ** 2) * noise -
                    (lr_scaled ** 4)
                )

                # XXX: This seems unnecessary?
                # sigma = torch.sqrt(torch.clamp(noise_scale, min=1e-16))
                # sample_t = torch.normal(mean=0., std=torch.tensor(1.)) * (sigma ** 2)
                sample_t = random_sample * noise_scale
                #  }}} Draw random sample #

                #  SGHMC Update {{{ #
                momentum_t = momentum.add_(
                    - (lr ** 2) * minv_t * gradient - mdecay * momentum + sample_t
                )

                parameter.data.add_(momentum_t)
                #  }}} SGHMC Update #

                # XXX: Try splitting apart learning rates by dimension and
                # optimizing that independently.

                """
                dxdlr_t = torch.reshape(dxdlr, (np.prod(dxdlr.shape),))
                # print("DXDLR", dxdlr_t)

                gradient_t = torch.reshape(gradient, (np.prod(dxdlr.shape),))

                print(torch.where(gradient_t > 0, torch.tensor(1.), torch.tensor(-1.)))

                dfdlr = torch.dot(dxdlr_t, gradient_t)

                from torch.optim import Adam
                if "hyperoptimizer" not in state:
                    state["hyperoptimizer"] = Adam(
                        params=(group["lr"],),
                        lr=1e-5,
                    )

                state["hyperoptimizer"].zero_grad()
                group["lr"].grad = dfdlr
                group["lr"].grad.data = dfdlr

                state["hyperoptimizer"].step()
                print(group["lr"])
                """

        return loss
