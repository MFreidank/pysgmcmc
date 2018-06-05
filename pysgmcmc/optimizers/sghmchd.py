# vim: foldmethod=marker
from collections import defaultdict, namedtuple, OrderedDict
import typing

import sympy
import torch
from torch.optim import Optimizer

SympyGraph = namedtuple("SympyGraph", ["update_rule", "symbols"])


class SGHMCHD(Optimizer):
    name = "SGHMCHD"

    def __init__(self,
                 params,
                 hypergradients_for: typing.Tuple[str]=(
                     "lr",
                     "mdecay",
                     "noise",
                 ),
                 lr: float=1e-2,
                 num_burn_in_steps: int=3000,
                 mdecay: float=0.05,
                 noise: float=1e-32,
                 scale_grad: float=1.) -> None:
        """ Stochastic Gradient Hamiltonian Monte-Carlo with Hypergradient Descent.
            TODO: Explanation

        Parameters
        ----------
        params: TODO
            Iterable of parameters to optimize or `dict`s defining parameter groups.
        hypergradients_for : typing.Tuple[str], optional
        lr : float, optional
            Initial value for the learning rate/stepsize parameter of SGHMC.
            If `"lr"` is specified in `hypergradients_for`, this quantity will be tuned with
            Hypergradient Descent. Otherwise it stays fixed.
            Defaults to `0.01`.
        num_burn_in_steps: int, optional
            Number of SGHMC burn-in update steps to perform.
            Defaults to `3000`.
        mdecay: float, optional
            Initial value for the momentum decay parameter of SGHMC.
            If `"mdecay"` is specified in `hypergradients_for`, this quantity will be tuned with
            Hypergradient Descent. Otherwise it stays fixed.
            Defaults to `0.05`.
        noise: float, optional
            Initial value for the noise level parameter of SGHMC.
            If `"noise"` is specified in `hypergradients_for`, this quantity will be tuned with
            Hypergradient Descent. Otherwise it stays fixed.
            Defaults to `1e-32`.
        scale_grad: float, optional
            TODO: DOKU

        """
        num_burn_in_steps = 13000
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if num_burn_in_steps < 0:
            raise ValueError("Invalid num_burn_in_steps: {}".format(num_burn_in_steps))

        self.hypergradients_for = hypergradients_for

        #  Construct hypergradient derivative functions with sympy {{{ #

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

        #  }}} Construct hypergradient derivative functions with sympy #

        defaults = dict(
            lr=lr, scale_grad=float(scale_grad),
            num_burn_in_steps=num_burn_in_steps,
            mdecay=mdecay,
            noise=noise
        )
        super().__init__(params, defaults)

    def step(self, closure=None):
        """ Performs a single optimization step.
            Hypergradient Descent updates for all hyperparameters
            specified in `hypergradients_for` are done and optimization is
            subsequently done using `SGHMC` with those updated hyperparameters.

        Parameters
        ----------
        closure : TODO, optional
            A closure that reevaluates the model and returns the loss.

        Returns
        ----------
        TODO

        Examples
        ----------
        TODO

        """
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
                    state["mdecay"] = torch.tensor(
                        torch.ones_like(parameter) * group["mdecay"], requires_grad=True
                    )
                    state["noise"] = torch.tensor(torch.ones_like(parameter) * group["noise"], requires_grad=True)
                    # from torch.optim import Adam
                    from torch.optim import Adamax
                    # state["hyperoptimizers"] = {
                    #     step: {
                    #         tensor_name: Adam(params=(state[tensor_name],), lr=1e-5)
                    #         for tensor_name in self.hypergradients_for
                    #     }
                    #     for step in ("burn-in", "sampling")
                    # }
                    state["hyperoptimizers"] = {
                        "burn-in": Adamax(params=(state[tensor_name] for tensor_name in self.hypergradients_for), lr=1e-5)
                    }

                #  }}} State initialization #

                state["iteration"] += 1

                #  Readability {{{ #
                mdecay, noise, lr = state["mdecay"], group["noise"], state["lr"]
                scale_grad = torch.tensor(group["scale_grad"])

                tau, g, v_hat = state["tau"], state["g"], state["v_hat"]
                momentum = state["momentum"]

                gradient = parameter.grad.data
                #  }}} Readability #

                r_t = 1. / (tau + 1.)

                random_sample = torch.normal(mean=0., std=torch.tensor(1.))

                #  Hypergradient updates {{{ #
                # Dictionary mapping tensor names to pytorch tensors.
                # Used to compute derivatives in sympy.
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

                stage = "burn-in" if state["iteration"] <= group["num_burn_in_steps"] else "sampling"
                # XXX: Move actual parameter updates *after* sghmc update?
                hyperoptimizer = state["hyperoptimizers"]["burn-in"]
                hyperoptimizer.zero_grad()
                for tensor_name in self.hypergradients_for:
                    # derivative of parameters `x` with respect to hyperparameter `h` with name `tensor_name`.
                    dxdh = self.derivatives[stage][tensor_name](*torch_tensors.values())
                    # Apply chain rule to compute derivative of `loss` with respect to hyperparameter `h`.
                    dfdh = dxdh * gradient

                    # NOTE: If we want one hyperoptimizer for all things, we want to do a single `step` and move zero_grad outside as well
                    # Assign gradient that we computed with sympy to pytorch `grad` attribute.
                    state[tensor_name].grad = dfdh
                    state[tensor_name].grad.data = dfdh

                    # hyperoptimizer.step()
                #  }}} Hypergradient updates #

                #  Burn-in updates {{{ #
                if state["iteration"] <= group["num_burn_in_steps"]:
                    # Update state
                    tau.add_(1. - tau * (g * g / v_hat))
                    g.add_(-g * r_t + r_t * gradient)
                    v_hat.add_(-v_hat * r_t + r_t * (gradient ** 2))

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
                # TODO: Why noise scale here? It used to be std=sigma but multiplying
                # a standard normal sample with `sigma` breaks...
                sample_t = random_sample * noise_scale
                #  }}} Draw random sample #

                #  SGHMC Update {{{ #
                momentum_t = momentum.add_(
                    - (lr ** 2) * minv_t * gradient - mdecay * momentum + sample_t
                )

                parameter.data.add_(momentum_t)
                #  }}} SGHMC Update #

                hyperoptimizer.step()

        return loss


class SGHMCHDSplitOptimizers(SGHMCHD):
    """ For this variant, each hypergradient is optimized with its own independent hyperoptimizer. """
    def step(self, closure=None):
        """ Performs a single optimization step.
            Hypergradient Descent updates for all hyperparameters
            specified in `hypergradients_for` are done and optimization is
            subsequently done using `SGHMC` with those updated hyperparameters.

        Parameters
        ----------
        closure : TODO, optional
            A closure that reevaluates the model and returns the loss.

        Returns
        ----------
        TODO

        Examples
        ----------
        TODO

        """
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
                    state["mdecay"] = torch.tensor(
                        torch.ones_like(parameter) * group["mdecay"], requires_grad=True
                    )
                    state["noise"] = torch.tensor(torch.ones_like(parameter) * group["noise"], requires_grad=True)
                    # from torch.optim import Adam
                    from torch.optim import Adamax
                    # state["hyperoptimizers"] = {
                    #     step: {
                    #         tensor_name: Adam(params=(state[tensor_name],), lr=1e-5)
                    #         for tensor_name in self.hypergradients_for
                    #     }
                    #     for step in ("burn-in", "sampling")
                    # }
                    state["hyperoptimizers"] = {
                        "burn-in": {tensor_name: Adamax(params=(state[tensor_name],), lr=1e-5) for tensor_name in self.hypergradients_for},
                        "sampling": {tensor_name: Adamax(params=(state[tensor_name],), lr=1e-5) for tensor_name in self.hypergradients_for}

                    }
                    # state["hyperoptimizers"] = {
                    #     "burn-in": Adamax(
                    #         params=tuple(state[tensor_name] for tensor_name in self.hypergradients_for),
                    #         lr=1e-5
                    #     ),
                    #     "sampling": Adamax(
                    #         params=tuple(state[tensor_name] for tensor_name in self.hypergradients_for),
                    #         lr=1e-5
                    #     )
                    # }

                #  }}} State initialization #

                state["iteration"] += 1

                #  Readability {{{ #
                mdecay, noise, lr = state["mdecay"], group["noise"], state["lr"]
                scale_grad = torch.tensor(group["scale_grad"])

                tau, g, v_hat = state["tau"], state["g"], state["v_hat"]
                momentum = state["momentum"]

                gradient = parameter.grad.data
                #  }}} Readability #

                r_t = 1. / (tau + 1.)

                random_sample = torch.normal(mean=0., std=torch.tensor(1.))

                #  Hypergradient updates {{{ #
                # Dictionary mapping tensor names to pytorch tensors.
                # Used to compute derivatives in sympy.
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

                stage = "burn-in" if state["iteration"] <= group["num_burn_in_steps"] else "sampling"
                # XXX: Move actual parameter updates *after* sghmc update?
                for tensor_name in self.hypergradients_for:
                    # derivative of parameters `x` with respect to hyperparameter `h` with name `tensor_name`.
                    dxdh = self.derivatives[stage][tensor_name](*torch_tensors.values())
                    # Apply chain rule to compute derivative of `loss` with respect to hyperparameter `h`.
                    dfdh = dxdh * gradient

                    hyperoptimizer = state["hyperoptimizers"]["burn-in"][tensor_name]
                    # NOTE: If we want one hyperoptimizer for all things, we want to do a single `step` and move zero_grad outside as well
                    hyperoptimizer.zero_grad()
                    # Assign gradient that we computed with sympy to pytorch `grad` attribute.
                    state[tensor_name].grad = dfdh
                    state[tensor_name].grad.data = dfdh

                    # hyperoptimizer.step()
                #  }}} Hypergradient updates #

                #  Burn-in updates {{{ #
                if state["iteration"] <= group["num_burn_in_steps"]:
                    # Update state
                    tau.add_(1. - tau * (g * g / v_hat))
                    g.add_(-g * r_t + r_t * gradient)
                    v_hat.add_(-v_hat * r_t + r_t * (gradient ** 2))

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
                # TODO: Why noise scale here? It used to be std=sigma but multiplying
                # a standard normal sample with `sigma` breaks...
                sample_t = random_sample * noise_scale
                #  }}} Draw random sample #

                #  SGHMC Update {{{ #
                momentum_t = momentum.add_(
                    - (lr ** 2) * minv_t * gradient - mdecay * momentum + sample_t
                )

                parameter.data.add_(momentum_t)
                #  }}} SGHMC Update #

                for tensor_name in self.hypergradients_for:
                    hyperoptimizer = state["hyperoptimizers"]["burn-in"][tensor_name]
                    hyperoptimizer.step()

        return loss
