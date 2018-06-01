import numpy as np
import sympy
import torch


# XXX: Make this more stand-alone, we don't want people having to see all the details of what goes on.
# XXX: We need to vectorize dfdx and dxdh to properly compute dot products.

def vectorize(tensor):
    return tensor
    total_dimensions = np.prod(tensor.shape)
    return tensor.reshape((total_dimensions,))


# XXX: What is so slow here?
def hypergradient(hyperparameter_name, sympy_graph, torch_dict, dfdx):
    assert sorted(sympy_graph.symbols.keys()) == sorted(torch_dict.keys())

    assert hyperparameter_name in sympy_graph.symbols

    parameter_names = sympy_graph.symbols.keys()

    sympy_tensors = tuple(sympy_graph.symbols[name] for name in parameter_names)
    torch_params = tuple(torch_dict[name] for name in parameter_names)

    # Compute dxdh [derivative of parameter x with respect to hyperparameter h]
    dxdh_sympy = sympy.diff(
        sympy_graph.update_rule, sympy_graph.symbols[hyperparameter_name]
    )
    lambdified_dxdh = sympy.lambdify(
        args=sympy_tensors, expr=dxdh_sympy,
        modules={
            "sqrt": torch.sqrt,
            # "Max": lambda a, b: torch.clamp(b, min=a),
        }
    )

    dxdh = lambdified_dxdh(*torch_params)

    # Return dfdh by chain rule application.
    return torch.dot(torch.reshape(dxdh, (np.prod(dxdh.shape),)), dfdx)

class Hyperoptimizer(object):
    def __init__(self,
                 hyperparameter,
                 torch_optimizer_cls=torch.optim.Adam,
                 **torch_optimizer_kwargs):
        self.hyperparameter = hyperparameter
        self.hyperoptimizer = torch_optimizer_cls(
            (hyperparameter,), **torch_optimizer_kwargs
        )

    def hypergradient(self, update_rule_tensor, parameter_gradient, torch_tensors):
        sympy_graph = SympyGraph.from_torch_tensor(update_rule_tensor)
        hyperparameter_gradient = sympy_graph.derivative(
            tensor_name=self.hyperparameter.tensor_name,
            torch_tensors=torch_tensors
        )

        return torch.dot(hyperparameter_gradient, parameter_gradient)

    def hyperupdate(self, update_rule_tensor, parameter_gradient, torch_tensors):
        self.hyperoptimizer.zero_grad()

        hypergradient = self.hypergradient(
            update_rule_tensor=update_rule_tensor,
            parameter_gradient=parameter_gradient,
            torch_tensors=torch_tensors
        )
        self.hyperparameter.grad = hypergradient
        self.hyperparameter.grad.data = hypergradient

        self.hyperoptimizer.step()
