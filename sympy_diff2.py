#!/usr/bin/python3
# -*- coding: iso-8859-15 -*-
# vim:foldmethod=marker
import warnings
from functools import wraps
from enum import Enum

import sympy
import torch

# 1. XXX: Testing that this works as intended
# 2. XXX: Heavy clean-up of the entire interface and lots of docs
# 3. XXX: More tests (flexibility wrt. what to derive for etc.)


#  Add names to tensors for sympy graphs {{{ #
def add_name(function):
    @wraps(function)
    def wrapped(data, name=None, requires_grad=True, *args, **kwargs):
        tensor = function(data, requires_grad=requires_grad, *args, **kwargs)
        if name is not None:
            tensor.tensor_name = name
        return tensor
    return wrapped

named_tensor = add_name(torch.tensor)

named_ones = add_name(torch.ones)
named_zeros = add_name(torch.zeros)

named_ones_like = add_name(torch.ones_like)
named_zeros_like = add_name(torch.zeros_like)

named_random = add_name(torch.rand)


def tensor_clamp(tensor, min_tensor=torch.tensor(1e-16, requires_grad=True)):
    return torch.where(tensor > min_tensor, tensor, torch.tensor(1e-16, requires_grad=True))
#  }}} Add names to tensors for sympy graphs #


#  Parsing torch graphs {{{ #
class TensorOperation(Enum):
    ConstantTensor = "ConstantTensor"
    Tensor = "Tensor"
    Mul = "Mul"
    Div = "Div"
    Add = "Add"
    Sub = "Sub"
    Sqrt = "Sqrt"
    Pow = "Pow"
    Clamp = "Clamp"


class TorchNode(object):
    def __init__(self, name, operation, constructing_tensors, value=None):
        self.name, self.operation = name, operation
        self.value = value
        self.constructing_tensors = constructing_tensors

    @classmethod
    def from_torch(cls, tensor):
        operation_name = str(tensor)
        print(operation_name)
        if operation_name.startswith("tensor") and tensor.grad_fn is None:
            assert hasattr(tensor, "tensor_name") and tensor.tensor_name is not None
            return TorchNode(
                name=tensor.tensor_name,
                operation=TensorOperation.Tensor,
                constructing_tensors=None
            )
        elif operation_name.startswith("tensor"):
            # tensor that is constructed from an operation
            return TorchNode.from_torch(tensor.grad_fn)
        elif operation_name.startswith("<MulBackward1"):
            (left_operand, _), (right_operand, _) = tensor.next_functions
            left_node = TorchNode.from_torch(left_operand)
            right_node = TorchNode.from_torch(right_operand)
            return TorchNode(
                name="Mul", operation=TensorOperation.Mul,
                constructing_tensors=(left_node, right_node)
            )
        elif operation_name.startswith("<DivBackward1"):
            (left_operand, _), (right_operand, _) = tensor.next_functions
            left_node = TorchNode.from_torch(left_operand)
            right_node = TorchNode.from_torch(right_operand)
            return TorchNode(
                name="Div", operation=TensorOperation.Div,
                constructing_tensors=(left_node, right_node)
            )
        elif operation_name.startswith("<AccumulateGrad"):
            var = tensor.variable
            assert var.grad_fn is None
            try:
                tensor_name = var.tensor_name
            except AttributeError:
                return TorchNode(
                    name="Constant Tensor",
                    operation=TensorOperation.ConstantTensor,
                    constructing_tensors=None,
                    value=var.detach().numpy(),
                )
            else:
                return TorchNode(
                    name=tensor_name,
                    operation=TensorOperation.Tensor,
                    constructing_tensors=None,
                    value=var.detach().numpy(),
                )

        elif operation_name.startswith("<CopyBackwards"):
            _, (argument, _) = tensor.next_functions
            return TorchNode.from_torch(argument)

        elif operation_name.startswith("<SqrtBackward"):
            (sqrt_argument, _), = tensor.next_functions
            argument_node = TorchNode.from_torch(sqrt_argument)
            return TorchNode(
                name="Sqrt", operation=TensorOperation.Sqrt,
                constructing_tensors=(argument_node,)
            )
        elif operation_name.startswith("<AddBackward1"):
            (left_operand, _), (right_operand, _) = tensor.next_functions
            left_node = TorchNode.from_torch(left_operand)
            right_node = TorchNode.from_torch(right_operand)
            return TorchNode(
                name="Add", operation=TensorOperation.Add,
                constructing_tensors=(left_node, right_node)
            )
        elif operation_name.startswith("<SubBackward1"):
            (left_operand, _), (right_operand, _) = tensor.next_functions
            left_node = TorchNode.from_torch(left_operand)
            right_node = TorchNode.from_torch(right_operand)
            return TorchNode(
                name="Sub", operation=TensorOperation.Sub,
                constructing_tensors=(left_node, right_node)
            )
        elif operation_name.startswith("<ExpandBackward"):
            (argument, _), = tensor.next_functions
            return TorchNode.from_torch(argument)
        elif operation_name.startswith("<PowBackward1"):
            (left_operand, _), (right_operand, _) = tensor.next_functions
            left_node = TorchNode.from_torch(left_operand)
            right_node = TorchNode.from_torch(right_operand)
            return TorchNode(
                name="Pow", operation=TensorOperation.Pow,
                constructing_tensors=(left_node, right_node)
            )
        elif operation_name.startswith("<SWhereBackward"):
            # TODO: Allow controlling this through a flag or something like this?
            warnings.warn(
                "To support sympy derivatives for clamped tensors, we treat "
                "all operations of the form: `torch.where(condition, x, y)`\n"
                "as mere aliases of `torch.clamp(x, min=y)`. "
            )
            (left_operand, _), (right_operand, _) = tensor.next_functions
            # left_node = TorchNode.from_torch(left_operand)
            left_node = TorchNode.from_torch(left_operand)
            right_node = TorchNode.from_torch(right_operand)
            return TorchNode(
                name="Clamp", operation=TensorOperation.Clamp,
                constructing_tensors=(left_node, right_node)
            )
        elif operation_name.startswith("<NormalBackward2"):
            return TorchNode(
                name="random_sample", operation=TensorOperation.Tensor,
                constructing_tensors=None
            )

        else:
            raise NotImplementedError(
                "Unsupported operation: {}\n"
                "Please rewrite any operation into a 2-argument format "
                "`operation(tensor1, tensor2)`\n with tensors `tensor1`"
                "and `tensor2` both with `requires_grad=True`.".format(operation_name)
            )

        print(tensor.tensor_name)
        print(tensor.grad_fn)

    def __str__(self):
        if self.operation is TensorOperation.Tensor:
            this_node = self.name
        elif self.operation is TensorOperation.ConstantTensor:
            print(self.name)
            this_node = str(self.value)
        elif self.operation is TensorOperation.Add:
            this_node = "({} + {})".format(*self.constructing_tensors)
        elif self.operation is TensorOperation.Sub:
            this_node = "({} - {})".format(*self.constructing_tensors)
        elif self.operation is TensorOperation.Mul:
            this_node = "({} * {})".format(*self.constructing_tensors)
        elif self.operation is TensorOperation.Div:
            this_node = "({} / {})".format(*self.constructing_tensors)
        elif self.operation is TensorOperation.Sqrt:
            this_node = "sqrt({})".format(*self.constructing_tensors)
        elif self.operation is TensorOperation.Pow:
            this_node = "({} ** {})".format(*self.constructing_tensors)
        elif self.operation is TensorOperation.Clamp:
            this_node = "clamp({}, {})".format(*self.constructing_tensors)
        else:
            raise NotImplementedError(self.operation)
        return this_node
#  }}} Parsing torch graphs #


#  Sympy Graph that can compute derivatives {{{ #

class SympyGraph(object):
    def __init__(self, torch_node):
        self.symbols = {}

        def parse_torch_node(torch_node):
            if torch_node.operation is TensorOperation.Tensor:
                name = torch_node.name
                symbol = sympy.symbols(name)
                self.symbols[name] = symbol
                return symbol
            elif torch_node.operation is TensorOperation.ConstantTensor:
                return torch_node.value
            elif torch_node.operation is TensorOperation.Add:
                left_torch_node, right_torch_node = torch_node.constructing_tensors
                return (
                    parse_torch_node(left_torch_node) + parse_torch_node(right_torch_node)
                )
            elif torch_node.operation is TensorOperation.Sub:
                left_torch_node, right_torch_node = torch_node.constructing_tensors
                return (
                    parse_torch_node(left_torch_node) - parse_torch_node(right_torch_node)
                )
            elif torch_node.operation is TensorOperation.Div:
                left_torch_node, right_torch_node = torch_node.constructing_tensors
                return (
                    parse_torch_node(left_torch_node) / parse_torch_node(right_torch_node)
                )
            elif torch_node.operation is TensorOperation.Mul:
                left_torch_node, right_torch_node = torch_node.constructing_tensors
                return (
                    parse_torch_node(left_torch_node) * parse_torch_node(right_torch_node)
                )
            elif torch_node.operation is TensorOperation.Pow:
                left_torch_node, right_torch_node = torch_node.constructing_tensors
                return (
                    parse_torch_node(left_torch_node) ** parse_torch_node(right_torch_node)
                )
            elif torch_node.operation is TensorOperation.Sqrt:
                argument_torch_node, = torch_node.constructing_tensors
                return sympy.sqrt(parse_torch_node(argument_torch_node))
            elif torch_node.operation is TensorOperation.Clamp:
                left_torch_node, right_torch_node = torch_node.constructing_tensors
                return sympy.Max(parse_torch_node(left_torch_node), parse_torch_node(right_torch_node))

            else:
                raise NotImplementedError(torch_node.operation)

        self.expression = parse_torch_node(torch_node)
        print(self.expression)

    @classmethod
    def from_torch_tensor(cls, tensor):
        return SympyGraph(TorchNode.from_torch(tensor))

    def derivative(self, tensor_name, torch_tensors):
        assert tensor_name in self.symbols
        target_symbol = self.symbols[tensor_name]

        torch_dict = {
            tensor.tensor_name: tensor for tensor in torch_tensors
        }

        print(sorted(self.symbols.keys()), sorted(torch_dict.keys()))

        assert sorted(self.symbols.keys()) == sorted(torch_dict.keys())

        parameter_names = self.symbols.keys()

        sympy_tensors = tuple(self.symbols[name] for name in parameter_names)
        torch_params = tuple(torch_dict[name] for name in parameter_names)

        derivative = sympy.diff(self.expression, target_symbol)
        print(derivative)

        lambdified_function = sympy.lambdify(
            args=sympy_tensors, expr=self.expression,
            modules={"sqrt": torch.sqrt, "Max": lambda a, b: torch.clamp(b, min=a)}
        )
        return lambdified_function(*torch_params)
#  }}} Sympy Graph that can compute derivatives #


def main():
    parameter = named_ones((50, 50), name="parameter")
    momentum = named_zeros_like(parameter, name="momentum")

    mdecay = named_tensor(0.05, name="mdecay")
    noise = named_tensor(0., name="noise")
    scale_grad = named_tensor(100., name="scale_grad")
    lr = named_tensor(1e-3, name="lr")
    minv_t = named_ones_like(parameter, name="minv_t")

    gradient = named_ones_like(parameter, name="gradient")

    lr_scaled = lr / torch.sqrt(scale_grad)

    #  Draw random sample {{{ #

    noise_scale = (
        torch.tensor(2., requires_grad=True) *
        (lr_scaled ** torch.tensor(2., requires_grad=True)) * mdecay * minv_t -
        torch.tensor(2., requires_grad=True) *
        (lr_scaled ** torch.tensor(3., requires_grad=True)) *
        (minv_t ** torch.tensor(2., requires_grad=True)) * noise -
        (lr_scaled ** torch.tensor(3., requires_grad=True))
    )

    sigma = torch.sqrt(tensor_clamp(noise_scale, min_tensor=torch.tensor(1e-16, requires_grad=True)))

    # sample_t = torch.normal(mean=0., std=torch.tensor(1.)) * sigma
    random_sample = named_tensor(torch.normal(mean=0., std=torch.tensor(1., requires_grad=True)), name="random_sample")
    sample_t = random_sample * sigma
    #  }}} Draw random sample #

    #  SGHMC Update {{{ #
    momentum_t = momentum - (lr ** torch.tensor(2., requires_grad=True)) * minv_t * gradient - mdecay * momentum + sample_t

    sympy_graph = SympyGraph.from_torch_tensor(momentum_t)
    print(sympy_graph.derivative("lr", (momentum, mdecay, noise, scale_grad, lr, minv_t, gradient, random_sample)))

    # parameter.data.add_(momentum_t)
    #  }}} SGHMC Update #

if __name__ == "__main__":
    main()
