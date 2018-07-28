import torch
from typing import Iterable, Callable, Union, NewType

Predictions = Iterable[torch.Tensor]
Targets = Iterable[torch.Tensor]
TorchLossFunction = Callable[[Predictions, Targets], torch.Tensor]
BNN_NLL = Callable[[Iterable[torch.Tensor], int], TorchLossFunction]
TorchLoss = Union[BNN_NLL, Callable[[], TorchLossFunction]]

VariancePrior = Callable[[torch.Tensor], torch.Tensor]
WeightPrior = Callable[[Iterable[torch.Tensor]], torch.Tensor]

InputDimension = int
NetworkFactory = Callable[[InputDimension], torch.nn.Module]
