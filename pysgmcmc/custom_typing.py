""" Type aliases for custom types, in particular for keras."""
from typing import (
    Union, Callable,
)
import tensorflow as tf
from theano.tensor.var import TensorVariable as TheanoVariable
from keras.optimizers import Optimizer
from keras.models import Model


KerasTensor = Union[tf.Tensor, TheanoVariable]
KerasVariable = Union[tf.Variable, TheanoVariable]
KerasOptimizer = Optimizer

KerasLabels = KerasTensor
KerasPredictions = KerasTensor
KerasLoss = KerasTensor
KerasLossFunction = Callable[[KerasLabels, KerasPredictions], KerasLoss]

KerasModelLoss = Callable[[Model, int, int], KerasLossFunction]

InputDimension = int
RandomSeed = int
KerasNetworkFactory = Callable[[InputDimension, RandomSeed], Model]
