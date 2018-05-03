#!/usr/bin/python3
# -*- coding: iso-8859-15 -*-

# XXX: Working bnn with adam, all in one - will be here only temporarily to saveguard it
import matplotlib.pyplot as plt
from itertools import islice

import numpy as np
import torch
import torch.nn as nn
from torch.utils import data as data_utils
from torch.nn.modules.loss import _Loss, _assert_no_grad
from torch.optim import Adam


class InfiniteDataLoader(data_utils.DataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.iterator = super().__iter__()

    def __iter__(self):
        return self

    def __next__(self):
        try:
            el = next(self.iterator)
        except StopIteration:
            self.iterator = super().__iter__()
            el = next(self.iterator)
        return el


def network(input_dimensionality: int):
    class AppendLayer(nn.Module):
        def __init__(self, bias=True, *args, **kwargs):
            super().__init__(*args, **kwargs)
            if bias:
                self.bias = nn.Parameter(torch.Tensor(1, 1))
            else:
                self.register_parameter('bias', None)

        def forward(self, x):
            return torch.cat((x, self.bias * torch.ones_like(x)), dim=1)

    def init_weights(module):
        if type(module) == AppendLayer:
            nn.init.constant_(module.bias, val=np.log(1e-3))
        elif type(module) == nn.Linear:
            nn.init.kaiming_normal_(module.weight, nonlinearity="linear")
            nn.init.constant_(module.bias, val=0.0)

    return nn.Sequential(
        nn.Linear(input_dimensionality, 50), nn.Tanh(),
        nn.Linear(50, 50), nn.Tanh(),
        nn.Linear(50, 50), nn.Tanh(),
        nn.Linear(50, 1),
        AppendLayer()
    ).apply(init_weights)


#  Data Preprocessing {{{ #

def zero_mean_unit_var_normalization(X, mean=None, std=None):
    mean = np.mean(X, axis=0) if mean is None else mean
    std = np.std(X, axis=0) if std is None else std
    return np.true_divide(X - mean, std), mean, std


def zero_mean_unit_var_unnormalization(X_normalized, mean, std):
    return X_normalized * std + mean
#  }}} Data Preprocessing #


#  Priors {{{ #
def log_variance_prior(log_variance, mean: float=1e-6, variance: float=0.01):
    return torch.mean(
        torch.sum(
            ((-(log_variance - torch.log(torch.tensor(mean))) ** 2) /
             ((2. * variance))) - 0.5 * torch.log(torch.tensor(variance)),
            dim=1
        )
    )


def weight_prior(parameters, wdecay: float=1.):
    num_parameters = torch.sum(torch.tensor([
        torch.prod(torch.tensor(parameter.size()))
        for parameter in parameters
    ]))

    log_likelihood = torch.sum(torch.tensor([
        torch.sum(-wdecay * 0.5 * (parameter ** 2))
        for parameter in parameters
    ]))

    return log_likelihood / (num_parameters.float() + 1e-16)
#  }}} Prios #


#  Loss {{{ #
class NegativeLogLikelihood(_Loss):
    def __init__(self, parameters, num_datapoints, size_average=False, reduce=True):
        assert (not size_average) and reduce
        super().__init__(size_average, reduce)
        self.parameters = tuple(parameters)
        self.num_datapoints = num_datapoints

    def forward(self, input, target):
        _assert_no_grad(target)

        batch_size, *_ = target.shape
        prediction_mean = input[:, 0].view(-1, 1)

        log_prediction_variance = input[:, 1].view(-1, 1)
        prediction_variance_inverse = 1. / (torch.exp(log_prediction_variance) + 1e-16)

        mean_squared_error = torch.pow(target - prediction_mean, 2)

        log_likelihood = torch.sum(
            torch.sum(
                -mean_squared_error * 0.5 * prediction_variance_inverse -
                0.5 * log_prediction_variance,
                dim=1
            )
        )

        log_likelihood /= batch_size

        log_likelihood += (
            log_variance_prior(log_prediction_variance) / self.num_datapoints
        )

        log_likelihood += weight_prior(self.parameters) / self.num_datapoints

        return -log_likelihood
#  }}} Loss #


class BayesianNeuralNetwork(object):
    def __init__(self, network_architecture=network,
                 normalize_input=True, normalize_output=True,
                 logging=True, loss=NegativeLogLikelihood, num_steps=13000,
                 burn_in_steps=3000, keep_every=100, optimizer=Adam):
        self.num_steps = num_steps
        self.num_burn_in_steps = burn_in_steps
        self.loss = loss
        self.keep_every = keep_every
        self.normalize_input = normalize_input
        self.normalize_output = normalize_output
        self.optimizer = optimizer

        self.logging = logging
        self.sampled_weights = []

    def _keep_sample(self, epoch: int):
        if epoch < self.num_burn_in_steps:
            return False
        sample_t = epoch - self.num_burn_in_steps
        return sample_t % self.keep_every == 0

    def _log_progress(self, epoch: int):
        return self.logging and (epoch % 100 == 0)

    @property
    def network_weights(self):
        return tuple(
            np.asarray(torch.tensor(parameter.data).numpy())
            for parameter in self.model.parameters()
        )

    @network_weights.setter
    def network_weights(self, weights):
        for parameter, sample in zip(self.model.parameters(), weights):
            parameter.copy_(torch.from_numpy(sample))

    def train(self, x_train, y_train):
        self.sampled_weights.clear()

        num_datapoints, input_dimensionality = x_train.shape

        x_train_ = np.asarray(x_train)

        if self.normalize_input:
            x_train_, self.x_mean, self.x_std = zero_mean_unit_var_normalization(x_train)

        y_train_ = np.asarray(y_train)

        if self.normalize_output:
            y_train_, self.y_mean, self.y_std = zero_mean_unit_var_normalization(y_train)

        train_loader = InfiniteDataLoader(
            data_utils.TensorDataset(torch.Tensor(x_train_), torch.Tensor(y_train_))
        )

        self.model = network(input_dimensionality=input_dimensionality)

        optimizer = self.optimizer(self.model.parameters())

        # XXX: Use fancy bar here

        for epoch, (x_batch, y_batch) in islice(enumerate(train_loader), self.num_steps):
            # loss = torch.nn.MSELoss()(model(x_batch)[:, 0], y_batch)
            loss = self.loss(self.model.parameters(), num_datapoints)(input=self.model(x_batch), target=y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if self._log_progress:
                print("Epoch: {}, Loss: {}".format(epoch, loss.item()))

            if self._keep_sample(epoch):
                self.sampled_weights.append(self.network_weights)

        self.is_trained = True

    def predict(self, x_test):
        x_test_ = np.asarray(x_test)

        if self.normalize_input:
            x_test_, *_ = zero_mean_unit_var_normalization(x_test, self.x_mean, self.x_std)

        def network_predict(x_test_, weights):
            with torch.no_grad():
                self.network_weights = weights
                return self.model(torch.from_numpy(x_test_).float()).numpy()[:, 0]

        network_outputs = [
            network_predict(x_test_, weights=weights)
            for weights in self.sampled_weights
        ]

        mean_prediction = np.mean(network_outputs, axis=0)
        variance_prediction = np.mean((network_outputs - mean_prediction) ** 2, axis=0)

        if self.normalize_output:
            mean_prediction = zero_mean_unit_var_unnormalization(
                mean_prediction, self.y_mean, self.y_std
            )
            variance_prediction *= self.y_std ** 2

        return mean_prediction, variance_prediction


input_dimensionality, num_datapoints = 1, 100
x_train = np.array([
    np.random.uniform(np.zeros(1), np.ones(1), input_dimensionality)
    for _ in range(num_datapoints)
])
y_train = np.sinc(x_train * 10 - 5).sum(axis=1)

x_test = np.linspace(0, 1, 100)[:, None]
y_test = np.sinc(x_test * 10 - 5).sum(axis=1)

# XXX: Try to get it running in this script, then copy it to bayesian_neural_network.py
optimizer = Adam
bnn = BayesianNeuralNetwork(optimizer=optimizer)
bnn.train(x_train, y_train)


prediction, variance_prediction = bnn.predict(x_test)
prediction_std = np.sqrt(variance_prediction)

plt.grid()

plt.plot(x_test[:, 0], y_test, label="true", color="black")
plt.plot(x_train[:, 0], y_train, "ro")

plt.plot(x_test[:, 0], prediction, label=optimizer.__name__, color="blue")
plt.fill_between(x_test[:, 0], prediction + prediction_std, prediction - prediction_std, alpha=0.2, color="indianred")
plt.legend()
plt.show()
