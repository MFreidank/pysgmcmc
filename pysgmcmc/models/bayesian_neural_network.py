# vim:foldmethod=marker
import logging
import typing
from itertools import islice

import numpy as np
import torch
import torch.nn as nn
from torch.utils import data as data_utils
from torch.optim import Adam
from tqdm import tqdm

from pysgmcmc.models.architectures import simple_tanh_network
from pysgmcmc.data.utils import (
    InfiniteDataLoader,
    zero_mean_unit_var_normalization,
    zero_mean_unit_var_unnormalization
)
from pysgmcmc.models.losses import NegativeLogLikelihood

class BayesianNeuralNetwork(object):
    def __init__(self, network_architecture=simple_tanh_network,
                 normalize_input=True, normalize_output=True,
                 logging=True,
                 loss=NegativeLogLikelihood, num_steps=13000,
                 burn_in_steps=3000, keep_every=100, optimizer=Adam):

        # XXX: Use inspect.signature to compute smart behaviour for num_steps, burn_in_steps etc.

        self.num_steps = num_steps
        self.num_burn_in_steps = burn_in_steps
        self.loss = loss
        self.keep_every = keep_every
        self.normalize_input = normalize_input
        self.normalize_output = normalize_output
        self.optimizer = optimizer
        self.network_architecture = network_architecture

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

        self.model = self.network_architecture(input_dimensionality=input_dimensionality)

        optimizer = self.optimizer(self.model.parameters())

        if self.logging:
            batch_generator = tqdm(
                islice(enumerate(train_loader), self.num_steps),
                total=self.num_steps,
                bar_format="{n_fmt}/{total_fmt}[{bar}] - {remaining} - {postfix}"
            )
        else:
            batch_generator = islice(enumerate(train_loader), self.num_steps)

        for epoch, (x_batch, y_batch) in batch_generator:
            # loss = torch.nn.MSELoss()(model(x_batch)[:, 0], y_batch)
            loss = self.loss(self.model.parameters(), num_datapoints)(input=self.model(x_batch), target=y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if self._log_progress:
                metric_names = (
                    "NLL",
                )
                metric_values = (
                    loss.item(),
                )
                batch_generator.set_postfix_str(
                    " - ".join([
                        "{name}: {value}".format(name=name, value=value)
                        for name, value in zip(metric_names, metric_values)

                    ])
                )

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
