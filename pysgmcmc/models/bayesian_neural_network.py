# vim:foldmethod=marker
import logging
# import typing
from itertools import islice

import numpy as np
import torch
import torch.nn as nn
from torch.utils import data as data_utils
from tqdm import tqdm

from pysgmcmc.data.utils import InfiniteDataLoader


#  Helpers {{{ #

def default_network(input_dimensionality: int, seed: int=None):
    # TODO: Use seed
    class AppendLayer(nn.Module):
        def __init__(self, output_features=1, bias=True):
            super().__init__()
            if bias:
                self.bias = nn.Parameter(torch.Tensor(output_features, 1))
            else:
                self.register_parameter('bias', None)

        def forward(self, x):
            return torch.cat((x, self.bias * torch.ones_like(x)), dim=1)

    def init_weights(module):
        if type(module) == AppendLayer:
            nn.init.constant_(module.bias, val=np.log(1e-3))
        elif type(module) == nn.Linear:
            nn.init.kaiming_normal_(module.weight, a=1.0)
            nn.init.constant_(module.bias, val=0.0)

    return nn.Sequential(
        nn.Linear(input_dimensionality, 50),
        nn.Tanh(),
        nn.Linear(50, 50),
        nn.Tanh(),
        nn.Linear(50, 50),
        nn.Tanh(),
        nn.Linear(50, 1),
        AppendLayer()
    ).apply(init_weights)


def safe_division(x, y, small_constant=1e-16):
    """ Computes `x / y` after adding a small appropriately signed constant to `y`.
        Adding a small constant avoids division-by-zero artefacts that may
        occur due to precision errors.

    Parameters
    ----------
    x: np.ndarray
        Left-side operand of division.
    y: np.ndarray
        Right-side operand of division.
    small_constant: float, optional
        Small constant to add to/subtract from `y` before computing `x / y`.
        Defaults to `1e-16`.

    Returns
    ----------
    division_result : np.ndarray
        Result of `x / y` after adding a small appropriately signed constant
        to `y` to avoid division by zero.

    Examples
    ----------

    Will safely avoid divisions-by-zero under normal circumstances:

    >>> import numpy as np
    >>> x = np.asarray([1.0])
    >>> inf_tensor = x / 0.0  # will produce "inf" due to division-by-zero
    >>> bool(np.isinf(inf_tensor))
    True
    >>> z = safe_division(x, 0., small_constant=1e-16)  # will avoid division-by-zero
    >>> bool(np.isinf(z))
    False

    To see that simply adding a positive constant may fail, consider the
    following example. Note that this function handles such corner cases correctly:

    >>> import numpy as np
    >>> x, y = np.asarray([1.0]), np.asarray([-1e-16])
    >>> small_constant = 1e-16
    >>> inf_tensor = x / (y + small_constant)  # simply adding constant exhibits division-by-zero
    >>> bool(np.isinf(inf_tensor))
    True
    >>> z = safe_division(x, y, small_constant=1e-16)  # will avoid division-by-zero
    >>> bool(np.isinf(z))
    False

    """
    if (np.asarray(y) == 0).all():
        return np.true_divide(x, small_constant)
    return np.true_divide(x, np.sign(y) * small_constant + y)


def zero_mean_unit_var_normalization(X, mean=None, std=None):
    if mean is None:
        mean = np.mean(X, axis=0)
    if std is None:
        std = np.std(X, axis=0)

    X_normalized = safe_division(X - mean, std)

    return X_normalized, mean, std


def zero_mean_unit_var_unnormalization(X_normalized, mean, std):
    return X_normalized * std + mean
#  }}} Helpers #


#  Loss {{{ #

#  Prios {{{ #

def log_variance_prior(log_variance,
                       mean: float=1e-6,
                       variance: float=0.01):
    return torch.mean(
        torch.sum(
            ((-(log_variance - torch.log(torch.Tensor([mean]))) ** 2) /
             ((2. * variance))) - 0.5 * torch.log(torch.Tensor([variance])),
            dim=1
        )
    )


def weight_prior(parameters,
                 wdecay: float=1.):
    log_likelihood = 0.
    num_parameters = 0

    for parameter in parameters:
        log_likelihood += torch.sum(-wdecay * 0.5 * (parameter ** 2))
        num_parameters += np.prod(parameter.shape)

    return log_likelihood / (num_parameters + 1e-6)
#  }}} Prios #


def negative_log_likelihood(model,
                            num_datapoints: int,
                            log_variance_prior=log_variance_prior,
                            weight_prior=weight_prior):
    def loss_function(y_true, y_pred, batch_size=20):
        f_mean = y_pred[:, 0].view(-1, 1)
        mean_squared_error = (y_true - f_mean) ** 2
        f_log_var = y_pred[:, 1].view(-1, 1)
        f_var_inv = 1. / (torch.exp(f_log_var) + 1e-16)

        log_likelihood = torch.sum(
            torch.sum(
                -mean_squared_error * 0.5 * f_var_inv - 0.5 * f_log_var,
                dim=1
            )
        )

        log_likelihood = log_likelihood / batch_size

        log_variance_prior_log_likelihood = log_variance_prior(f_log_var)

        log_likelihood += (
            log_variance_prior_log_likelihood / num_datapoints
        )

        weight_prior_log_likelihood = weight_prior(model.parameters())

        log_likelihood += weight_prior_log_likelihood / num_datapoints

        return -log_likelihood
    return loss_function

#  }}} Loss #


class BayesianNeuralNetwork(object):
    def __init__(self, network_architecture=default_network,
                 normalize_input=True, normalize_output=True,
                 loss=negative_log_likelihood,
                 metrics=(nn.MSELoss(),),
                 num_steps=50000, burn_in_steps=3000,
                 keep_every=100, num_nets=100, batch_size=20,
                 seed: int=None,
                 progress=True,
                 optimizer=torch.optim.SGD, **optimizer_kwargs) -> None:

        assert num_steps > burn_in_steps
        self.burn_in_steps = burn_in_steps
        self.num_steps = num_steps - self.burn_in_steps

        assert batch_size > 0
        self.batch_size = batch_size

        assert keep_every > 0
        self.keep_every = keep_every

        assert num_nets > 0
        self.num_nets = num_nets

        self.num_steps = min(
            self.num_steps, self.keep_every * self.num_nets
        )
        logging.info(
            "Performing '{}' iterations in total.".format(
                self.num_steps + self.burn_in_steps
            )
        )

        assert isinstance(normalize_input, bool)
        self.normalize_input = normalize_input

        assert isinstance(normalize_output, bool)
        self.normalize_output = normalize_output

        self.seed = seed

        self.network_architecture = network_architecture

        self.optimizer = optimizer
        self.loss_function = loss

        self.metric_functions = list(metrics)

        self.sampled_weights = []  # type: typing.List[typing.List[np.ndarray]]

        self.optimizer_kwargs = optimizer_kwargs

        self.progress = progress

    def _keep_sample(self, epoch: int) -> bool:
        if epoch < self.burn_in_steps:
            return False
        sample_step = epoch - self.burn_in_steps
        return (sample_step % self.keep_every) == 0

    def train(self, x_train: np.ndarray, y_train: np.ndarray):
        self.sampled_weights.clear()

        self.x_train, self.y_train = np.asarray(x_train), np.asarray(y_train)

        if self.normalize_input:
            self.x_train, self.x_mean, self.x_std = zero_mean_unit_var_normalization(
                self.x_train
            )

        if self.normalize_output:
            self.y_train, self.y_mean, self.y_std = zero_mean_unit_var_normalization(
                self.y_train
            )

        num_datapoints, input_dimensionality = self.x_train.shape

        self.model = self.network_architecture(
            input_dimensionality=input_dimensionality,
            seed=self.seed
        )

        optimizer = self.optimizer(
            self.model.parameters(), **self.optimizer_kwargs

        )
        train_dataset = data_utils.TensorDataset(
            torch.Tensor(self.x_train), torch.Tensor(self.y_train)
        )

        train_loader = InfiniteDataLoader(
            dataset=train_dataset, batch_size=self.batch_size, shuffle=True
        )

        loss_function = self.loss_function(
            self.model, num_datapoints=num_datapoints
        )

        if self.progress:
            progress_bar = tqdm(
                islice(enumerate(train_loader), self.num_steps + self.burn_in_steps),
                total=self.num_steps + self.burn_in_steps,
                bar_format="{n_fmt}/{total_fmt}[{bar}] - {remaining} - {postfix}"
            )
        else:
            progress_bar = islice(train_loader, self.num_steps + self.burn_in_steps)

        for epoch, (x_batch, y_batch) in progress_bar:
            # properly handle partial batches with less than `self.batch_size`
            # datapoints.
            batch_size, *_ = x_batch.shape
            batch_prediction = self.model(x_batch)
            loss = loss_function(
                y_pred=self.model(x_batch), y_true=y_batch, batch_size=batch_size
            )
            metric_values = [
                metric_function(input=batch_prediction[:, 0], target=y_batch)
                for metric_function in self.metric_functions
            ]

            def get_name(metric):
                try:
                    name = metric.__name__
                except AttributeError:
                    return metric.__class__.__name__
                else:
                    if metric == negative_log_likelihood:
                        return "NLL"
                    return name

            metric_names = [
                get_name(metric)
                for metric in self.metric_functions + [self.loss_function]
            ]

            if self.progress and epoch % 100 == 0:
                progress_bar.set_postfix_str(" - ".join([
                    "{name}: {value}".format(name=name, value=value.detach().numpy())
                    for name, value in zip(
                        metric_names, metric_values + [loss]
                    )
                ]))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if self._keep_sample(epoch):
                sample = tuple(
                    np.array(parameter.detach().numpy())
                    for parameter in self.model.parameters()
                )
                self.sampled_weights.append(sample)

        self.is_trained = True

    #  Predict {{{ #

    def predict(self, x_test: np.ndarray, return_individual_predictions: bool=False):
        assert self.is_trained
        assert isinstance(return_individual_predictions, bool)

        if self.normalize_input:
            x_test_, _, _ = zero_mean_unit_var_normalization(
                x_test, self.x_mean, self.x_std
            )

        def network_predict(weights, test_data):
            for parameter, sample in zip(self.model.parameters(), weights):
                parameter.data.copy_(torch.from_numpy(sample))

            return self.model(test_data).detach().numpy()[:, 0]

        network_outputs = [
            network_predict(weights=sample, test_data=torch.from_numpy(x_test_).float())
            for sample in self.sampled_weights
        ]
        prediction_mean = np.mean(network_outputs, axis=0)

        prediction_variance = np.mean(
            (network_outputs - prediction_mean) ** 2, axis=0
        )

        if self.normalize_output:
            prediction_mean = zero_mean_unit_var_unnormalization(
                prediction_mean, self.y_mean, self.y_std
            )
            # TODO: Why does this look like this actually?
            prediction_variance *= self.y_std ** 2

        return prediction_mean, prediction_variance
    #  }}} Predict #
