# vim:foldmethod=marker
import logging
# import typing
from itertools import islice

import numpy as np
import torch
import torch.nn as nn
from torch.utils import data as data_utils


#  Helpers {{{ #

def default_network(input_dimensionality: int, seed: int=None):
    # TODO: Use seed
    class AppendLayer(nn.Module):
        def __init__(self, output_features=1, bias=True):
            super().__init__()
            if bias:
                self.bias = nn.Parameter(torch.Tensor(output_features))
            else:
                # You should always register all possible parameters, but the
                # optional ones can be None if you want.
                self.register_parameter('bias', None)

        def forward(self, x):
            return torch.cat((x, self.bias * torch.ones_like(x)), 1)

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

def log_variance_prior(log_variance,
                       mean: float=1e-6,
                       variance: float=0.01):

    return torch.mean(
        torch.sum(
            -torch.pow(log_variance - torch.log(torch.Tensor([mean])), 2) /
            (2. * variance)
        ) - 0.5 * torch.log(torch.Tensor([variance]))
    )


def weight_prior(parameters,
                 wdecay: float=1.):
    log_likelihood = 0.
    num_parameters = 0

    for parameter in parameters:
        param = torch.from_numpy(parameter.detach().numpy())
        log_likelihood += torch.sum(-wdecay * 0.5 * torch.pow(param, 2))
        num_parameters += np.prod(parameter.shape)

    # XXX: Might need casting for n_parameters
    return safe_division(log_likelihood, num_parameters).float()


def negative_log_likelihood(model,
                            num_datapoints: int,
                            batch_size: int=20,
                            log_variance_prior=log_variance_prior,
                            weight_prior=weight_prior):
    def loss_function(y_true, y_pred):
        f_mean = torch.reshape(y_pred[:, 0], shape=(-1, 1))
        mean_squared_error = torch.pow(y_true - f_mean, 2)
        f_log_var = y_pred[:, 1]
        f_var_inv = 1. / torch.exp(f_log_var)

        log_likelihood = torch.sum(
            torch.sum(-mean_squared_error * 0.5 * f_var_inv - 0.5 * f_log_var, dim=1)
        ) / batch_size

        log_variance_prior_log_likelihood = log_variance_prior(f_log_var)
        log_likelihood += (
            log_variance_prior_log_likelihood / (num_datapoints + 1e-16)
        )

        weight_prior_log_likelihood = weight_prior(model.parameters())

        log_likelihood += weight_prior_log_likelihood / (num_datapoints + 1e-16)

        return -log_likelihood
    return loss_function

#  }}} Loss #


class BayesianNeuralNetwork(object):
    def __init__(self, network_architecture=default_network,
                 normalize_input=True, normalize_output=True,
                 loss=negative_log_likelihood,
                 n_steps=50000, burn_in_steps=3000,
                 keep_every=100, num_nets=100, batch_size=20,
                 seed: int=None,
                 optimizer=torch.optim.SGD, **optimizer_kwargs) -> None:

        assert n_steps > burn_in_steps
        self.burn_in_steps = burn_in_steps
        self.n_steps = n_steps - self.burn_in_steps

        assert batch_size > 0
        self.batch_size = batch_size

        assert keep_every > 0
        self.keep_every = keep_every

        assert num_nets > 0
        self.num_nets = num_nets

        self.n_steps = min(
            self.n_steps, self.keep_every * self.num_nets
        )
        logging.info(
            "Performing '{}' iterations in total.".format(
                self.n_steps + self.burn_in_steps
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

        self.sampled_weights = []  # type: typing.List[typing.List[np.ndarray]]

        self.optimizer_kwargs = optimizer_kwargs

    def train(self, x_train: np.ndarray, y_train: np.ndarray):
        self.sampled_weights.clear()

        self.x_train, self.y_train = x_train, y_train

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

        train_set = data_utils.TensorDataset(
            torch.Tensor(self.x_train), torch.Tensor(self.y_train)
        )

        train_loader = data_utils.DataLoader(
            dataset=train_set,
            batch_size=self.batch_size,
            shuffle=False
        )

        loss_function = self.loss_function(
            self.model, batch_size=self.batch_size, num_datapoints=num_datapoints
        )

        for batch_index, (x_batch, y_batch) in islice(enumerate(train_loader), self.n_steps):
            loss = loss_function(y_pred=self.model(x_batch), y_true=y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        self.sampled_weights.append(
            [np.array(parameter.detach().numpy()) for parameter in self.model.parameters()]
        )

        self.is_trained = True

    #  Predict {{{ #

    def predict(self, x_test: np.ndarray, return_individual_predictions: bool=False):
        assert self.is_trained
        assert isinstance(return_individual_predictions, bool)

        if self.normalize_input:
            x_test_, _, _ = zero_mean_unit_var_normalization(
                x_test, self.x_mean, self.x_std
            )

        # XXX: Loader for test data

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

    #  Incumbent {{{ #
    @property
    def incumbent(self):
        if self.normalize_input:
            x = zero_mean_unit_var_unnormalization(
                self.x_train, self.x_mean, self.x_std
            )
            mean, _ = self.predict(x)
        else:
            mean, _ = self.predict(self.x_train)

        incumbent_index = np.argmin(self.y_train)
        incumbent = self.x_train[incumbent_index]
        incumbent_value = mean[incumbent_index]

        if self.normalize_input:
            incumbent = zero_mean_unit_var_unnormalization(
                incumbent, self.x_mean, self.x_std
            )

        if self.normalize_output:
            incumbent_value = zero_mean_unit_var_unnormalization(
                incumbent_value, self.y_mean, self.y_std
            )

        return incumbent_value
    #  }}} Incumbent #
