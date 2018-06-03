# vim:foldmethod=marker
from collections import OrderedDict
import logging
import typing
from itertools import islice

import numpy as np
import torch
import torch.nn as nn
from torch.utils import data as data_utils

from pysgmcmc.models.architectures import simple_tanh_network
from pysgmcmc.data.utils import (
    infinite_dataloader,
    zero_mean_unit_var_normalization,
    zero_mean_unit_var_unnormalization
)
from pysgmcmc.optimizers import get_optimizer
from pysgmcmc.optimizers.sghmc import SGHMC
from pysgmcmc.models.losses import NegativeLogLikelihood, get_loss, to_bayesian_loss
from pysgmcmc.progressbar import TrainingProgressbar
from pysgmcmc.torch_utils import get_name


# TODO: Documentation, doctests and test for fit/predict


class BayesianNeuralNetwork(object):
    def __init__(self,
                 network_architecture=simple_tanh_network,
                 batch_size=20,
                 normalize_input: bool=True,
                 normalize_output: bool=True,
                 num_steps: int=13000,
                 burn_in_steps: int=3000,
                 keep_every: int=100,
                 loss=NegativeLogLikelihood,
                 metrics=(nn.MSELoss,),
                 logging_configuration: typing.Dict[str, typing.Any]={
                     "level": logging.INFO, "datefmt": "y/m/d"
                 },
                 optimizer=SGHMC,
                 **optimizer_kwargs)-> None:
        """TODO: Docstring for __init__.

        Parameters
        ----------
        network_architecture : TODO, optional
            Function mapping integer input dimensionality to an (initialized) `torch.nn.Module`.
            No explicit initialization is done in this class.
        normalize_input: bool, optional

        normalize_output: bool, optional

        num_steps: int : TODO, optional

        burn_in_steps: int : TODO, optional

        keep_every: int : TODO, optional

        loss : TODO, optional

        logging_configuration : typing.Dict[str, typing.Any], optional

        optimizer : TODO, optional
            Function that returns a `torch.optim.optimizer.Optimizer` subclass.

        """

        # TODO: pretty handling for burn_in_steps, num_steps etc?

        assert burn_in_steps >= 0
        assert keep_every >= 1
        assert num_steps > burn_in_steps + keep_every
        assert batch_size >= 1

        self.batch_size = batch_size

        self.num_steps = num_steps
        self.num_burn_in_steps = burn_in_steps
        self.loss = loss
        self.metrics = metrics
        self.keep_every = keep_every
        self.normalize_input = normalize_input
        self.normalize_output = normalize_output

        self.optimizer = optimizer
        self.optimizer_kwargs = optimizer_kwargs

        self.network_architecture = network_architecture

        self.sampled_weights = []  # type: typing.List[typing.Tuple[np.ndarray]]

        logging.basicConfig(**logging_configuration)

        try:
            self.debug_level = logging_configuration["level"]
        except KeyError:
            self.debug_level = logging.INFO
            logging.warn(
                "No level specified in 'logging_configuration' argument.\n"
                "Falling back to 'logging.INFO'."
            )

        self.debug_level = logging_configuration.get("level", logging.INFO)
        logging.info("Performing %d iterations" % (self.num_steps))

        self.use_progressbar = self.debug_level <= logging.INFO

    def _keep_sample(self, epoch: int) -> bool:
        if epoch < self.num_burn_in_steps:
            logging.debug("Skipping burn-in sample, epoch = %d" % epoch)
            return False
        sample_t = epoch - self.num_burn_in_steps
        return sample_t % self.keep_every == 0

    # XXX: Remove this by shifting these decisions into our progressbar.
    def _log_progress(self, epoch: int):
        return self.debug_level <= logging.INFO and (epoch % 100 == 0)

    @property
    def network_weights(self):
        return tuple(
            np.asarray(torch.tensor(parameter.data).numpy())
            for parameter in self.model.parameters()
        )

    # XXX: Why is this doctest never triggered/executed?
    @network_weights.setter
    def network_weights(self, weights: typing.List[np.ndarray]) -> None:
        """ Assign new `weights` to our neural networks parameters.

        Parameters
        ----------
        weights : typing.List[np.ndarray]
            List of weight values to assign.
            Individual list elements must have shapes that match
            the network parameters with the same index in `self.network_weights`.

        Examples
        ----------
        This serves as a handy bridge between our pytorch parameters
        and corresponding values for them represented as numpy arrays:

        >>> import numpy as np
        >>> bnn = BayesianNeuralNetwork()
        >>> input_dimensionality = 1
        >>> bnn.model = bnn.network_architecture(input_dimensionality)
        >>> dummy_weights = [np.random.rand(parameter.shape) for parameter in bnn.model.parameters()]
        >>> bnn.network_weights = dummy_weights
        >>> assert np.allclose(bnn.network_weights, dummy_weights)

        """
        logging.debug("Assigning new network weights: %s" % str(weights))
        for parameter, sample in zip(self.model.parameters(), weights):
            parameter.copy_(torch.from_numpy(sample))

    def train_on_dataset(self, torch_dataset):
        logging.debug("Training started.")

        logging.debug("Clearing list of sampled weights.")
        self.sampled_weights.clear()

        assert isinstance(torch_dataset, torch.utils.data.Dataset)

        train_loader = infinite_dataloader(
            data_utils.DataLoader(
                torch_dataset,
                batch_size=self.batch_size,
                shuffle=True
            )
        )

        num_datapoints = len(torch_dataset)

        try:
            architecture_name = self.network_architecture.__name__
        except AttributeError:
            architecture_name = str(self.network_architecture)

        logging.debug("Using network architecture: %s" % architecture_name)

        self.model = self.network_architecture()

        self.model.train()

        optimizer = get_optimizer(
            optimizer_cls=self.optimizer,
            parameters=self.model.parameters(),
            num_datapoints=num_datapoints,
            **self.optimizer_kwargs
        )

        # TODO: make more general
        loss_function = torch.nn.functional.nll_loss

        if self.use_progressbar:
            logging.info(
                "Progress bar enabled. To disable pass "
                "`logging_configuriation={level: debug.WARN}`."
            )

            losses = OrderedDict(((get_name(self.loss), loss_function),))

            batch_generator = TrainingProgressbar(
                iterable=islice(enumerate(train_loader), self.num_steps),
                update_every=1,
                losses=losses,
                total=self.num_steps,
                bar_format="{n_fmt}/{total_fmt}[{bar}] - {remaining} - {postfix}"
            )
        else:
            batch_generator = islice(enumerate(train_loader), self.num_steps)

        for epoch, (x_batch, y_batch) in batch_generator:
            optimizer.zero_grad()
            loss = loss_function(input=self.model(x_batch), target=y_batch)
            loss.backward()
            optimizer.step()

            if self.use_progressbar:
                predictions = self.model(x_batch)
                batch_generator.update(
                    predictions=predictions, y_batch=y_batch, epoch=epoch
                )

            if self._keep_sample(epoch):
                logging.debug("Recording sample, epoch = %d " % (epoch))
                weights = self.network_weights
                logging.debug("Sampled weights:\n%s" % str(weights))
                self.sampled_weights.append(weights)

        self.is_trained = True
        return self

    def train(self, x_train: np.ndarray, y_train: np.ndarray):
        logging.debug("Training started.")

        logging.debug("Clearing list of sampled weights.")
        self.sampled_weights.clear()

        num_datapoints, input_dimensionality = x_train.shape
        logging.debug(
            "Processing %d training datapoints "
            " with % dimensions each." % (num_datapoints, input_dimensionality)
        )

        x_train_ = np.asarray(x_train)

        if self.normalize_input:
            logging.debug(
                "Normalizing training datapoints to "
                " zero mean and unit variance."
            )
            x_train_, self.x_mean, self.x_std = zero_mean_unit_var_normalization(x_train)

        y_train_ = np.asarray(y_train)

        if self.normalize_output:
            logging.debug("Normalizing training labels to zero mean and unit variance.")
            y_train_, self.y_mean, self.y_std = zero_mean_unit_var_normalization(y_train)

        train_loader = infinite_dataloader(
            data_utils.DataLoader(
                data_utils.TensorDataset(
                    torch.from_numpy(x_train_).float(),
                    torch.from_numpy(y_train_).float()
                ),
                batch_size=self.batch_size
            )
        )

        try:
            architecture_name = self.network_architecture.__name__
        except AttributeError:
            architecture_name = str(self.network_architecture)
        logging.debug("Using network architecture: %s" % architecture_name)

        self.model = self.network_architecture(
            input_dimensionality=input_dimensionality
        )

        try:
            optimizer_name = self.optimizer.__name__
        except AttributeError:
            optimizer_name = str(self.optimizer)

        logging.debug("Using optimizer: %s" % optimizer_name)

        optimizer = get_optimizer(
            optimizer_cls=self.optimizer,
            parameters=self.model.parameters(),
            num_datapoints=num_datapoints,
            **self.optimizer_kwargs
        )

        loss_function = get_loss(
            self.loss, parameters=self.model.parameters(),
            num_datapoints=num_datapoints, size_average=True
        )

        if self.use_progressbar:
            logging.info(
                "Progress bar enabled. To disable pass "
                "`logging_configuriation={level: debug.WARN}`."
            )

            losses = OrderedDict(((get_name(self.loss), loss_function),))
            losses.update(
                (get_name(metric), to_bayesian_loss(metric)())
                for metric in self.metrics
            )

            batch_generator = TrainingProgressbar(
                iterable=islice(enumerate(train_loader), self.num_steps),
                losses=losses,
                total=self.num_steps,
                bar_format="{n_fmt}/{total_fmt}[{bar}] - {remaining} - {postfix}"
            )
        else:
            batch_generator = islice(enumerate(train_loader), self.num_steps)

        for epoch, (x_batch, y_batch) in batch_generator:
            optimizer.zero_grad()
            loss = loss_function(input=self.model(x_batch), target=y_batch)
            loss.backward()
            optimizer.step()

            if self.use_progressbar:
                predictions = self.model(x_batch)
                batch_generator.update(
                    predictions=predictions, y_batch=y_batch, epoch=epoch
                )

            if self._keep_sample(epoch):
                logging.debug("Recording sample, epoch = %d " % (epoch))
                weights = self.network_weights
                logging.debug("Sampled weights:\n%s" % str(weights))
                self.sampled_weights.append(weights)

        self.is_trained = True
        return self

    #  Predict {{{ #
    # XXX: implement return individual predictions
    def predict(self, x_test: np.ndarray, return_individual_predictions: bool=False):
        logging.debug("Predicting started.")
        x_test_ = np.asarray(x_test)

        logging.debug(
            "Processing %d test datapoints "
            " with %d dimensions each." % (x_test_.shape)
        )

        if self.normalize_input:
            logging.debug(
                "Normalizing test datapoints to "
                " zero mean and unit variance."
            )
            x_test_, *_ = zero_mean_unit_var_normalization(x_test, self.x_mean, self.x_std)

        def network_predict(x_test_, weights):
            logging.debug(
                "Predicting on data:\n%s Using weights:\n%s" % (
                    str(x_test_), str(weights)
                )
            )
            with torch.no_grad():
                self.network_weights = weights
                return self.model(torch.from_numpy(x_test_).float()).numpy()[:, 0]

        logging.debug("Predicting with %d networks." % len(self.sampled_weights))
        network_outputs = [
            network_predict(x_test_, weights=weights)
            for weights in self.sampled_weights
        ]

        mean_prediction = np.mean(network_outputs, axis=0)
        variance_prediction = np.mean((network_outputs - mean_prediction) ** 2, axis=0)

        if self.normalize_output:
            logging.debug("Unnormalizing predictions.")
            logging.debug(
                "Mean of network predictions "
                "before unnormalization:\n%s" % str(mean_prediction)
            )
            logging.debug(
                "Variance/Uncertainty of network predictions "
                "before unnormalization:\n%s" % str(variance_prediction)
            )

            mean_prediction = zero_mean_unit_var_unnormalization(
                mean_prediction, self.y_mean, self.y_std
            )
            variance_prediction *= self.y_std ** 2

            logging.debug(
                "Mean of network predictions "
                "after unnormalization:\n%s" % str(mean_prediction)
            )
            logging.debug(
                "Variance/Uncertainty of network predictions "
                "after unnormalization:\n%s" % str(variance_prediction)
            )

        return mean_prediction, variance_prediction
    #  }}} Predict #
