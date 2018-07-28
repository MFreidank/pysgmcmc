import typing

import torch
from torch.nn.modules.loss import _Loss, _assert_no_grad

from pysgmcmc.torch_typing import (
    VariancePrior, WeightPrior, Predictions, Targets,
    TorchLoss, TorchLossFunction
)
from pysgmcmc.models.priors import log_variance_prior, weight_prior


class NegativeLogLikelihood(_Loss):
    """ Impementation of BNN negative log likelihood for regression problems. """

    name = "NLL"

    def __init__(self, parameters: typing.Iterable[torch.Tensor],
                 num_datapoints: int,
                 variance_prior: VariancePrior=log_variance_prior,
                 weight_prior: WeightPrior=weight_prior,
                 size_average: bool=True, reduce: bool=False) -> None:
        """ Instantiate a loss object for given network `parameters`.
            Requires `num_datapoints` of the entire regression dataset
            for proper scaling.

        Parameters
        ----------
        parameters : typing.Iterable[torch.Tensor]
            Pytorch variables of BNN parameters.
        num_datapoints : int
            Total number of datapoints of the entire regression dataset to process.
        variance_prior : pysgmcmc.torch_typing.VariancePrior, optional
            Prior for BNN variance. Default: `pysgmcmc.models.priors.log_variance_prior`.
        weight_prior : pysgmcmc.torch_typing.WeightPrior, optional
            Prior for BNN weights. Default: `pysgmcmc.models.priors.weight_prior`.

        """
        assert size_average and not reduce

        super().__init__()
        self.parameters = tuple(parameters)
        self.num_datapoints = num_datapoints

        self.log_variance_prior = log_variance_prior
        self.weight_prior = weight_prior

    def forward(self, input: Predictions, target: Targets) -> torch.Tensor:
        """ Compute NLL for 2d-network predictions `input` and (batch) labels `target`.

        Parameters
        ----------
        input : pysgmcmc.torch_typing.Predictions
            Network predictions.
        target : pysgmcmc.torch_typing.Targets
            Labels for each datapoint in the current batch.

        Returns
        ----------
        nll: torch.Tensor
            Scalar value.
            NLL of BNN predictions given as `input` with respect to labels `target`.
        """
        _assert_no_grad(target)

        batch_size = input.size(0)

        prediction_mean = input[:, 0].view((-1, 1))
        log_prediction_variance = input[:, 1].view((-1, 1))
        prediction_variance_inverse = 1. / (torch.exp(log_prediction_variance) + 1e-16)

        mean_squared_error = (target.view(-1, 1) - prediction_mean) ** 2

        log_likelihood = torch.sum(torch.sum(-mean_squared_error * (0.5 * prediction_variance_inverse) - 0.5 * log_prediction_variance, dim=1))

        log_likelihood = log_likelihood / batch_size

        log_likelihood += (
            self.log_variance_prior(log_prediction_variance) / self.num_datapoints
        )

        log_likelihood += self.weight_prior(self.parameters) / self.num_datapoints

        return -log_likelihood


def get_loss(loss_cls: TorchLoss, **loss_kwargs) -> TorchLossFunction:
    """ Wrapper to use `NegativeLogLikelihood` interchangeably with other pytorch losses.
        `loss_kwargs` is expected to be a dict with key `parameters` mapped to
        network parameters and key `num_datapoints` mapped to an integer
        representing the amount of datapoints in the entire regression dataset.

    Parameters
    ----------
    loss_cls : pysgmcmc.torch_typing.TorchLoss
        Class type of a loss, e.g. `pysgmcmc.models.losses.NegativeLogLikelihood`.
    loss_kwargs : dict
        Keyword arguments to be passed to `loss_cls`.
        Must contain keys `parameters` for BNN parameters and `num_datapoints`
        for the amount of datapoints in the entire regression dataset.

    Returns
    ----------
    loss_instance: pysgmcmc.torch_typing.TorchLossFunction
        Instance of `loss_cls`.

    """
    if loss_cls is NegativeLogLikelihood:
        return NegativeLogLikelihood(**loss_kwargs)
    loss_kwargs.pop("parameters")
    loss_kwargs.pop("num_datapoints")
    return loss_cls(**loss_kwargs)


def to_bayesian_loss(torch_loss):
    """ Wrapper to make pytorch losses compatible with our BNN predictions.
        BNN predictions are 2-d, with the second dimension representing model variance.
        This wrapper essentially passes only the network mean prediction into `torch_loss`, which allows us to evaluate `torch_loss` on our network predictions as normally.

    Parameters
    ----------
    torch_loss: pysgmcmc.torch_typing.TochLoss
        Class type of a pytorch loss to evaluate on our BNN, e.g. `torch.nn.MSELoss`.

    Returns
    ----------
    torch_loss_changed:
        Class type that behaves like `torch_loss` but assumes inputs coming from a BNN.
        It will evaluate `torch_loss` on the BNN predictions first dimension,
        on the mean prediction, only.

    """
    class BayesianLoss(torch_loss):
        def forward(self, input, target):
            return super().forward(input=input[:, 0], target=target)
    BayesianLoss.__name__ = torch_loss.__name__
    return BayesianLoss
