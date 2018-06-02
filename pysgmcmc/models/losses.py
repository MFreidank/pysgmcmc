import torch
from torch.nn.modules.loss import _Loss, _assert_no_grad

from pysgmcmc.models.priors import log_variance_prior, weight_prior


class NegativeLogLikelihood(_Loss):

    name = "NLL"

    def __init__(self, parameters, num_datapoints, size_average=True, reduce=False):
        assert size_average and not reduce
        super().__init__()
        self.parameters = tuple(parameters)
        self.num_datapoints = num_datapoints

    def forward(self, input, target):
        _assert_no_grad(target)

        # print(input.size())
        batch_size = input.size(0)

        prediction_mean = torch.squeeze(input[:, 0])


        log_prediction_variance = input[:, 1].view(-1, 1)
        prediction_variance_inverse = 1. / (torch.exp(log_prediction_variance) + 1e-16)

        mean_squared_error = torch.pow(target - prediction_mean, 2)

        log_likelihood = (
            torch.sum(
                torch.sum(
                    -mean_squared_error * 0.5 * prediction_variance_inverse -
                    0.5 * log_prediction_variance,
                    dim=1
                )
            )
        )

        log_likelihood = log_likelihood / batch_size

        log_likelihood += (
            log_variance_prior(log_prediction_variance) / self.num_datapoints
        )

        log_likelihood += weight_prior(self.parameters) / self.num_datapoints

        return -log_likelihood


def get_loss(loss_cls, **loss_kwargs):
    if loss_cls is NegativeLogLikelihood:
        return NegativeLogLikelihood(**loss_kwargs)
    loss_kwargs.pop("parameters")
    loss_kwargs.pop("num_datapoints")
    return loss_cls(**loss_kwargs)


def to_bayesian_loss(torch_loss):
    class BayesianLoss(torch_loss):
        def forward(self, input, target):
            return super().forward(input=input[:, 0], target=target)
    BayesianLoss.__name__ = torch_loss.__name__
    return BayesianLoss
