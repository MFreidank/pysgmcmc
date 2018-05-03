import torch


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
