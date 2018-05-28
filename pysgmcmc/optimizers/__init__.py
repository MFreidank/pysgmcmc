from pysgmcmc.optimizers.sghmc import SGHMC


def get_optimizer(optimizer_cls, parameters, num_datapoints, **optimizer_kwargs):
    if optimizer_cls is SGHMC:
        return SGHMC(parameters, scale_grad=num_datapoints, **optimizer_kwargs)
    return optimizer_cls(parameters, **optimizer_kwargs)
