from pysgmcmc.optimizers.sghmc import SGHMC
from pysgmcmc.optimizers.sgld import SGLD


def get_optimizer(optimizer_cls, parameters, num_datapoints, **optimizer_kwargs):
    if optimizer_cls is SGHMC:
        return SGHMC(parameters, scale_grad=num_datapoints, **optimizer_kwargs)
    return optimizer_cls(parameters, **optimizer_kwargs)
