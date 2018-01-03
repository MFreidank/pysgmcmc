from pysgmcmc.optimizers.relativistic_sghmc import RelativisticSGHMC as optimizer_cls
from pysgmcmc.samplers.base_classes import sampler_from_optimizer

# Automatic conversion of one of our optimizers to a corresponding sampler class.
RelativisticSGHMCSampler = sampler_from_optimizer(optimizer_cls)
