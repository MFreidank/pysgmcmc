from pysgmcmc.optimizers.sghmc4 import SGHMC as optimizer_cls
from pysgmcmc.samplers.base_classes import sampler_from_optimizer

# Automatic conversion of one of our optimizers to a corresponding sampler class.
SGHMCSampler = sampler_from_optimizer(optimizer_cls)
