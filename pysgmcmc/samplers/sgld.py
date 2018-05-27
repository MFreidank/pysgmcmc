from pysgmcmc.samplers.mixin import SamplerMixin
from pysgmcmc.optimizers.sgld import SGLD as SGLDOptimizer


class SGLD(SamplerMixin, SGLDOptimizer):
    """ SGLD Sampler. """
