from pysgmcmc.samplers.mixin import SamplerMixin
from pysgmcmc.optimizers.sgld import SGLD as SGLDOptimizer


class SGLD(SamplerMixin, SGLDOptimizer):
    """ SGLD Sampler that performs his updates like `pysgmcmc.optimizers.sgld.SGLDOptimizer`. """
