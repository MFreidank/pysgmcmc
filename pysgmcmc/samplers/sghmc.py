from pysgmcmc.samplers.mixin import SamplerMixin
from pysgmcmc.optimizers.sghmc import SGHMC as SGHMCOptimizer


class SGHMC(SamplerMixin, SGHMCOptimizer):
    """ SGHMC Sampler. """
