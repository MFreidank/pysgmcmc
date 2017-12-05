from keras import backend as K
from pysgmcmc.metaoptimization.environment import SamplerEnv
from pysgmcmc.diagnostics.objective_functions import (
    banana_log_likelihood, to_negative_log_likelihood
)
from pysgmcmc.samplers.optimizer_sghmc import SGHMCSampler


class SGHMCBananaEnv(SamplerEnv):
    """docstring for ClassName"""
    def __init__(self):
        def param_factory():
            return [K.variable(0.0), K.variable(6.0)]
        super().__init__(
            param_factory=param_factory,
            loss_function=to_negative_log_likelihood(banana_log_likelihood),
            sampler_constructor=SGHMCSampler,
            initial_stepsize=1e-3
        )
