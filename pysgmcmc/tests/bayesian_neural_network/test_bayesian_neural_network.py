from pysgmcmc.tests.bnn_testing import sampler_test
from numpy.random import randint

from itertools import product

from pysgmcmc.sampling import Sampler
from pysgmcmc.diagnostics.objective_functions import sinc


def passing_criterion(mean_prediction, variance_prediction, labels):
    assert(True)


objective_functions = (
    {
        "function": sinc,
        "dimensionality": 1,
        "domain": (0., 1.),
        "n_train_points": 100,
        "passing_criterion": passing_criterion,
        "sampler_args": {
            "SGHMC": dict(),
            "SGLD": dict(),
        }  # Add arguments for each sampler if necessary
    },
)


def test_samplers():
    seed = randint(1, 1000)
    samplers = set(Sampler).difference(set((Sampler.RelativisticSGHMC,)))
    for sampler, objective_function in product(samplers, objective_functions):
        sampler_test(
            objective_function["function"],
            dimensionality=objective_function["dimensionality"],
            passing_criterion=objective_function["passing_criterion"],
            n_train_points=objective_function["n_train_points"],
            function_domain=objective_function["domain"],
            seed=seed,
            sampling_method=sampler,
            sampler_args=objective_function["sampler_args"].get(sampler, dict())
        )
