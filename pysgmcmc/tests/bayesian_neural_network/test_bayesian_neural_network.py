from pysgmcmc.tests.bnn_testing import sampler_test

from itertools import product

from pysgmcmc.sampling import Sampler
from pysgmcmc.diagnostics.objective_functions import sinc

import pytest

try:
    from hypothesis import given
    from hypothesis.strategies import integers
except ImportError:
    hypothesis_installed = False
else:
    hypothesis_installed = True


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
        }  # XXX: Add arguments for each sampler if necessary
    },
)


@pytest.mark.skipif(
    not hypothesis_installed, reason="Package 'hypothesis' not installed!"
)
@given(integers(min_value=1, max_value=2 ** 32 - 1))
def test_samplers(seed):
    for sampler, objective_function in product(Sampler, objective_functions):
        sampler_test(
            objective_function["function"],
            passing_criterion=objective_function["passing_criterion"],
            n_train_points=objective_function["n_train_points"],
            function_domain=objective_function["domain"],
            seed=seed,
            sampling_method=sampler,
            sampler_args=objective_function["sampler_args"].get(sampler, dict())
        )
