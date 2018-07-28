from os.path import dirname, join as path_join

import numpy as np
import pytest
import torch
try:
    import theano.tensor as T
    from pysgmcmc.tests.models.reference_implementation.priors import (
        ReferenceWeightPrior, ReferenceLogVariancePrior
    )
except ImportError:
    THEANO_INSTALLED = False
else:
    THEANO_INSTALLED = True

from pysgmcmc.models.priors import log_variance_prior, weight_prior


@pytest.mark.skipif(
    not THEANO_INSTALLED, reason="Package 'theano' not installed!"
)
def test_log_variance_prior():
    # intermediate value for f_log_var taken from running RoBo/models/bnn.py to fit sinc
    f_log_var = [[-11.25474104],
                 [-11.25474104],
                 [-11.25474104],
                 [-11.25474104],
                 [-11.25474104],
                 [-11.25474104],
                 [-11.25474104],
                 [-11.25474104],
                 [-11.25474104],
                 [-11.25474104],
                 [-11.25474104],
                 [-11.25474104],
                 [-11.25474104],
                 [-11.25474104],
                 [-11.25474104],
                 [-11.25474104],
                 [-11.25474104],
                 [-11.25474104],
                 [-11.25474104],
                 [-11.25474104]]

    theano_prior = ReferenceLogVariancePrior(1e-6, 0.01)

    theano_log_var = T.as_tensor_variable(f_log_var, name="v")

    reference_result = theano_prior.log_like(theano_log_var).eval()

    torch_result = log_variance_prior(torch.Tensor(f_log_var)).numpy()

    assert np.allclose(torch_result, reference_result)


@pytest.mark.skipif(
    not THEANO_INSTALLED, reason="Package 'theano' not installed!"
)
def test_weight_prior():

    # load inputs
    weight_inputs = np.load(
        path_join(dirname(__file__), "test_data", "weight_inputs.npy")
    )

    inputs = [
        torch.Tensor(p) for p in weight_inputs
    ]

    result = weight_prior(inputs).numpy()

    # load precomputed ground truth
    theano_prior = ReferenceWeightPrior(alpha=1., beta=1.)
    reference_result = theano_prior.log_like(weight_inputs).eval()

    assert np.allclose(result, reference_result)
