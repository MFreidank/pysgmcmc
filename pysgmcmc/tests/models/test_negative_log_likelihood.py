from os.path import dirname, join as path_join

import numpy as np
import pytest
import torch


try:
    import theano as _  # noqa
    import theano.tensor as T
except ImportError:
    THEANO_INSTALLED = False
else:
    THEANO_INSTALLED = True

try:
    import lasagne
except ImportError:
    LASAGNE_INSTALLED = False
else:
    LASAGNE_INSTALLED = True

if THEANO_INSTALLED and LASAGNE_INSTALLED:
    from pysgmcmc.tests.models.reference_implementation.priors import (
        ReferenceWeightPrior, ReferenceLogVariancePrior
    )
    from pysgmcmc.tests.models.reference_implementation.bnn import (
        BayesianNeuralNetwork
    )

    from pysgmcmc.models.losses import NegativeLogLikelihood
    from pysgmcmc.models.architectures import simple_tanh_network

from pysgmcmc.diagnostics.objective_functions import sinc
from pysgmcmc.tests.utils import init_random_uniform


def predict_pytorch(network, weights, x_train):
    x_torch = torch.from_numpy(x_train).float()

    for parameter, sample in zip(network.parameters(), weights):
        with torch.no_grad():
            parameter.copy_(torch.from_numpy(sample.T))

    return network(x_torch)


@pytest.mark.skipif(
    not (THEANO_INSTALLED and LASAGNE_INSTALLED),
    reason="Packages 'theano' and 'lasagne' required!"
)
def test_nll():
    bnn = BayesianNeuralNetwork(normalize_input=False, normalize_output=False)

    num_datapoints = 100
    X = init_random_uniform(
        lower=np.zeros(1), upper=np.ones(1), n_points=num_datapoints,
        rng=np.random.RandomState(1)
    )

    y = sinc(X)

    _, input_dimensionality = X.shape  # noqa
    net = bnn.get_net(n_inputs=input_dimensionality)

    weights = np.load(
        path_join(dirname(__file__), "test_data", "weight_inputs.npy")
    )
    lasagne.layers.set_all_param_values(net, weights)

    reference_nll = bnn.negativ_log_likelihood(
        net, X=X, y=y, n_examples=num_datapoints,
        variance_prior=ReferenceLogVariancePrior(1e-6, 0.01),
        weight_prior=ReferenceWeightPrior(alpha=1., beta=1.)
    )[0]

    grads_theano = np.asarray([
        grad.eval()
        for grad in T.grad(reference_nll, lasagne.layers.get_all_params(net))
    ])

    train_y = torch.from_numpy(y).float()

    model = simple_tanh_network(input_dimensionality=input_dimensionality)
    y_pred = predict_pytorch(network=model, weights=weights, x_train=X)

    nll = NegativeLogLikelihood(tuple(model.parameters()), num_datapoints=num_datapoints)(
        input=y_pred, target=train_y
    )

    nll.backward()
    grads = np.asarray([
        parameter.grad.numpy() for parameter in model.parameters()
    ])

    assert np.allclose(reference_nll.eval(), nll.detach().numpy())

    for grad_theano, grad_pytorch in zip(grads_theano, grads):
        assert np.allclose(grad_theano.T, grad_pytorch)
