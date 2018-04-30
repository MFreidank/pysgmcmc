from os.path import dirname, join as path_join

import numpy as np
import pytest
import torch
try:
    import theano
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
    from pysgmcmc.tests.models.reference_implementation.bnn import (
        get_default_net as reference_default_network
    )


from pysgmcmc.models.bayesian_neural_network import (
    default_network
)
from pysgmcmc.tests.utils import init_random_uniform


def predict_theano(network, weights, x_train):
    x_theano = T.matrix()
    lasagne.layers.set_all_param_values(network, weights)

    theano_predict = theano.function(
        [x_theano],
        lasagne.layers.get_output(network, x_theano)
    )
    return theano_predict(x_train)


def predict_pytorch(network, weights, x_train):
    x_torch = torch.from_numpy(x_train).float()

    for parameter, sample in zip(network.parameters(), weights):
        print(parameter.shape, sample.shape)
    for parameter, sample in zip(network.parameters(), weights):
        parameter.data.copy_(torch.from_numpy(sample).view(parameter.shape))

    for parameter, sample in zip(network.parameters(), weights):
        # Assert that weights are assigned properly; this works!
        assert np.allclose(parameter.view(sample.shape).detach().numpy(), sample)

    return network(x_torch).detach().numpy()


@pytest.mark.skipif(
    not(THEANO_INSTALLED and LASAGNE_INSTALLED),
    reason="Packages 'theano' and 'lasagne' required!"
)
def test_simple_architecture():
    x_train = init_random_uniform(
        lower=np.zeros(1), upper=np.ones(1), n_points=100,
        rng=np.random.RandomState(1)
    )
    _, input_dimensionality = x_train.shape
    reference_network = reference_default_network(n_inputs=input_dimensionality)

    # load inputs
    weights = np.load(
        path_join(dirname(__file__), "test_data", "weight_inputs.npy")
    )

    theano_predictions = predict_theano(reference_network, weights, x_train)

    network = default_network(input_dimensionality=input_dimensionality)

    predictions = predict_pytorch(network, weights, x_train)

    # Assert that predictions are the same for equal weights and equal input data
    # TODO: This fails, forward passes seem to have differing behavior!

    assert np.allclose(predictions, theano_predictions)
