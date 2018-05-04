# vim:foldmethod=marker
import numpy as np
import pytest


try:
    import theano as _  # noqa
except ImportError:
    THEANO_INSTALLED = False
else:
    THEANO_INSTALLED = True

try:
    import lasagne as _  # noqa
except ImportError:
    LASAGNE_INSTALLED = False
else:
    LASAGNE_INSTALLED = True

if THEANO_INSTALLED and LASAGNE_INSTALLED:
    from pysgmcmc.tests.models.reference_implementation.bnn import (
        BayesianNeuralNetwork as ReferenceBayesianNeuralNetwork
    )

    from pysgmcmc.models.bayesian_neural_network import BayesianNeuralNetwork
    from pysgmcmc.models.architectures import simple_tanh_network

from pysgmcmc.diagnostics.objective_functions import sinc
from pysgmcmc.tests.utils import init_random_uniform


@pytest.mark.skipif(
    not (THEANO_INSTALLED and LASAGNE_INSTALLED),
    reason="Packages 'theano' and 'lasagne' required!"
)
def test_predict():
    X = init_random_uniform(
        lower=np.zeros(1), upper=np.ones(1), n_points=np.random.randint(1, 200),
    )

    y = sinc(X)

    x_test = np.linspace(0, 1, 100)[:, None]

    reference_bnn = ReferenceBayesianNeuralNetwork(
        normalize_input=True, normalize_output=True,
        burn_in=1000, sample_steps=20, n_nets=20
    )
    reference_bnn.train(X, y)
    reference_predictions = reference_bnn.predict(np.asarray(x_test))

    samples = reference_bnn.samples

    bnn = BayesianNeuralNetwork(normalize_input=True, normalize_output=True)
    bnn.model = simple_tanh_network(input_dimensionality=X.shape[1])


    #  Copy sampled weights of `reference_bnn` over. {{{ #
    bnn.sampled_weights = []

    for sample in samples:
        current_sample = []
        for parameter, sample_dimension in zip(bnn.network_weights, sample):
            current_sample.append(sample_dimension.T)
        bnn.sampled_weights.append(np.asarray(current_sample))

    # Simply claim that we are trained.
    bnn.is_trained = True

    # Copy data normalizers over.
    bnn.x_mean, bnn.x_std = reference_bnn.x_mean, reference_bnn.x_std
    bnn.y_mean, bnn.y_std = reference_bnn.y_mean, reference_bnn.y_std
    #  }}} Copy sampled weights of `reference_bnn` over. #

    predictions = bnn.predict(np.asarray(x_test))

    assert np.allclose(reference_predictions, predictions, atol=1e-2)
