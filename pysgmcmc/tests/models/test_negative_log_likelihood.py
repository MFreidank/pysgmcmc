import numpy as np
import pytest
import torch
from torch.utils import data as data_utils


try:
    import theano as _
except ImportError:
    THEANO_INSTALLED = False
else:
    THEANO_INSTALLED = True

try:
    import lasagne as _
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

    from pysgmcmc.models.bayesian_neural_network import (
        negative_log_likelihood, default_network
    )

from pysgmcmc.diagnostics.objective_functions import sinc
from pysgmcmc.tests.utils import init_random_uniform

@pytest.mark.skipif(
    # not (THEANO_INSTALLED and LASAGNE_INSTALLED),
    True,
    reason="Packages 'theano' and 'lasagne' required!"
)
def test_nll():
    bnn = BayesianNeuralNetwork(normalize_input=False, normalize_output=False)


    num_datapoints = 100
    X = init_random_uniform(
        lower=np.zeros(1), upper=np.ones(1), n_points=num_datapoints
    )

    y = sinc(X)

    _, input_dimensionality = X.shape
    net = bnn.get_net(n_inputs=input_dimensionality)

    reference_nll = bnn.negativ_log_likelihood(
        net, X=X, y=y, n_examples=num_datapoints,
        variance_prior=ReferenceLogVariancePrior(1e-6, 0.01),
        weight_prior=ReferenceWeightPrior(alpha=1., beta=1.)
    )[0].eval()

    train_dataset = data_utils.TensorDataset(
        torch.Tensor(X), torch.Tensor(y)
    )

    train_loader = data_utils.DataLoader(
        dataset=train_dataset, batch_size=num_datapoints, shuffle=False
    )

    batch_x, batch_y = next(iter(train_loader))

    model = default_network(input_dimensionality=input_dimensionality)
    y_pred = model(batch_x)
    f = negative_log_likelihood(model, num_datapoints=num_datapoints)


    nll = negative_log_likelihood(model, num_datapoints=num_datapoints)(
        y_true=batch_y, y_pred=y_pred
    ).detach().numpy()

    assert np.allclose(reference_nll, nll)
