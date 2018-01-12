import numpy as np
from pysgmcmc.models.objective_functions import sinc


def init_random_uniform(lower, upper, n_points, rng=None):
    if rng is None:
        rng = np.random.RandomState(np.random.randint(0, 10000))

    n_dims = lower.shape[0]

    return np.array(
        [rng.uniform(lower, upper, n_dims) for _ in range(n_points)]
    )


OBJECTIVES = {
    "sinc": (
        sinc, lambda n_datapoints: init_random_uniform(
            lower=np.zeros(1), upper=np.ones(1), n_points=n_datapoints
        )
    )
}
