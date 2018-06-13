from pysgmcmc.optimizers.sghmc import SGHMC
from pysgmcmc.optimizers.relativistic_sghmc import RelativisticSGHMC
from pysgmcmc.optimizers.sghmchd_new import SGHMCHD
from pysgmcmc.optimizers.sgld import SGLD
from tensorflow.python.training import optimizer as tf_optimizer
from keras.optimizers import TFOptimizer


supported_optimizers = {
    "SGHMC": SGHMC,
    "RelativisticSGHMC": RelativisticSGHMC,
    "SGHMCHD": SGHMCHD,
    "SGLD": SGLD,
}


__all__ = sorted(supported_optimizers.keys())


def get_optimizer(optimizer_name, seed=None, n_datapoints=None, batch_size=None,
                  parameter_shapes=None, burn_in_steps=None,
                  learning_rate=None):

    assert optimizer_name in supported_optimizers

    optimizer_cls = supported_optimizers[optimizer_name]

    if optimizer_name == "SGLD":
        num_pseudo_batches = n_datapoints // batch_size
        optimizer = optimizer_cls(
            seed=seed,
            num_pseudo_batches=num_pseudo_batches,
            burn_in_steps=burn_in_steps,
            lr=learning_rate
        )
    elif optimizer_name == "SGHMC":
        optimizer = optimizer_cls(
            seed=seed, burn_in_steps=burn_in_steps,
            scale_grad=n_datapoints,
            # parameter_shapes=parameter_shapes,
            lr=learning_rate
        )
    elif optimizer_name == "SGHMCHD":
        optimizer = optimizer_cls(
            seed=seed, burn_in_steps=burn_in_steps,
            scale_grad=n_datapoints,
            lr=learning_rate
        )
    elif optimizer_name == "RelativisticSGHMC":
        optimizer = optimizer_cls(
            seed=seed, burn_in_steps=burn_in_steps,
            scale_grad=n_datapoints,
            lr=learning_rate
        )

    if isinstance(optimizer, tf_optimizer.Optimizer):
        optimizer = TFOptimizer(optimizer)
        optimizer.__class__.__name__ = optimizer_name

    return optimizer
