import sys
from os.path import dirname, realpath, join as path_join

import numpy as np
from scipy.stats import norm
import logging

sys.path.insert(0, path_join(dirname(realpath(__file__)), "..", ".."))


def ei(x, model):
    # Implementation of ei as acquisition function.
    mean, variance = model.predict(x)
    standard_deviation = np.sqrt(variance)

    if (standard_deviation == 0).any():
        return np.array([0])
    else:
        z = (model.incumbent - mean) / standard_deviation

        r = standard_deviation * (z * norm.cdf(z) + norm.pdf(z))
        return r


def random_points(num_points, parameter_bounds, seed=None):
    # draw random uniform num_points
    lowerbound, upperbound = parameter_bounds

    random_state = np.random.RandomState(seed)

    n_dims, *_ = lowerbound.shape

    x = np.array([
        random_state.uniform(lowerbound, upperbound, n_dims)
        for _ in range(num_points)
    ])
    return x


def scipy_maximizer(model, parameter_bounds, acquisition_function=ei, restarts=10, seed=None, **scipy_args):
    from scipydirect import minimize

    return minimize(
        func=lambda x: -acquisition_function(model=model, x=x.reshape((1, x.shape[0]))),
        bounds=parameter_bounds
    )


def random_sample_maximizer(model, parameter_bounds, acquisition_function=ei, num_samples=100, seed=None):
    # Maximize acquisition function by drawing random samples from the configuration space.

    points = random_points(num_points=num_samples, seed=seed, parameter_bounds=parameter_bounds)
    n_dims, *_ = points[0].shape

    acquisition_values = np.asarray([
        acquisition_function(model=model, x=np.reshape(point, (1, n_dims))) for point in points
    ])
    return points[acquisition_values.argmax()]


def initial_design(num_points, parameter_bounds, objective_function, seed=None):
    # Construct an initial design for bayesian optimization
    x = random_points(num_points=num_points, parameter_bounds=parameter_bounds, seed=seed)
    y = []

    incumbents, incumbent_values = [], []
    current_incumbent_value = float("inf")

    for index, configuration in enumerate(x):
        y_configuration = objective_function(configuration)
        y.append(y_configuration)

        if y_configuration < current_incumbent_value:
            incumbents.append(configuration.tolist())
            incumbent_values.append(y_configuration)
            current_incumbent_value = y_configuration

    return x, np.asarray(y), incumbents, incumbent_values


def bayesian_optimization(objective_function, parameter_bounds, model_function,
                          num_iterations=30, train_every=1,
                          acquisition_function=ei,
                          acquisition_maximizer=scipy_maximizer,
                          num_initial_points=3, seed=None, **model_args):
    """ Run a bayesian optimization loop to minimize a given `objective_function`.
        Parameter bounds specifies the space of all valid inputs
        of `objective_function` to try.
        `model_function` is a callable that takes `model_args` and returns a
        (machine learning) model that must have methods "train, predict" and
        a property "incumbent" that returns the best solution that
        this model has seen so far.

    Parameters
    ----------
    objective_function : callable
        Target function of this optimization, should be *minimized*
    parameter_bounds : typing.Tuple[np.ndarray, np.ndarray]
        Tuple of lower and upper bounds for each parameter.
        `lowerbound[i], upperbound[i]` correspond to lower and upper bound
        of one parameter of `objective_function`.
    model_function : callable
    **model_args : kwargs
    num_iterations : int, optional
    train_every : int, optional
    acquisition_function : callable, optional
    acquisition_maximizer : callable, optional
    num_initial_points : int, optional
    seed : typing.Union[None, int], optional

    Returns
    ----------
    TODO

    Examples
    ----------
    TODO

    """
    # run bayesian optimization

    model = model_function(**model_args)

    x, y, incumbents, incumbent_values = initial_design(
        objective_function=objective_function, num_points=num_initial_points,
        parameter_bounds=parameter_bounds,
    )

    current_incumbent_value = incumbent_values[-1]

    for iteration in range(num_initial_points, num_iterations):

        if iteration % train_every == 0:
            model.train(x, y)

        configuration = acquisition_maximizer(
            acquisition_function=ei, model=model,
            parameter_bounds=parameter_bounds,
        )

        y_configuration = objective_function(configuration)

        x = np.append(x, configuration[None, :], axis=0)
        y = np.append(y, y_configuration)

        if y_configuration < current_incumbent_value:
            incumbents.append(configuration.tolist())
            incumbent_values.append(y_configuration)
            current_incumbent_value = y_configuration

        logging.info(
            "Current best configuration '{configuration}'"
            " with function value '{value}'".format(configuration=incumbents[-1], value=current_incumbent_value)
        )

    return {
        "best": {"configuration": incumbents[-1],
                 "objective value": incumbent_values[-1]},
        "all": [
            {"configuration": incumbent, "objective value": incumbent_value}
            for incumbent, incumbent_value in zip(incumbents, incumbent_values)
        ]
    }
