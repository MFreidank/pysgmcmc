import tensorflow as tf
from pysgmcmc.models.bayesian_neural_network import BayesianNeuralNetwork


def data_for(function, n_train_points):
    # construct train/test values for `function` (like in sinc)
    raise ValueError("Reading input dimensionality from docstrings not yet supported!")


def _sampler_test(objective_function, n_train_points, passing_criterion,
                  sampling_method, **sampler_args):

    X_train, y_train, X_test, y_test = data_for(
        objective_function, n_train_points=n_train_points
    )

    with tf.Session() as session:
        model = BayesianNeuralNetwork(
            sampling_method=sampling_method, sampler_args=sampler_args,
            session=session
        )

        model.fit(X_train, y_train)

        prediction_mean, prediction_variance = model.predict(X_test)

    # XXX Check that test passes by evaluating the passing criterion
    passing_criterion(prediction_mean, y_test)
