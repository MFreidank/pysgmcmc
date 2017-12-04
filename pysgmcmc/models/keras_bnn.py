import numpy as np
from keras import backend as K
from keras.models import Sequential
from keras.layers import Concatenate, Dense, Lambda
from keras.callbacks import LambdaCallback
from keras.activations import tanh
from keras.initializers import VarianceScaling
from pysgmcmc.diagnostics.metrics import metric_function
from pysgmcmc.keras_utils import safe_division
from pysgmcmc.data_batches import keras_generate_batches as generate_batches
from pysgmcmc.models.base_model import (
    zero_mean_unit_var_normalization,
    zero_mean_unit_var_unnormalization
)
from pysgmcmc.optimizers.sghmc import SGHMC


def log_variance_prior(log_variance, mean=1e-6, variance=0.01):
    return K.mean(
        K.sum(
            safe_division(
                -K.square(log_variance - K.log(mean)),
                (2. * variance)
            ) - 0.5 * K.log(variance), axis=1
        )
    )


def weight_prior(parameters, wdecay=1.):
    log_likelihood, n_parameters = 0., 0

    for parameter in parameters:
        log_likelihood += K.sum(-wdecay * 0.5 * K.square(parameter))
        n_parameters += K.prod(parameter.shape)

    return safe_division(log_likelihood, K.cast(n_parameters, K.floatx()))


def default_network(input_dimension, seed=None):
    concat = Concatenate(axis=1)

    model = Sequential([
        Dense(
            units=50, input_dim=input_dimension, activation=tanh,
            kernel_initializer=VarianceScaling(seed=seed)
        ),
        Dense(
            units=50, activation=tanh,
            kernel_initializer=VarianceScaling(seed=seed),
        ),
        Dense(
            units=50, activation=tanh,
            kernel_initializer=VarianceScaling(seed=seed),
        ),
        Dense(units=1, kernel_initializer=VarianceScaling(seed=seed)),
        Lambda(
            lambda dense_inputs: concat(
                [dense_inputs, K.log(1e-3) * K.ones_like(dense_inputs)]
            )
        )
    ])

    return model


def negative_log_likelihood(model, n_datapoints, batch_size=20):
    def cost_function(y_true, y_pred):
        f_mean = K.reshape(y_pred[:, 0], shape=(-1, 1))
        mean_squared_error = K.square(y_true - f_mean)

        f_log_var = K.reshape(y_pred[:, 1], shape=(-1, 1))

        f_var_inv = safe_division(1., (K.exp(f_log_var) + 1e-16))

        log_likelihood = K.sum(
            K.sum(
                -mean_squared_error * 0.5 * f_var_inv - 0.5 * f_log_var,
                axis=1
            )
        )

        log_likelihood = safe_division(log_likelihood, batch_size)

        log_variance_prior_log_likelihood = log_variance_prior(
            f_log_var
        )

        log_likelihood += safe_division(
            log_variance_prior_log_likelihood, n_datapoints
        )

        weight_prior_log_likelihood = weight_prior(
            model.trainable_weights
        )

        log_likelihood += safe_division(
            weight_prior_log_likelihood, n_datapoints
        )

        return -log_likelihood
    return cost_function


# XXX Introduce parameters to control how many samples to keep
class BayesianNeuralNetwork(object):
    def __init__(self, network_architecture=default_network,
                 train_callbacks=None,
                 loss_function=negative_log_likelihood,
                 metrics=("mse", "mae",),
                 normalize_input=True, normalize_output=True,
                 n_steps=10000, n_burn_in_steps=3000,
                 batch_size=20,
                 optimizer=SGHMC,
                 seed=None,
                 **optimizer_hyperparameters):

        assert n_steps > n_burn_in_steps
        self.n_steps = n_steps
        self.n_burn_in_steps = n_burn_in_steps

        assert batch_size > 0
        self.batch_size = batch_size

        assert isinstance(normalize_input, bool)
        self.normalize_input = normalize_input

        assert isinstance(normalize_output, bool)
        self.normalize_output = normalize_output

        assert train_callbacks is None or hasattr(train_callbacks, "__len__")

        if train_callbacks is None:
            self.train_callbacks = []
        else:
            self.train_callbacks = list(train_callbacks)

        self.train_callbacks.append(
            LambdaCallback(on_batch_end=self._extract_samples)
        )

        self.seed = seed

        self.network_architecture = network_architecture
        self.loss_function = loss_function

        self.optimizer = optimizer
        self.optimizer_hyperparameters = optimizer_hyperparameters

        self.metrics = {metric: metric_function(metric) for metric in metrics}

        self.sampled_weights = []

    def _extract_samples(self, epoch, logs):
        if epoch > self.n_burn_in_steps:
            weight_values = K.batch_get_value(self.model.trainable_weights)
            self.sampled_weights.append(weight_values)

    def train(self, x_train, y_train, validation_data=None):
        self.sampled_weights.clear()

        self.x_train, self.y_train = x_train, y_train

        if self.normalize_input:
            self.x_train, self.x_mean, self.x_std = zero_mean_unit_var_normalization(
                self.x_train
            )
        if self.normalize_output:
            self.y_train, self.y_mean, self.y_std = zero_mean_unit_var_normalization(
                self.y_train
            )

        n_datapoints, input_dimension = self.x_train.shape
        self.model = self.network_architecture(
            input_dimension=input_dimension, seed=self.seed
        )

        if callable(self.optimizer):
            self.optimizer = self.optimizer(
                seed=self.seed, burn_in_steps=self.n_burn_in_steps,
                scale_grad=n_datapoints,
                **self.optimizer_hyperparameters
            )

        self.model.compile(
            optimizer=self.optimizer,
            loss=self.loss_function(
                model=self.model,
                n_datapoints=n_datapoints,
                batch_size=self.batch_size
            ),
            metrics=list(self.metrics.values())
        )

        self.model.fit_generator(
            generate_batches(
                self.x_train, self.y_train,
                batch_size=self.batch_size, seed=self.seed
            ),
            epochs=1,
            steps_per_epoch=self.n_steps,
            callbacks=self.train_callbacks,
            # validation_data
        )

        self.is_trained = True

    def predict(self, x_test, return_individual_predictions=False):
        # XXX: return_individual_predictions
        assert self.is_trained

        x_test_ = x_test

        if self.normalize_input:
            x_test_, _, _ = zero_mean_unit_var_normalization(
                x_test, self.x_mean, self.x_std
            )

        def neural_network_predict(weights):
            self.model.set_weights(weights)
            return self.model.predict(x_test_)[:, 0]

        network_outputs = np.asarray([
            neural_network_predict(weights) for weights in self.sampled_weights
        ])

        mean_prediction = np.mean(network_outputs, axis=0)
        # Total variance
        variance_prediction = np.mean((network_outputs - mean_prediction) ** 2, axis=0)

        if self.normalize_output:
            mean_prediction = zero_mean_unit_var_unnormalization(
                mean_prediction, self.y_mean, self.y_std
            )
            variance_prediction *= self.y_std ** 2

        return mean_prediction, variance_prediction
