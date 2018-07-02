# vim:foldmethod=marker
import logging
import typing
import numpy as np
import keras
from keras import backend as K
from keras.models import Sequential
from keras.layers import Concatenate, Layer, Dense
from keras.callbacks import LambdaCallback
from keras.activations import tanh
from keras.initializers import Constant, VarianceScaling
from keras.losses import kullback_leibler_divergence, cosine_proximity

from pysgmcmc.data_batches import generate_batches
from pysgmcmc.models.base_model import (
    zero_mean_unit_var_normalization,
    zero_mean_unit_var_unnormalization
)
from pysgmcmc.optimizers import get_optimizer
from pysgmcmc.optimizers.sghmc import SGHMC
from pysgmcmc.custom_typing import (
    KerasPrior, KerasLossFunction, KerasModelLoss,
    KerasNetworkFactory, KerasOptimizer,
    KerasTensor, KerasVariable
)

#  Utils {{{ #

#  Priors {{{ #


def log_variance_prior(log_variance: KerasTensor,
                       mean: float=1e-6,
                       variance: float=0.01) -> KerasTensor:

    with K.name_scope(log_variance_prior.__name__):
        return K.mean(K.sum(
            -K.square(log_variance - K.log(K.constant(mean, dtype=K.floatx()))) /
            (2 * K.constant(variance, dtype=K.floatx())) -
            0.5 * K.log(K.constant(variance, dtype=K.floatx())), axis=1
        ))


def weight_prior(parameters: typing.List[KerasVariable],
                 wdecay: float=1.) -> KerasTensor:
    with K.name_scope(weight_prior.__name__):
        log_likelihood = K.constant(0., dtype=K.floatx())
        n_parameters = K.constant(0, dtype="int64")

        for parameter in parameters:
            log_likelihood += K.sum(-wdecay * 0.5 * K.square(parameter))
            n_parameters += K.cast(K.prod(parameter.shape), "int64")

        return log_likelihood / K.cast(n_parameters, K.floatx())


#  }}} Priors #

#  Network Architecture {{{ #

def default_network(input_dimension: int,
                    seed: int=None) -> Sequential:
    class AppendLayer(Layer):
        def __init__(self, b, **kwargs):
            self.b = b
            self.concat = Concatenate(axis=1)
            super().__init__(**kwargs)

        def build(self, input_shape):
            self.bias = self.add_weight(
                name="bias",
                shape=(1, 1),
                initializer=Constant(value=self.b)
            )
            super().build(input_shape)

        def call(self, x):
            return self.concat([x, self.bias * K.ones_like(x)])

        def compute_output_shape(self,
                                 input_shape: typing.Tuple[int, int])-> typing.Tuple[int, int]:
            return (input_shape[0], input_shape[1] * 2)

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
        AppendLayer(b=np.log(1e-3))
    ])

    return model

#  }}} Network Architecture #

#  Loss function (Negative Log Likelihood) {{{ #


def negative_log_likelihood(model: Sequential,
                            n_datapoints: int,
                            hyperloss=None,
                            log_variance_prior: KerasPrior=log_variance_prior,
                            weight_prior: KerasPrior=weight_prior) -> KerasLossFunction:

    def loss_function(y_true: KerasTensor, y_pred: KerasTensor):
        with K.name_scope("negative_log_likelihood"):
            batch_size = K.cast(K.shape(y_true)[0], K.floatx())

            f_mean = K.reshape(y_pred[:, 0], shape=(-1, 1))

            f_log_var = K.reshape(y_pred[:, 1], shape=(-1, 1))

            f_var_inv = 1. / (K.exp(f_log_var) + K.epsilon())

            mean_squared_error = K.square(y_true - f_mean)

            if hyperloss:
                hyperloss_tensor = hyperloss(y_true=y_true, y_pred=y_pred)
                for param in model.trainable_weights:
                    param.hypergradient = K.gradients(hyperloss_tensor, param)

            log_likelihood = K.sum(
                K.sum(
                    -mean_squared_error * (0.5 * f_var_inv) - 0.5 * f_log_var,
                    axis=1
                )
            )

            log_likelihood /= batch_size

            log_likelihood += log_variance_prior(f_log_var) / n_datapoints

            log_likelihood += weight_prior(model.trainable_weights) / n_datapoints

            return -log_likelihood
    return loss_function

#  }}} Loss function (Negative Log Likelihood) #

#  }}} Utils #


class BayesianNeuralNetwork(object):
    def __init__(self,
                 network_architecture: KerasNetworkFactory=default_network,
                 train_callbacks: typing.List[keras.callbacks.Callback]=None,
                 loss_function: KerasModelLoss=negative_log_likelihood,
                 metrics: typing.Tuple[str, ...]=("mse", "mae",),
                 normalize_input: bool=True, normalize_output: bool=True,
                 n_steps: int=50000, burn_in_steps: int=3000,
                 # hyperloss=lambda y_true, y_pred: kullback_leibler_divergence(y_true=y_true, y_pred=y_pred[:, 0]),
                 hyperloss=lambda y_true, y_pred: cosine_proximity(y_true=y_true, y_pred=y_pred[:, 0]),
                 keep_every: int=100,
                 n_nets: int=100,
                 batch_size: int=20,
                 optimizer: KerasOptimizer=SGHMC,
                 seed: int=None,
                 **optimizer_hyperparameters) -> None:

        assert n_steps > burn_in_steps
        self.burn_in_steps = burn_in_steps
        self.n_steps = n_steps - self.burn_in_steps

        assert batch_size > 0
        self.batch_size = batch_size

        assert keep_every > 0
        self.keep_every = keep_every

        assert n_nets > 0
        self.n_nets = n_nets

        self.n_steps = min(
            self.n_steps, self.keep_every * self.n_nets
        )
        logging.info(
            "Performing '{}' iterations in total.".format(
                self.n_steps + self.burn_in_steps
            )
        )

        assert isinstance(normalize_input, bool)
        self.normalize_input = normalize_input

        assert isinstance(normalize_output, bool)
        self.normalize_output = normalize_output

        assert train_callbacks is None or hasattr(train_callbacks, "__len__")

        if train_callbacks is None:
            self.train_callbacks = []  # type: typing.List[keras.callbacks.Callback]
        else:
            self.train_callbacks = list(train_callbacks)

        self.train_callbacks.append(
            LambdaCallback(on_batch_end=self._extract_samples)
        )

        self.seed = seed

        self.hyperloss = hyperloss

        self.network_architecture = network_architecture
        self.loss_function = loss_function

        self.optimizer = optimizer
        self.optimizer_hyperparameters = optimizer_hyperparameters

        self.metrics = metrics

        self.sampled_weights = []  # type: typing.List[typing.List[np.ndarray]]

    def _keep_sample(self, epoch: int) -> bool:
        """ Check if we should store a sample extracted at a given `epoch`.
            Samples are stored after burn-in and only every `self.keep_every` steps.

        Parameters
        ----------
        epoch: int
            Current training epoch.

        Returns
        ----------
        should_keep: bool
            `True` if and only if a `epoch` fits our criteria for sampling.

        Examples
        ----------

        During burn-in, we do not keep any sampled networks:

        >>> bnn = BayesianNeuralNetwork(burn_in_steps=3000, keep_every=10)
        >>> epoch = 0
        >>> bnn.burn_in_steps > epoch
        True
        >>> bnn._keep_sample(0)
        False

        After burn-in, we keep every `bnn.keep_every`th network:

        """
        if epoch < self.burn_in_steps:
            return False
        sample_t = epoch - self.burn_in_steps
        return (sample_t % self.keep_every) == 0

    def _extract_samples(self,
                         epoch: int,
                         logs: typing.Dict[str, typing.Any]) -> None:
        """ Extract current sampled network weights at a given `epoch` and store them internally.

        Parameters
        ----------
        epoch: int
            Keras training epoch.
        logs: typing.Dict[str, typing.Any]
            Keras logs recorded at `epoch`.

        """
        if self._keep_sample(epoch):
            weight_values = K.batch_get_value(self.model.trainable_weights)
            self.sampled_weights.append(weight_values)

    def log_learning_rate(self, epoch: int, logs: typing.Dict[str, typing.Any]):
        if hasattr(self.optimizer, "lr"):
            logging.debug(" Learning rate: {}".format(K.batch_get_value([self.optimizer.lr])))
        else:
            logging.debug(
                " Cannot print learning rate of optimizer,"
                " ensure the corresponding parameter is named 'lr'."
            )

    def train(self, x_train: np.ndarray, y_train: np.ndarray, *args, **kwargs) -> None:
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

        with K.name_scope("neural_network"):
            self.model = self.network_architecture(
                input_dimension, self.seed
            )

        assert callable(self.optimizer)


        # NOTE: Do not reuse the same optimizer instance multiple times -- use a new one on each call to `train`,
        # otherwise `n_datapoints` does not get updated and scale_grad introduces larger and larger errors
        self.optimizer_instance = get_optimizer(
            optimizer_name=self.optimizer.__name__,
            n_datapoints=n_datapoints,
            batch_size=self.batch_size,
            burn_in_steps=self.burn_in_steps,
            learning_rate=self.optimizer_hyperparameters["learning_rate"],
            seed=self.seed,
        )

        self.model.compile(
            optimizer=self.optimizer_instance,
            loss=self.loss_function(
                self.model, n_datapoints, hyperloss=self.hyperloss
            ),
            metrics=list(self.metrics)
        )

        self.model.fit_generator(
            generate_batches(
                self.x_train, self.y_train,
                batch_size=self.batch_size, seed=self.seed
            ),
            epochs=1,
            steps_per_epoch=self.n_steps + self.burn_in_steps,
            callbacks=self.train_callbacks + [LambdaCallback(on_batch_end=self.log_learning_rate)],
        )

        self.is_trained = True

    def predict(self,
                x_test: np.ndarray,
                return_individual_predictions: bool=False) -> typing.Tuple[np.ndarray, np.ndarray]:

        # XXX: implement return_individual_predictions
        assert self.is_trained
        assert isinstance(return_individual_predictions, bool)

        if self.normalize_input:
            x_test_, _, _ = zero_mean_unit_var_normalization(
                x_test, self.x_mean, self.x_std
            )

        def neural_network_predict(weights: np.ndarray):
            self.model.set_weights(weights)
            return self.model.predict(x_test_)[:, 0]

        network_outputs = np.asarray([
            neural_network_predict(weights) for weights in self.sampled_weights
        ])

        mean_prediction = np.mean(network_outputs, axis=0)
        # Total variance
        variance_prediction = np.mean(
            (network_outputs - mean_prediction) ** 2, axis=0
        )

        if self.normalize_output:
            mean_prediction = zero_mean_unit_var_unnormalization(
                mean_prediction, self.y_mean, self.y_std
            )
            variance_prediction *= self.y_std ** 2

        return mean_prediction, variance_prediction

    @property
    def incumbent(self):
        """ Returns the best observed point and its function value.

        Returns
        ----------
        incumbent: ndarray (D,)
            Current Incumbent.
        incumbent_value: ndarray (N,)
            Observed value of the current incumbent.
        """
        if self.normalize_input:
            x = zero_mean_unit_var_unnormalization(
                self.x_train, self.x_mean, self.x_std
            )
            mean, _ = self.predict(x)
        else:
            mean, _ = self.predict(self.x_train)

        best_idx = np.argmin(self.y_train)
        incumbent = self.x_train[best_idx]
        incumbent_value = mean[best_idx]

        if self.normalize_input:
            incumbent = zero_mean_unit_var_unnormalization(
                incumbent, self.x_mean, self.x_std
            )

        if self.normalize_output:
            incumbent_value = zero_mean_unit_var_unnormalization(
                incumbent_value, self.y_mean, self.y_std
            )

        return incumbent, incumbent_value

    def get_incumbent(self):
        return self.incumbent
