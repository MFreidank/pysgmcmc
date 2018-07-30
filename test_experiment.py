#!/usr/bin/python3
# -*- coding: iso-8859-15 -*-
from os.path import join as path_join

import numpy as np

from keras.models import Sequential
from keras.losses import cosine_proximity
from keras.layers import Dense, Embedding, Reshape, Merge
from keras.losses import kullback_leibler_divergence, cosine_proximity
from keras.activations import tanh
import pandas as pd

from pysgmcmc.models.network_architectures import simple_tanh_network

from pysgmcmc.optimizers.sghmchd4 import SGHMCHD
from pysgmcmc.optimizers.sghmc import SGHMC
from pysgmcmc.models.bayesian_neural_network import (
    BayesianNeuralNetwork
)
from pysgmcmc.models.network_architectures import simple_embedding_network

import pandas as pd


def architecture(features_dataframe, categorical_columns, seed=None):
    layers = []

    num_continuous_columns = len(
        set(features_dataframe.columns).difference(categorical_columns)
    )

    def embed_input(column_name):
        num_unique_categories = features_dataframe[column_name].nunique()
        embedding_size = int(min(np.ceil(num_unique_categories / 2), 50))
        vocabulary_size = num_unique_categories + 1
        return Sequential([
            Embedding(vocabulary_size, embedding_size, input_length=1),
            Reshape(target_shape=(embedding_size,))
        ])

    layers = [
        embed_input(column) for column in categorical_columns
    ]

    layers.append(
        Sequential([Dense(units=10, input_dim=num_continuous_columns, activation=tanh)])
    )
    return Sequential([
        Merge(layers, mode="concat"),
        *simple_tanh_network(seed=seed, units=50)
    ])


def batch_generator_embedding(x_train, y_train, batch_size, seed=None):
    n_examples, *_ = y_train.shape

    if seed is None:
        seed = np.random.randint(1, 100000)

    rng = np.random.RandomState()
    rng.seed(seed)

    # Check if we have enough data points to form a minibatch
    # otherwise set the batchsize equal to the number of input points
    initial_batch_size = batch_size
    batch_size = min(initial_batch_size, n_examples)

    while True:
        # `np.random.randint` is end-exclusive
        # => for n_examples == batch_size, start == 0 holds

        start = rng.randint(0, (n_examples - batch_size + 1))

        # When using embedding layers, batches must be lists.
        minibatch_x = [
            x_train[dim][start:start + batch_size, ...]
            for dim in range(len(x_train))
        ]

        minibatch_y = np.squeeze(y_train)[start:start + batch_size, None]

        yield (minibatch_x, minibatch_y)


def categories(df, column):
    index = {
        name: index for index, name in enumerate(df[column].unique())
    }
    df[column] = [index[name] for name in df[column]]
    return df

DATASET_FILENAME = lambda x: path_join("./datasets/", x)
data_seed = 1
test_split = 0.1
sampler = SGHMCHD
stepsize = 1e-2
num_steps = 15000
burn_in_steps = 5000
num_nets = 100
batch_size = 32

categorical_columns = ("cylinders", "model year", "origin", "car name")
label_column = "mpg"

df = pd.read_csv(DATASET_FILENAME("auto-mpg.data_cleaned"), sep="\t")

for column in categorical_columns:
    df = categories(df, column)

# reorder-columns; categorical columns first
df = df[["cylinders", "model year", "origin", "car name", "displacement", "horsepower", "weight", "acceleration", "mpg"]]

test_data = df.sample(frac=test_split, random_state=data_seed)
train_data = df[~df.index.isin(test_data.index)]

train_features = train_data.drop(label_column, axis=1)
continuous_columns = tuple(
    column for column in train_features.columns
    if column not in categorical_columns
)

x_train = [train_features[column].as_matrix() for column in categorical_columns]
x_train.append(train_features[list(continuous_columns)].as_matrix())
y_train = np.squeeze(train_data[label_column].as_matrix())
num_datapoints, = y_train.shape

network_factory = lambda _, seed: architecture(
    features_dataframe=df.drop(label_column, axis=1),
    categorical_columns=categorical_columns,
    seed=data_seed
)

model = BayesianNeuralNetwork(
    n_steps=num_steps,
    num_nets=num_nets,
    burn_in_steps=burn_in_steps,
    batch_size=batch_size,
    batch_generator=batch_generator_embedding,
    network_architecture=network_factory,
    optimizer=sampler,
    hyperloss=lambda y_true, y_pred: cosine_proximity(
        y_true=y_true, y_pred=y_pred[:, 0]
    ),
    # Normalization?
    normalize_input=False,
    normalize_output=True,
    learning_rate=stepsize,
)

model.train(x_train, np.asarray(y_train))
print(model.model.summary())
from keras.utils import plot_model
plot_model(model.model, "./model.png")
