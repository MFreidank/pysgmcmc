import typing

import numpy as np

from keras import backend as K
from keras.models import Sequential
from keras.layers import Concatenate, Layer, Dense, Embedding, Reshape, Merge
from keras.activations import tanh
from keras.initializers import Constant, VarianceScaling


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


def simple_tanh_network(input_dimension: int=None,
                        seed: int=None, units=50) -> Sequential:
    if input_dimension is None:
        # NOTE: Allow using this network inside `Sequential` constructors
        # to construct other models that use it as building block.
        return [
            Dense(
                units=units, activation=tanh,
                kernel_initializer=VarianceScaling(seed=seed)
            ),
            Dense(
                units=units, activation=tanh,
                kernel_initializer=VarianceScaling(seed=seed),
            ),
            Dense(
                units=units, activation=tanh,
                kernel_initializer=VarianceScaling(seed=seed),
            ),
            Dense(units=1, kernel_initializer=VarianceScaling(seed=seed)),
            AppendLayer(b=np.log(1e-3))
        ]

    else:
        return Sequential([
            Dense(
                units=units, input_dim=input_dimension, activation=tanh,
                kernel_initializer=VarianceScaling(seed=seed)
            ),
            Dense(
                units=units, activation=tanh,
                kernel_initializer=VarianceScaling(seed=seed),
            ),
            Dense(
                units=units, activation=tanh,
                kernel_initializer=VarianceScaling(seed=seed),
            ),
            Dense(units=1, kernel_initializer=VarianceScaling(seed=seed)),
            AppendLayer(b=np.log(1e-3))
        ])


def simple_embedding_network(dataframe, categorical_columns, label_column, seed: int=None):
    print(dataframe.columns)
    features_dataframe = dataframe.drop(label_column, axis=1)

    continuous_columns = tuple(
        column_name for column_name in features_dataframe.columns
        if column_name not in categorical_columns
    )

    def embed_input(column_name):
        num_unique_categories = features_dataframe[column_name].nunique()
        # embedding_size = int(min(np.ceil(num_unique_categories / 2), 50))
        embedding_size = 5
        vocabulary_size = num_unique_categories + 1
        return Sequential([
            Embedding(vocabulary_size, embedding_size, input_length=1),
            Reshape(target_shape=(embedding_size,))
        ])

    input_layers = [
        embed_input(categorical_variable) for categorical_variable in categorical_columns
    ]

    # XXX: Handle numerical columns here

    return Sequential([
        Merge(input_layers, mode="concat"),
        *simple_tanh_network()
    ])
