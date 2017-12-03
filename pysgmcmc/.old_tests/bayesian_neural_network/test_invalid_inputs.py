import pytest
import tensorflow as tf
try:
    from hypothesis import given
    from hypothesis.strategies import (
        one_of, floats, complex_numbers, lists,
        sets, fractions, text, integers
    )
except ImportError:
    HYPOTHESIS_INSTALLED = False
else:
    HYPOTHESIS_INSTALLED = True

from pysgmcmc.models.bayesian_neural_network import BayesianNeuralNetwork


@given(
    one_of(
        floats(), complex_numbers(), lists(integers(), max_size=10),
        sets(integers(), max_size=10), fractions(), text(),
        integers(max_value=0)
    )
)
def test_invalid_n_nets(n_nets):
    with pytest.raises(AssertionError):
        graph = tf.Graph()
        with tf.Session(graph=graph) as session:
            BayesianNeuralNetwork(n_nets=n_nets, session=session)


@given(
    one_of(
        floats(), complex_numbers(), lists(integers(), max_size=10),
        sets(integers(), max_size=10), fractions(), text(),
        integers(max_value=0)
    )
)
def test_invalid_n_iters(n_iters):
    with pytest.raises(AssertionError):
        graph = tf.Graph()
        with tf.Session(graph=graph) as session:
            BayesianNeuralNetwork(n_iters=n_iters, session=session)


@given(
    one_of(
        floats(), complex_numbers(), lists(integers(), max_size=10),
        sets(integers(), max_size=10), fractions(), text(),
        integers(max_value=-1)
    )
)
def test_invalid_burn_in_steps(burn_in_steps):
    with pytest.raises(AssertionError):
        graph = tf.Graph()
        with tf.Session(graph=graph) as session:
            BayesianNeuralNetwork(burn_in_steps=burn_in_steps, session=session)


@given(
    one_of(
        floats(), complex_numbers(), lists(integers(), max_size=10),
        sets(integers(), max_size=10), fractions(), text(),
        integers(max_value=-1)
    )
)
def test_invalid_sample_steps(sample_steps):
    with pytest.raises(AssertionError):
        graph = tf.Graph()
        with tf.Session(graph=graph) as session:
            BayesianNeuralNetwork(sample_steps=sample_steps, session=session)


@given(
    one_of(
        floats(), complex_numbers(), lists(integers(), max_size=10),
        sets(integers(), max_size=10), fractions(), text(),
        integers(max_value=-1)
    )
)
def test_invalid_batch_size(batch_size):
    with pytest.raises(AssertionError):
        graph = tf.Graph()
        with tf.Session(graph=graph) as session:
            BayesianNeuralNetwork(batch_size=batch_size, session=session)


@given(
    one_of(
        floats(), complex_numbers(), lists(integers(), max_size=10),
        sets(integers(), max_size=10), fractions(), text(),
        integers()
    )
)
def test_invalid_sampling_methods(sampling_method):
    with pytest.raises(ValueError):
        graph = tf.Graph()
        with tf.Session(graph=graph) as session:
            BayesianNeuralNetwork(
                sampling_method=sampling_method, session=session
            )
