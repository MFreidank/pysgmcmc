import numpy as np
from scipy.spatial.distance import pdist, squareform
import tensorflow as tf
import unittest
import pytest

from pysgmcmc.tensor_utils import (
    pdist as pdist_tf, squareform as squareform_tf, median as median_tf
)


class TestPDist(unittest.TestCase):
    """ Test our port of `scipy.spatial.distance.pdist` to `tensorflow` types. """
    def test_invalid_metric(self):
        input_scipy = np.random.rand(np.random.randint(2, 20), np.random.randint(0, 10))
        input_tf = tf.constant(input_scipy)
        # sanity check: input works for default metric, i.e.
        # a `NotImplementedError` is not always raised
        try:
            pdist_tf(input_tf)
        except NotImplementedError:
            self.fail(
                "Calling pdist_tf(input) [with default metric] should not "
                "raise a `NotImplementedError`, yet it did.."
            )

        # for non-supported metric inputs, we raise a NotImplementedError
        self.assertRaises(NotImplementedError, lambda: pdist_tf(input_tf, "NOTAMETRIC"))
        self.assertRaises(NotImplementedError, lambda: pdist_tf(input_tf, 1))
        self.assertRaises(NotImplementedError, lambda: pdist_tf(input_tf, 1.3))

    def test_invalid_input_shape(self):
        invalid_inputs = [
            np.random.rand(np.random.randint(2, 20), np.random.randint(3, 10), np.random.randint(0, 2)),
        ]

        invalid_inputs_tf = [
            tf.constant(invalid_input) for invalid_input in invalid_inputs
        ]

        with tf.Session() as session:
            for input_scipy, input_tensorflow in zip(invalid_inputs, invalid_inputs_tf):
                self.assertRaises(ValueError, lambda: pdist(input_scipy))
                self.assertRaises(ValueError, lambda: session.run(pdist_tf(input_tensorflow)))

    def test_valid_inputs_euclidean_pdist(self):
        n_test_arrays = 10
        test_arrays = [
            np.random.rand(np.random.randint(2, 20), np.random.randint(0, 10)) for _ in range(n_test_arrays)
        ]
        test_arrays_tensorflow = [
            tf.constant(array) for array in test_arrays
        ]

        for input_scipy, input_tensorflow in zip(test_arrays, test_arrays_tensorflow):
            result_scipy = pdist(
                input_scipy, metric="euclidean"
            )

            with tf.Session() as session:
                result_tensorflow = session.run(
                    pdist_tf(input_tensorflow)
                )
            self.assertTrue(np.allclose(result_scipy, result_tensorflow))


class TestSquareform(unittest.TestCase):
    """ Test our port of `scipy.spatial.distance.squareform` to `tensorflow` types. """
    def test_valid_vector_input_squareform(self):
        n_test_arrays = 10
        test_arrays = [
            pdist(np.random.rand(np.random.randint(2, 20), np.random.randint(0, 10)))
            for _ in range(n_test_arrays)
        ]
        test_arrays_tensorflow = [
            tf.constant(array) for array in test_arrays
        ]

        for input_scipy, input_tensorflow in zip(test_arrays, test_arrays_tensorflow):
            result_scipy = squareform(input_scipy)

            with tf.Session() as session:
                result_tensorflow = session.run(
                    squareform_tf(input_tensorflow)
                )
            self.assertTrue(np.allclose(result_scipy, result_tensorflow))

    def test_zero_input_squareform(self):
        with tf.Session() as session:
            result_tensorflow = session.run(
                squareform_tf(tf.constant([], dtype=tf.float64))
            )

        assert(np.array_equal(result_tensorflow, np.zeros((1, 1), dtype=np.float64)))

    def test_invalid_inputs_squareform(self):
        with tf.Session() as session:
            with pytest.raises(ValueError):
                session.run(
                    squareform_tf(tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], dtype=tf.float64))
                )


class TestMedian(unittest.TestCase):

    def test_valid_inputs(self):
        n_test_arrays = 10
        test_inputs = [
            np.random.rand(np.random.randint(2, 1000))
            for _ in range(n_test_arrays)
        ]
        test_inputs_tensorflow = [
            tf.constant(vector) for vector in test_inputs
        ]
        for input_numpy, input_tensorflow in zip(test_inputs, test_inputs_tensorflow):
            result_numpy = np.median(input_numpy)

            with tf.Session() as session:
                result_tensorflow = session.run(
                    median_tf(input_tensorflow)
                )
            self.assertTrue(np.allclose(result_numpy, result_tensorflow, atol=1e-02))
