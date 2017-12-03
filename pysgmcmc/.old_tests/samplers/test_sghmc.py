import tensorflow as tf

from pysgmcmc.samplers.sghmc import SGHMCSampler
from pysgmcmc.tests.samplers.sampler_testing import (
    seed_test, reset_test
)


def test_seeding():
    seed_test(SGHMCSampler, dtype=tf.float32)


def test_resetting():
    reset_test(SGHMCSampler, dtype=tf.float32)
