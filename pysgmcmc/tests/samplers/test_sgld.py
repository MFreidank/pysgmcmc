import tensorflow as tf

from pysgmcmc.samplers.sgld import SGLDSampler
from pysgmcmc.tests.samplers.sampler_testing import seed_test


def test_seeding():
    seed_test(SGLDSampler, dtype=tf.float32)
