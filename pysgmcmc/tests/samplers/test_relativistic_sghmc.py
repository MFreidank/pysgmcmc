import tensorflow as tf

from pysgmcmc.samplers.relativistic_sghmc import RelativisticSGHMCSampler
from pysgmcmc.tests.samplers.sampler_testing import seed_test


def test_seeding():
    seed_test(RelativisticSGHMCSampler, dtype=tf.float32)
