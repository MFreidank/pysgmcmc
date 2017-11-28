from itertools import islice
from random import choice

import tensorflow as tf

from numpy import allclose
from numpy.random import randint

from pysgmcmc.diagnostics.objective_functions import (
    gmm1_log_likelihood as gmm1,
    banana_log_likelihood as banana,
)

objective_functions = (
    (gmm1, lambda: [tf.Variable(0., dtype=tf.float32, name="x")]),
    (banana, lambda: [tf.Variable(0., dtype=tf.float32, name="x"),
                      tf.Variable(6., dtype=tf.float32, name="y")]),


)


def cost_function(log_likelihood_function):
    def wrapped(*args, **kwargs):
        return -log_likelihood_function(*args, **kwargs)
    return wrapped


def seed_test(sampler_constructor, **sampler_args):

    objective_function = choice(objective_functions)
    seed = randint(0, 2 ** 32 - 1)
    n_samples = randint(1, 100)

    function, param_generator = objective_function

    def fresh_chain():
        tf.reset_default_graph()
        graph = tf.Graph()

        with tf.Session(graph=graph) as session:
            params = param_generator()
            sampler = sampler_constructor(
                params=params,
                cost_fun=cost_function(function),
                seed=seed,
                session=session,
                **sampler_args
            )

            session.run(tf.global_variables_initializer())

            return list(islice(sampler, n_samples))

    chain1, chain2 = fresh_chain(), fresh_chain()

    for (sample1, cost1), (sample2, cost2) in zip(chain1, chain2):
        assert allclose(cost1, cost2)
        assert allclose(sample1, sample2)


def reset_test(sampler_constructor, **sampler_args):
    objective_function = choice(objective_functions)
    seed = randint(0, 2 ** 32 - 1)

    function, param_generator = objective_function

    graph = tf.Graph()
    with tf.Session(graph=graph) as session:
        params = param_generator()
        sampler = sampler_constructor(
            params=params,
            cost_fun=cost_function(function),
            seed=seed,
            session=session,
            **sampler_args
        )

        session.run(tf.global_variables_initializer())

        theta = session.run(sampler.params)

        theta, costs, _ = sampler.leapfrog()
        # resetting sampler
        sampler.reset()

        theta_ = session.run(sampler.params)
        print(theta, theta_)

        assert allclose(theta, theta_, atol=1e-01)
