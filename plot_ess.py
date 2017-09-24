#!/usr/bin/python3
# -*- coding: iso-8859-15 -*-
"""
Small helper script to plot the relationship of ess with the stepsize
of relativistic sghmc on the banana function.
"""
def main():
    from pysgmcmc.diagnostics.sample_chains import PYSGMCMCTrace
    from pymc3.backends.base import MultiTrace
    from pymc3.diagnostics import effective_n as ess

    import matplotlib.pyplot as plt
    import tensorflow as tf
    import numpy as np
    from pysgmcmc.samplers.relativistic_sghmc import RelativisticSGHMCSampler

    from pysgmcmc.diagnostics.objective_functions import (
        banana_log_likelihood,
        gmm1_log_likelihood, gmm2_log_likelihood,
        gmm3_log_likelihood
    )

    from collections import namedtuple

    ObjectiveFunction = namedtuple(
        "ObjectiveFunction", ["function", "dimensionality"]
    )

    objective_functions = (
        ObjectiveFunction(
            function=banana_log_likelihood, dimensionality=2
        ),
        ObjectiveFunction(
            function=gmm1_log_likelihood, dimensionality=1
        ),
        ObjectiveFunction(
            function=gmm2_log_likelihood, dimensionality=1
        ),
        ObjectiveFunction(
            function=gmm3_log_likelihood, dimensionality=1
        ),
    )


    def cost_function(log_likelihood_function):
        def wrapped(*args, **kwargs):
            return -log_likelihood_function(*args, **kwargs)
        wrapped.__name__ = log_likelihood_function.__name__
        return wrapped


    def extract_samples(sampler, n_samples=1000, keep_every=10):
        from itertools import islice
        n_iterations = n_samples * keep_every
        return np.asarray(
            [sample for sample, _ in
             islice(sampler, 0, n_iterations, keep_every)]
        )

    banana, dimensionality = objective_functions[0]

    # stepsizes = np.arange(0.01, 0.55, 0.05)  # vary stepsize over grid
    stepsizes = np.arange(0.01, 3.5, 1.00)  # vary stepsize over grid

    ess_vals = []

    for stepsize in stepsizes:

        tf.reset_default_graph()
        graph = tf.Graph()

        with tf.Session(graph=graph) as session:
            params = [
                tf.Variable(0., dtype=tf.float32, name="x"),
                tf.Variable(6., dtype=tf.float32, name="y")
            ]
            sampler = RelativisticSGHMCSampler(
                epsilon=stepsize,
                params=params,
                cost_fun=cost_function(banana),
                session=session,
                dtype=tf.float32
            )
            session.run(tf.global_variables_initializer())

            n_chains = 20

            traces = []

            for chain_id in range(n_chains):
                samples = extract_samples(sampler, n_samples=10 ** 4, keep_every=10)
                single_trace = PYSGMCMCTrace(
                    chain_id=chain_id, samples=samples, varnames=["x", "y"]
                )
                traces.append(single_trace)
            multi_trace = MultiTrace(traces)

            mean_ess = np.mean(list(ess(multi_trace).values()))
            ess_vals.append(mean_ess)

    plt.plot(stepsizes, ess_vals)
    plt.show()


if __name__ == "__main__":
    main()
