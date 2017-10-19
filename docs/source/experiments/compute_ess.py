#!/usr/bin/python3
# -*- coding: iso-8859-15 -*-
import argparse
import json
import tensorflow as tf
import numpy as np

from collections import namedtuple, defaultdict
from pymc3.backends.base import MultiTrace
from pymc3.diagnostics import effective_n as ess

import sys
from os.path import dirname, realpath, join as path_join
SCRIPT_PATH = dirname(realpath(__file__))
sys.path.insert(0, path_join(SCRIPT_PATH, "..", "..", ".."))

from pysgmcmc.samplers.relativistic_sghmc import RelativisticSGHMCSampler
from pysgmcmc.stepsize_schedules import ConstantStepsizeSchedule
from pysgmcmc.samplers.sghmc import SGHMCSampler
from pysgmcmc.samplers.sgld import SGLDSampler

from pysgmcmc.diagnostics.sample_chains import PYSGMCMCTrace
from pysgmcmc.diagnostics.objective_functions import (
    banana_log_likelihood,
    gmm1_log_likelihood, gmm2_log_likelihood,
    gmm3_log_likelihood
)


def main():
    parser = argparse.ArgumentParser(
        description="Small script to study the relationship between stepsize "
                    "of a sampler and effective sample sizes (ESS) on "
                    "on four different benchmarks."
    )

    parser.add_argument(
        "benchmark",
        help="Benchmark function to sample from. "
        "One of: 'banana', 'gmm1', 'gmm2', 'gmm3'. "
        "For reference, see: http://proceedings.mlr.press/v54/lu17b/lu17b.pdf.",
    )

    parser.add_argument(
        "--sampler",
        help="Sampler to study.",
        default="RelativisticSGHMC",
        action="store", dest="sampler"
    )

    parser.add_argument(
        "--n-iterations",
        help="Number of total iterations to perform for each stepsize",
        action="store", type=int,
        dest="n_iterations",
        default=1
    )

    parser.add_argument(
        "--n-chains",
        help="Number of chains to extract for each stepsize. Defaults to `20`.",
        dest="n_chains",
        action="store", type=int,
        default=20
    )

    parser.add_argument(
        "--samples-per-chain",
        help="Number of samples to extract for each chain. Defaults to `10 ** 4`.",
        dest="samples_per_chain",
        action="store", type=int,
        default=10**4
    )

    parser.add_argument(
        "--keep-every",
        help="Keep only every nth sample during sampling. Defaults to `10`.",
        dest="keep_every",
        action="store", type=int,
        default=10
    )

    parser.add_argument(
        "--stepsize-min",
        help="Minimal stepsize to evaluate. Defaults to `0.01`",
        dest="stepsize_min",
        action="store", type=float,
        default=0.01
    )

    parser.add_argument(
        "--stepsize-max",
        help="Maximal stepsize to evaluate. Defaults to `8.0`",
        dest="stepsize_max",
        action="store", type=float,
        default=8.0
    )

    parser.add_argument(
        "--stepsize-increment",
        help="Increment for the range of stepsizes tried. "
             "Total values will range from `stepsize-min` to `stepsize-max`."
             "Defaults to `0.05`",
        dest="stepsize_step",
        action="store", type=float,
        default=0.05
    )

    parser.add_argument(
        "--stepsize",
        help="Stepsize to use. Note that this overwrites ranges for the stepsize specified via "
             "--stepsize_min, --stepsize_max, --stepsize_step",
        dest="stepsize",
        action="store", type=float,
        default=None
    )

    parser.add_argument(
        "-o", "--output-file",
        help="Output filename to write results to. Defaults to 'output.json'.",
        action="store",
        dest="output_filename",
        default="output.json"



    )

    args = parser.parse_args()

    ObjectiveFunction = namedtuple(
        "ObjectiveFunction", ["function", "dimensionality"]
    )

    objective_functions = {
        "banana": ObjectiveFunction(
            function=banana_log_likelihood, dimensionality=2
        ),
        "gmm1": ObjectiveFunction(
            function=gmm1_log_likelihood, dimensionality=1
        ),
        "gmm2": ObjectiveFunction(
            function=gmm2_log_likelihood, dimensionality=2
        ),
        "gmm3": ObjectiveFunction(
            function=gmm3_log_likelihood, dimensionality=3
        ),
    }

    if args.benchmark not in objective_functions:
        raise ValueError(
            "Unsupported benchmark function: '{}'. "
            "Must be one of: "
            "'banana', 'gmm1', 'gmm2', 'gmm3'.".format(args.benchmark)
        )

    function_name = args.benchmark

    n_iterations = args.n_iterations
    assert n_iterations >= 1, "--n-iterations: must be >= 1"

    n_chains = args.n_chains
    assert n_chains >= 2, "--n-chains: must be >= 2 to compute ess"

    samples_per_chain = args.samples_per_chain
    assert samples_per_chain >= 1, "--samples-per-chain: must be >= 1"

    keep_every = args.keep_every
    assert keep_every >= 1, "--keep-every: must be >= 1"

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

    function, dimensionality = objective_functions[function_name]

    assert args.stepsize_min <= args.stepsize_max, "--stepsize-min must be <= --stepsize-max"
    assert args.stepsize_min >= 0.0, "--stepsize-min must be >= 0.0"
    assert args.stepsize_step > 0, "--stepsize-increment must be > 0.0"

    samplers = {
        "RelativisticSGHMC": RelativisticSGHMCSampler,
        "SGHMC": SGHMCSampler,
        "SGLD": SGLDSampler,
    }

    assert args.sampler in samplers

    sampler_fun = samplers[args.sampler]

    if args.stepsize is None:
        stepsizes = np.arange(
            args.stepsize_min, args.stepsize_max, args.stepsize_step
        )
    else:
        stepsizes = (args.stepsize,)

    ess_vals = defaultdict(list)

    for _ in range(n_iterations):
        for stepsize in stepsizes:
            tf.reset_default_graph()
            graph = tf.Graph()

            with tf.Session(graph=graph) as session:
                if function_name == "banana":
                    params = [
                        tf.Variable(0., dtype=tf.float32, name="x"),
                        tf.Variable(6., dtype=tf.float32, name="y")
                    ]
                    varnames = ["x", "y"]
                else:
                    params = [tf.Variable(0., dtype=tf.float32, name="x")]
                    varnames = ["x"]

                sampler = sampler_fun(
                    epsilon=ConstantStepsizeSchedule(stepsize),
                    params=params,
                    cost_fun=cost_function(function),
                    session=session,
                    dtype=tf.float32
                )

                session.run(tf.global_variables_initializer())

                traces = []

                for chain_id in range(n_chains):
                    samples = extract_samples(
                        sampler, n_samples=samples_per_chain, keep_every=10
                    )
                    single_trace = PYSGMCMCTrace(
                        chain_id=chain_id, samples=samples, varnames=varnames
                    )
                    traces.append(single_trace)
                multi_trace = MultiTrace(traces)

                mean_ess = np.mean(list(ess(multi_trace).values()))

                ess_vals[stepsize].append(mean_ess)

    with open("/home/freidanm/cluster_utils/results/{filename}".format(filename=args.output_filename), "w") as f:
        json.dump(ess_vals, f)


if __name__ == "__main__":
    main()
