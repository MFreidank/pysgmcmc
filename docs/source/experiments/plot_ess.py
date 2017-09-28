#!/usr/bin/python3
# -*- coding: iso-8859-15 -*-
"""
Visualize results in a big json file by producing multiple plot figures.
"""
import argparse
import json
from os.path import isfile, abspath
from experiment_helpers import partition_dict

import numpy as np
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(
        description="Small script to plot ess for (a subset of) 'banana',"
                    "'gmm{1,2,3}' functions after loading data from "
                    "given 'json_file'."
    )

    parser.add_argument(
        "json_file",
        help="Input json file that contains ess data for all functions"
             "obtained by running a sampler on any of the given 'functions'."
    )

    parser.add_argument(
        "--functions",
        help="Functions to produce plots for. "
             "Valid inputs: 'banana', 'gmm1', 'gmm2', 'gmm3' and any combination, "
             "specified as: FUNCTION1,FUNCTION2,...",
        action="store",
        default="banana,gmm1,gmm2,gmm3"
    )

    args = parser.parse_args()
    args.json_file = abspath(args.json_file)

    assert isfile(args.json_file)

    with open(args.json_file, "r") as f:
        data = json.load(f)

    functions = args.functions.split(",")

    valid_functions = ("banana", "gmm1", "gmm2", "gmm3")

    assert all(function in valid_functions for function in functions)

    for function in functions:
        ess_data = sorted(data[function].items())

        stepsizes = tuple(
            stepsize for stepsize, _ in ess_data
        )

        average_ess = tuple(
            np.mean(ess_values) for _, ess_values in ess_data
        )

        # XXX: Add legend describing what function we see

        plt.plot(stepsizes, average_ess)
        plt.show()


if __name__ == "__main__":
    main()
