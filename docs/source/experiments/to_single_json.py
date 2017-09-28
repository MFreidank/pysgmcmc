#!/usr/bin/python3
# -*- coding: iso-8859-15 -*-
"""
Combine effective sample size results spread across many small json files
into one big json file which can be used to visualize those results.
"""
import argparse
from glob import glob
import json
from collections import defaultdict


def main():
    parser = argparse.ArgumentParser(
        description="Combine ess results spread across multiple files "
                    "obtained by running many cluster jobs into "
                    "one single json file."
    )

    parser.add_argument(
        "json_directory",
        help="Directory containing many small json files to combine."
    )

    parser.add_argument(
        "--output-filename",
        help="Filename to write combined json data to. Defaults to `out.json`",
        action="store",
        dest="output_filename",
        default="out.json"
    )

    args = parser.parse_args()

    functions = (
        "banana", "gmm1", "gmm2", "gmm3"
    )

    json_data = {function_name: defaultdict(list) for function_name in functions}

    for filename in glob("{}/*".format(args.json_directory)):
        function_name = next(
            function_name for function_name in functions if function_name in filename
        )
        with open(filename, "r") as f:
            data = json.load(f)
        print(tuple(data.items()))
        stepsize, result = next(iter(data.items()))

        json_data[function_name][stepsize].append(result)

    with open(args.output_filename, "w") as f:
        json.dump(json_data, f)


if __name__ == "__main__":
    main()
