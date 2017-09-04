from collections import defaultdict
from itertools import islice
import numpy as np


# XXX: Doku and finishing/polishing
class ChainExhaustedError(StopIteration):
    """docstring for ChainExhaustedError"""


class Chain(object):
    """ A single chain of samples. """
    def __init__(self, sampler, chain_length=None, parameter_names=None):
        self.sampler = sampler
        self.samples, self.costs = [], []
        self.max_samples = chain_length
        if chain_length is not None:
            for sample, cost in islice(self.sampler, chain_length):
                self.samples.append(sample)
                self.costs.append(cost)

        self.parameter_names = parameter_names

    def __iter__(self):
        return self

    def __next__(self):
        if len(self.samples) == self.max_samples:
            raise ChainExhaustedError("...")
        else:
            sample, cost = next(self.sampler)
            self.samples.append(sample)
            self.costs.append(cost)
            return sample, cost

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, key):
        return self.samples[key]

    def _parameter_samples(self, parameter_names=None):
        assert(self.samples)

        parameter_names = {
            True: [str(index) for index in range(len(self.samples[0]))],
            self.parameter_names is not None: self.parameter_names,
            parameter_names is not None: parameter_names
        }[True]

        parameter_results = [[] for _ in range(len(parameter_names))]

        for sample in self.samples:
            for parameter_index, parameter in enumerate(sample):
                parameter_results[parameter_index].append(parameter)

        return list(zip(parameter_names, parameter_results))

    def aggregate(self, aggregate_function):
        return [
            aggregate_function(parameter_samples)
            for _, parameter_samples in self._parameter_samples()
        ]

    def mean(self):
        def mean_sample_fun(sample_results):
            return np.mean(sample_results, axis=0)
        return self.aggregate(aggregate_function=mean_sample_fun)

    # XXX: Implement this method, that allows plotting a chain in different ways
    # e.g. mean/var plots of parameters, histogram plot (ess plot?), etc.
    def plot(self, kind="hist"):
        raise NotImplementedError("Plotting not yet implemented.")


class MultiChain(object):
    def __init__(self, *chains, parameter_names=None):
        # chain has n samples, each sample has values for each parameter
        self.chains = tuple(chain for chain in chains)
        self.n_chains = len(self.chains)
        # NOTE: Sanitize chains?

        self.n_parameters = 0 if self.n_chains == 0 else len(self.chains[0][0])

        if parameter_names is None:
            self.parameter_names = [
                str(index) for index in range(self.n_parameters)
            ]
            self.named = False
        else:
            self.parameter_names = parameter_names
            self.named = True

        self.parameter_dict = defaultdict(list)

        # XXX: Below this line things are broken
        for chain in self.chains:
            for sample in chain:
                for parameter_name, parameter_sample in zip(parameter_names, sample):
                    self.parameter_dict[parameter_name].append(parameter_sample)
        self.param_results = sorted(
            self.parameter_dict.items(),
            key=lambda t: self.parameter_names.index(t[0])
        )

    def info(self):
        if self.named:
            parameter_names = "\n\t".join(self.parameter_names)
        else:
            parameter_names = ""
        return (
            "MultiChain with {n_chains} chains over {n_parameters} parameters.\n"
            "{parameter_names}:\n\t"
        ).format(
            n_chains=self.n_chains, n_parameters=self.n_parameters,
            parameter_names=parameter_names
        )

    def mean_chain(self):
        return self.aggregate(aggregate_function=np.mean)

    def aggregate(self, aggregate_function=np.mean):
        return [
            aggregate_function(parameter_values)
            for _, parameter_values in self.param_results
        ]
