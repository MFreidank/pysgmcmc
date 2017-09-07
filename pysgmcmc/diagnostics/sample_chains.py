"""
This module contains adapter functions to obtain `pymc3.(Multi-)Trace` objects
from any of our samplers.
This allows us to use any diagnostics supported by `pymc3` to quantify
our samplers.
"""

import tensorflow as tf
import numpy as np
import pymc3


class PYSGMCMCTrace(object):
    """
    Adapter class to connect the worlds of pysgmcmc and pymc3.
    Represents a single chain/trace of samples obtained from a sgmcmc sampler.
    """
    def __init__(self, chain_id, samples, varnames=None):
        """TODO: Docstring for __init__.

        Parameters
        ----------
        chain_id : int
            TODO: doku

        samples : List[List]
            Single chain of samples extracted from
            a `pysgmcmc.MCMCSampler` instance.

        varnames : List[String] or NoneType, optional
            TODO: doku

        Examples
        ----------
        TODO

        """
        self.chain = chain_id

        assert(hasattr(samples, "__len__"))
        assert(len(samples) >= 1)
        self.samples = samples

        first_sample = self.samples[0]
        self.n_vars = len(first_sample)

        if varnames is None:
            # use anonymous variable names: enumerate
            self.varnames = [
                str(index) for index in range(self.n_vars)
            ]
        else:
            self.varnames = varnames

        assert(len(self.varnames) == self.n_vars)

    @classmethod
    def from_sampler(cls, chain_id, sampler, n_samples, varnames=None):
        """ Instantiate a trace with id `chain_id` by extracting `n_samples`
        from `sampler`.

        Parameters
        ----------
        chain_id : int
            TODO: DOKU

        sampler : pysgmcmc.sampling.MCMCSampler subclass
            TODO: DOKU

        n_samples : int
            TODO: DOKU

        varnames : List[String] or NoneType, optional
            TODO: DOKU

        Returns
        ----------
        trace : PYSGMCMCTrace
            TODO: DOKU

        Examples
        ----------
        TODO

        """
        from itertools import islice
        samples = [
            sample for sample, _ in islice(sampler, n_samples)
        ]

        # try to read variable names from sampler parameters
        if varnames is None:
            try:
                varnames = [
                    param.name for param in sampler.params
                ]
            except AttributeError:
                # could not read sampler parameters, passing `None`
                # which will use enumerated names for the parameters
                # XXX: Log that this happened
                varnames = None
        return PYSGMCMCTrace(chain_id, samples, varnames)

    def __len__(self):
        """ Length of a trace/chain is the number of samples in it. """
        return len(self.samples)

    def get_values(self, varname, burn=0, thin=1):
        """
        Get all sampled values in this trace for variable with name `varname`.

        Parameters
        ----------
        varname : string
            TODO:DOKU

        Returns
        ----------
        sampled_values : np.ndarray (N, D)
            TODO: DOKU

        Examples
        ----------
        TODO

        """
        # XXX: Error message
        if varname not in self.varnames:
            raise ValueError

        var_index = self.varnames.index(varname)

        return np.asarray(
            [sample[var_index] for sample in self.samples]
        )


def pymc3_multitrace(get_sampler, n_chains=2, samples_per_chain=100, parameter_names=None):
    """ Extract chains from `sampler` and return them as `pymc3.MultiTrace`
        object.

    Parameters
    ----------
    sampler : pysgmcmc.sampling.MCMCSampler subclass
        An instance of one of our samplers.

    parameter_names : List[String] or NoneType, optional
        List of names for each target parameter of the sampler.
        If set to `None`, simply enumerate the parameters and use those numbers
        as names.
        Defaults to `None`.

    Returns
    ----------
    multitrace : pymc3.backends.base.MultiTrace
        TODO: DOKU


    Examples
    ----------
    TODO ADD EXAMPLE

    """

    straces = []

    for chain_id in range(n_chains):
        with tf.Session() as session:
            sampler = get_sampler(session=session)
            session.run(tf.global_variables_initializer())
            trace = PYSGMCMCTrace.from_sampler(chain_id=chain_id, sampler=sampler, n_samples=samples_per_chain, varnames=parameter_names)
            straces.append(trace)

    return pymc3.backends.base.MultiTrace(straces)
