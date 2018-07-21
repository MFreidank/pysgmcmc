"""
This module contains adapter functions to convert chains obtained by any of
our samplers into `pymc3.MultiTrace` objects.
This allows us to directly apply `pymc3.diagnostics` and `pymc3.plots` to
any of our chains.
"""
from collections import OrderedDict
import warnings

import numpy as np
import pymc3 as pm


class PYSGMCMCTrace(pm.backends.base.BaseTrace):
    def __init__(self, chain_values, chain_id=0, varnames=None,):
        assert len(chain_values.shape) == 2

        self.num_steps, self.num_parameters = chain_values.shape

        self.varnames = varnames

        if self.varnames is None:
            warnings.warn(
                "Not given any variable names, "
                "enumerating all parameter dimensions."
            )
            self.varnames = tuple(map(str, range(self.num_parameters)))

        assert len(self.varnames) == self.num_parameters

        self.var_shapes = {
            varname: () for varname in self.varnames
        }

        self.chain = chain_id

        self.parameter_dict = OrderedDict((
            (varname, chain_values[:, varindex])
            for varindex, varname in enumerate(self.varnames)
        ))

    def __getitem__(self, index):
        raise NotImplementedError()
        assert isinstance(index, int)
        assert 0 <= index < len(self.varnames)

        return self.get_values(self.varnames[index])

    def __len__(self):
        return self.num_steps

    def get_values(self, varname, burn=0, thin=1):
        if varname not in self.varnames:
            raise ValueError(
                "Queried `PYSGMCMCTrace` for values of parameter with "
                "name '{name}' but the trace does not contain any "
                "parameter of that name. "
                "Known variable names were: '{varnames}'"
                .format(name=varname, varnames=self.varnames)
            )

        return np.asarray(
            self.parameter_dict[varname][burn::thin]
        )
