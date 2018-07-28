import numpy as np
import torch


class SamplerMixin(object):
    """ Mixin class that turns a `torch.nn.optim.Optimizer` into a MCMC sampler."""
    def __init__(self, negative_log_likelihood, params, *args, **kwargs):
        """ Instantiate a sampler object.
            (Initial) parameters are passed as iterable `params`,
            `negative_log_likelihood` is a function mapping parameters to
            a NLL value and `*args` and `**kwargs` allow specifying additional
            arguments to pass to a sampler, e.g. `lr` or `mdecay`.

        Parameters
        ----------
        negative_log_likelihood : TODO
        params : TODO

        See also
        ----------
        pysgmcmc.samplers.sghmc.SGHMC: SGHMC sampler that uses this mixin.

        """
        self.negative_log_likelihood = negative_log_likelihood
        assert callable(self.negative_log_likelihood)
        self.params = tuple(params)
        super().__init__(params=self.params, *args, **kwargs)

    @property
    def parameters(self):
        """ Return last sample as tuple of numpy arrays.

        Returns
        ----------
        current_parameters: TODO
            Tuple of numpy arrays containing last sampled values.
        """
        return tuple(
            np.asarray(torch.tensor(parameter.data).numpy())
            for parameter in self.params
        )

    def sample_step(self):
        """ Perform a single step with the sampler.

        Returns
        ----------
        step_results: TODO

        """
        self.zero_grad()
        last_parameters = self.parameters
        last_loss = self.negative_log_likelihood(self.params)
        last_loss.backward()
        self.step()

        return last_parameters, last_loss, self.parameters

    def __next__(self):
        """ Perform a step of this sampler and return parameters with costs.
            Together with `__iter__`, this allows using samplers as iterables.

        Returns
        ----------
        parameters: TODO

        cost: TODO
            NLL associated with `parameters`.
        """
        parameters, cost, _ = self.sample_step()
        return parameters, cost

    def __iter__(self):
        return self
