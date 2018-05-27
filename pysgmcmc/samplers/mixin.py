import numpy as np
import torch


class SamplerMixin(object):
    def __init__(self, negative_log_likelihood, params, *args, **kwargs):
        self.negative_log_likelihood = negative_log_likelihood
        assert callable(self.negative_log_likelihood)
        self.params = tuple(params)
        super().__init__(params=self.params, *args, **kwargs)

    @property
    def parameters(self):
        return tuple(
            np.asarray(torch.tensor(parameter.data).numpy())
            for parameter in self.params
        )

    def sample_step(self):
        self.zero_grad()
        last_parameters = self.parameters
        last_loss = self.negative_log_likelihood(self.params)
        last_loss.backward()
        self.step()

        return last_parameters, last_loss, self.parameters

    def __next__(self):
        parameters, cost, _ = self.sample_step()
        return parameters, cost

    def __iter__(self):
        return self
