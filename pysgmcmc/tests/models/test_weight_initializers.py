import numpy as np
import pytest

try:
    from lasagne.init import HeNormal
except ImportError:
    LASAGNE_INSTALLED = False
else:
    LASAGNE_INSTALLED = True

import torch
from torch.nn.init import kaiming_normal_

@pytest.mark.skipif(not LASAGNE_INSTALLED, reason="Package 'lasagne' required!")
def test_he_normal():
    # this tests demonstrates that `pytorch.nn.init.kaiming_normal` must be called
    # with mode `"fan_out"`, rather than `"fan_in"` as in e.g. keras.init.he_normal
    # to achieve the same result as `lasagne.init.HeNormal()` used by our
    # reference implementation.
    shapes = ((1000, 2000), (2000 * 1000, 1), (1, 1000 * 2000))
    for shape in shapes:
        samples_lasagne = HeNormal()(shape)
        mean_lasagne, var_lasagne = np.mean(samples_lasagne), np.var(samples_lasagne)

        samples = torch.from_numpy(np.zeros(shape))
        samples_pytorch = kaiming_normal_(samples, mode="fan_out", nonlinearity="linear")
        mean_pytorch = np.mean(samples_pytorch.numpy())
        var_pytorch = np.var(samples_pytorch.numpy())

        assert np.allclose(mean_lasagne, mean_pytorch, atol=1e-3)
        assert np.allclose(var_lasagne, var_pytorch, atol=1e-3)

        samples_pytorch = torch.from_numpy(np.zeros(shape))
        kaiming_normal_(samples_pytorch, nonlinearity="linear")
        mean_pytorch = np.mean(samples_pytorch.numpy())
        var_pytorch = np.var(samples_pytorch.numpy())
        assert np.allclose(mean_lasagne, mean_pytorch, atol=1e-3)
        # Default kaiming_normal_ produces differing results! => not close
        assert not np.allclose(var_lasagne, var_pytorch, atol=1e-4)
