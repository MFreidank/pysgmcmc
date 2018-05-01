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


def close_initializers(lasagne_initializer, pytorch_initializer, shape):
    samples1 = lasagne_initializer(shape)
    mean1, var1 = np.mean(samples1), np.var(samples1)
    samples2 = pytorch_initializer(shape).numpy()
    mean2, var2 = np.mean(samples2), np.var(samples2)
    return np.allclose(mean1, mean2, atol=1e-2) and np.allclose(var1, var2, atol=1e-2)

@pytest.mark.skipif(not LASAGNE_INSTALLED, reason="Package 'lasagne' required!")
def test_he_normal():
    # this tests demonstrates that `pytorch.nn.init.kaiming_normal` must be called
    # with mode `"fan_out"`, rather than `"fan_in"` as in e.g. keras.init.he_normal
    # to achieve the same result as `lasagne.init.HeNormal()` used by our
    # reference implementation.
    shapes = ((3000, 2000), (3000 * 2000, 1), (1, 3000 * 2000), (2000, 3000))

    lasagne_initializer = lambda shape: HeNormal()(shape)
    pytorch_initializer1 = lambda shape: kaiming_normal_(
        torch.from_numpy(np.zeros(shape)), mode="fan_out", nonlinearity="linear"
    )
    pytorch_initializer2 = lambda shape: kaiming_normal_(
        torch.from_numpy(np.zeros(shape)), nonlinearity="linear"
    )


    # pytorch_initializer1 is correct, but differs from e.g. keras.init.he_normal
    # in that it uses mode == "fan_out"
    assert all(
        close_initializers(lasagne_initializer, pytorch_initializer1, shape)
        for shape in shapes
    )

    # pytorch_initializer2 is wrong some of the time (not close),
    # but uses mode == "fan_in" like e.g. keras.init.he_normal
    assert any(
        not close_initializers(lasagne_initializer, pytorch_initializer2, shape)
        for shape in shapes
    )
