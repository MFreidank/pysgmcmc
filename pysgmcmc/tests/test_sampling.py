import pytest

from pysgmcmc.sampling import Sampler


def test_get_sampler_import_missing():
    # simulating the following scenario:
    # we introduced a new supported sampler, but we forgot to add an import
    # statement in Sampler.get_sampler
    new_sampling_method = "NEW_SAMPLER"

    def mock_supported_sampler(sampling_method):
        return sampling_method in (
            Sampler.SGHMC, Sampler.SGLD, new_sampling_method
        )

    Sampler.is_supported = mock_supported_sampler

    with pytest.raises(ValueError):
        Sampler.get_sampler(new_sampling_method)
