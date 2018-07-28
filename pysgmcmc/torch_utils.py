import torch
import typing


def get_name(object_: typing.Any) -> str:
    """ Get a string representation of the name of `object_`.
        Defaults to `object_.__name__` for most objects.
        For classes in module `pysgmcmc.optimizers` and `pysgmcmc.models.losses`
        this returns an abbreviated name identifying the loss or optimizer.

    Parameters
    ----------
    object_ : typing.Any
        Any python object. Must have a `__name__` attribute.

    Returns
    ----------
    name: str
        String represenation of the name of `object_`.

    Examples
    ----------

    For most objects, this function simply returns their `__name__` attribute:

    >>> from torch.optim import Adam
    >>> get_name(Adam) == Adam.__name__
    True

    If an object sets a `name` attribute, that is used instead:

    >>> from pysgmcmc.models.losses import NegativeLogLikelihood
    >>> get_name(NegativeLogLikelihood) == "NLL" != NegativeLogLikelihood.__name__
    True

    """
    try:
        return object_.name
    except AttributeError:
        return object_.__name__
