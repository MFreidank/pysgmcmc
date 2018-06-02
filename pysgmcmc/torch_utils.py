import torch


def get_name(object_):
    try:
        return object_.name
    except AttributeError:
        return object_.__name__


def heaviside(x):
    return torch.ones_like(x) * (x > 0).float() + (0.5 * (x == 0).float())
