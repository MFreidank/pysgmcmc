from collections import OrderedDict
from pysgmcmc.optimizers.sgld import SGLD


optimizer_classes = OrderedDict((
    ("SGLD", SGLD),
))

__all__ = tuple(optimizer_name for optimizer_name in optimizer_classes)
