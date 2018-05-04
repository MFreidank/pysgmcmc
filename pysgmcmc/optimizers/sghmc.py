import numpy as np
import torch
from torch.optim import Optimizer


class SGHMC(Optimizer):
    name = "SGHMC"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def step(self):
        pass
