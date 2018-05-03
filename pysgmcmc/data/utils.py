import numpy as np
from torch.utils.data import DataLoader


class InfiniteDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.iterator = super().__iter__()

    def __iter__(self):
        return self

    def __next__(self):
        try:
            el = next(self.iterator)
        except StopIteration:
            self.iterator = super().__iter__()
            el = next(self.iterator)
        return el


#  Data Preprocessing {{{ #

def zero_mean_unit_var_normalization(X, mean=None, std=None):
    mean = np.mean(X, axis=0) if mean is None else mean
    std = np.std(X, axis=0) if std is None else std
    return np.true_divide(X - mean, std), mean, std


def zero_mean_unit_var_unnormalization(X_normalized, mean, std):
    return X_normalized * std + mean
#  }}} Data Preprocessing #
