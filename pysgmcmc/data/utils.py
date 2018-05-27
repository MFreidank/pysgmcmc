import numpy as np


def infinite_dataloader(dataloader):
    while True:
        for batch in dataloader:
            yield batch


#  Data Preprocessing {{{ #

def zero_mean_unit_var_normalization(X, mean=None, std=None):
    mean = np.mean(X, axis=0) if mean is None else mean
    std = np.std(X, axis=0) if std is None else std
    return np.true_divide(X - mean, std), mean, std


def zero_mean_unit_var_unnormalization(X_normalized, mean, std):
    return X_normalized * std + mean
#  }}} Data Preprocessing #
