import typing
import logging

import numpy as np

__all__ = (
    "generate_batches",
)


def generate_batches(x: np.ndarray, y: np.ndarray,
                     batch_size: int=20, seed: int=None,
                     shuffle: bool=False) -> typing.Iterator[typing.Tuple[np.ndarray, np.ndarray]]:
    """ Generate batches of data from numpy dataset `(x, y)`.
        Also supports random shuffling prior to batch extraction.

    Parameters
    ----------
    x: np.ndarray
        Datapoints in an array of shape (N, D)
    y: np.ndarray
        Labels corresponding to datapoints in `x`, shape (N, 1)
    batch_size: int, optional
        Number of datapoints to put in a single batch.
        Will be decreased appropriately if `x` does not have enough datapoints.
        Defaults to `20`.
    seed: int, optional
        Integer random seed to use. Also controls shuffling, if specified.

    shuffle:bool, optional
        Flag that controls if (a copy of) the dataset is shuffled prior to batching.
        Defaults to `False`.

    Yields
    ----------
    batch: typing.Tuple[np.ndarray, np.ndarray]
        A single batch of data.
        Tuple of two numpy arrays `(x_batch, y_batch)` where:
        x_batch has shape `(batch_size, D)` and has datapoints.
        y_batch has shape `(batch_size, 1)` and has corresponding labels.

    Examples
    ----------
    TODO

    """

    n_examples, *_ = x.shape

    if seed is None:
        seed = np.random.randint(1, 100000)

    x_ = np.asarray(x)
    y_ = np.asarray(x)

    if shuffle:
        rng_x, rng_y = np.random.RandomState(), np.random.RandomState()
        rng_x.seed(seed)
        rng_y.seed(seed)

        x_ = rng_x.shuffle(x_)
        y_ = rng_x.shuffle(y_)

    rng = np.random.RandomState()
    rng.seed(seed)

    # Check if we have enough data points to form a minibatch
    # otherwise set the batchsize equal to the number of input points
    initial_batch_size = batch_size
    batch_size = min(initial_batch_size, n_examples)

    if initial_batch_size != batch_size:
        logging.error("Not enough datapoints to form a minibatch. "
                      "Batchsize was set to %s", batch_size)

    while True:
        # `np.random.randint` is end-exclusive => for n_examples == batch_size, start == 0 holds
        start = rng.randint(0, (n_examples - batch_size + 1))

        minibatch_x = x[start:start + batch_size]
        minibatch_y = y[start:start + batch_size, None]

        yield (minibatch_x, minibatch_y)
