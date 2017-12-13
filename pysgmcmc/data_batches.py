import numpy as np
import logging

__all__ = (
    "generate_batches",
)


def generate_batches(x, y, batch_size=20, seed=None, shuffle=False):

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
