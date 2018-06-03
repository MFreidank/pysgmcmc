import logging

try:
    from tqdm import tqdm
except ImportError:
    logging.warn(
        "`tqdm` package could not be found. No progressbar is available!\n"
        "Do `pip install tqdm` to display a progressbar."
    )  # XXX: Warn that tqdm is not installed and that no progressbar will be displayed as a result.

    class TrainingProgressbar(object):
        def __init__(self, iterable, *args, **kwargs):
            self.iterable = iterable
            pass

        def __iter__(self):
            return self

        def __next__(self):
            return next(self.iterable)
else:
    class TrainingProgressbar(tqdm):
        def __init__(self, losses=None, update_every=100, *args, **kwargs):
            super().__init__(*args, **kwargs)

            self.losses = losses
            self.update_every = update_every

            if not losses:
                self.losses = dict()

        def update(self, predictions, y_batch, epoch):
            if epoch % self.update_every != 0:
                return

            postfix = tuple(
                "{loss}: {value}".format(
                    loss=loss_name, value=loss_function(input=predictions, target=y_batch)
                )
                for loss_name, loss_function in self.losses.items()
            )

            self.set_postfix_str(" - ".join(postfix))
