import logging

try:
    from tqdm import tqdm
except ImportError:
    logging.warn(
        "`tqdm` package could not be found. No progressbar is available!\n"
        "Do `pip install tqdm` to display a progressbar."
    )

    class TrainingProgressbar(object):
        """ Dummy progressbar that displays nothing and simply iterates the iterable.
            Used as placeholder if `tqdm` is not installed.
        """
        def __init__(self, iterable, *args, **kwargs):
            self.iterable = iterable

        def __iter__(self):
            return self

        def __next__(self):
            return next(self.iterable)
else:
    class TrainingProgressbar(tqdm):
        """ Slightly customized `tqdm` progressbar. """
        def __init__(self, losses=None, update_every=100, *args, **kwargs):
            """ Set up progressbar to track `losses` and update in a given interval.

            Parameters
            ----------
            losses : typing.Iterable[pysgmcmc.torch_typing.TorchLossFunction], optional
                Iterable of `torch.nn.modules.loss._Loss` subclasses to display.
                Default: `None`, do not display additional loss metrics.
            update_every : int, optional
                Interval to update this progressbar.
                Default: `100`, update every `100` iterations.

            """
            super().__init__(*args, **kwargs)

            self.losses = losses
            self.update_every = update_every

            if not losses:
                self.losses = dict()

        def update(self, predictions, y_batch, epoch):
            """ Check this progressbar for update.
                Recompute loss values and prettyprints them.

            Parameters
            ----------
            predictions: pysgmcmc.torch_typing.Predictions
                BNN predictions on current batch.
            y_batch: pysgmcmc.torch_typing.Targets
                Labels of current batch.
            epoch: int
                Current epoch count.

            """
            if epoch % self.update_every != 0:
                return

            postfix = tuple(
                "{loss}: {value}".format(
                    loss=loss_name, value=loss_function(input=predictions, target=y_batch)
                )
                for loss_name, loss_function in self.losses.items()
            )

            self.set_postfix_str(" - ".join(postfix))
