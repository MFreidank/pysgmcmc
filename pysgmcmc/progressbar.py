from tqdm import tqdm


class TrainingProgressbar(tqdm):
    def __init__(self, losses=None, update_every=100, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.losses = losses
        self.update_every = 100

        if not losses:
            self.losses = dict()


    def update(self, predictions, y_batch, epoch):
        if epoch % self.update_every != 0:
            return

        loss_values = (
            loss(input=predictions, target=y_batch) for loss in self.losses
        )

        postfix = tuple(
            "{loss}: {value}".format(
                loss=loss_name, value=loss_function(input=predictions, target=y_batch)
            )
            for loss_name, loss_function in self.losses.items()
        )

        self.set_postfix_str(" - ".join(postfix))
