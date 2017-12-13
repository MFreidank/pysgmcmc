# Base class
from keras.optimizers import Optimizer
from keras import backend as K


def to_metaoptimizer(optimizer):
    # Turn any keras.optimizer into a metaoptimizer we can use to tune
    # our learning rate parameter
    old_get_updates = optimizer.get_updates

    def new_get_updates(self, gradients, params):
        self.get_gradients = lambda *args, **kwargs: gradients
        return old_get_updates(loss=None, params=params)
    optimizer.get_updates = new_get_updates
    return optimizer


class Hyperoptimizer(Optimizer):
    def __init__(self, metaoptimizer, **kwargs):
        super().__init__(**kwargs)

        self.metaoptimizer = to_metaoptimizer(metaoptimizer)

    def hypergradient_update(self, dfdx, dxdlr):
        gradient = K.reshape(K.dot(K.transpose(dfdx), dxdlr), self.lr.shape)
        metaupdates = self.metaoptimizer.get_updates(
            self.metaoptimizer,
            gradients=[gradient], params=[self.lr]
        )

        self.updates.extend(metaupdates)
        *_, lr_t = metaupdates
        return lr_t
