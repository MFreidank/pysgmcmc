import typing

from keras import backend as K
from keras.optimizers import Adam

from pysgmcmc.custom_typing import KerasOptimizer, KerasTensor, KerasVariable
from pysgmcmc.keras_utils import to_vector


def to_hyperoptimizer(optimizer):
    # Turn any keras.optimizer into a metaoptimizer we can use to tune
    # our learning rate parameter
    old_get_updates = optimizer.get_updates

    def new_get_updates(self,
                        gradients: typing.List[KerasTensor],
                        params: typing.List[KerasVariable]) -> typing.List[KerasTensor]:
        self.get_gradients = lambda *args, **kwargs: gradients
        return old_get_updates(loss=None, params=params)

    optimizer.get_updates = new_get_updates
    return optimizer


class Hyperoptimizer(object):
    def __init__(self,
                 hyperoptimizer: KerasOptimizer=Adam(lr=1e-5),
                 hyperloss: typing.Union[None, typing.Callable[..., KerasTensor]]=None,
                 **kwargs) -> None:
        """ Set up a hyperoptimizer for a given loss.
            A hyperoptimizer is a standard optimizer that tunes hyperparameters
            of a different optimizer using gradient information.

        Parameters
        ----------
        hyperloss: typing.Union[None, typing.Callable[..., KerasTensor]], optional
            Hyperloss for which we want to optimize hyperparameters.
            Defaults to `None`, which indicates that we should optimize
            for the normal loss that all parameters are optimized for.
        hyperoptimizer: KerasOptimizer, optional
            Keras optimizer used to minimize our loss and tune a hyperparameter.
            Defaults to `Adam` with (deliberately small) learning rate `1e-5`.

        """
        super().__init__(**kwargs)  # type: ignore
        self.hyperoptimizer = to_hyperoptimizer(hyperoptimizer)

        assert hyperloss is None or callable(hyperloss)
        self.hyperloss = hyperloss

    def hypergradient_update(self,
                             loss: KerasTensor,
                             params: typing.List[KerasVariable],
                             dxdlr: KerasTensor,
                             hyperparameter: KerasVariable) -> typing.List[KerasTensor]:
        """TODO: Docstring for hypergradient_update.

        Parameters
        ----------
        loss: KerasTensor
            TODO
        params: typing.List[KerasVariable]
            TODO
        dxdlr: KerasTensor
            TODO
        hyperparameter: KerasVariable
            TODO

        Returns
        ----------
        hyperupdates: typing.List[KerasTensor]
            TODO

        """
        if self.hyperloss:
            # Derive hyperloss(params) wrt params to get dfdx
            hyperloss = self.hyperloss(params)
            dfdx = K.expand_dims(
                to_vector(K.gradients(hyperloss, params)), axis=1
            )
        else:
            # No hyperloss given, use normal loss function.
            dfdx = K.expand_dims(to_vector(K.gradients(loss, params)), axis=1)

        gradient = K.reshape(K.dot(K.transpose(dfdx), dxdlr), hyperparameter.shape)

        hyperupdates = self.hyperoptimizer.get_updates(
            self.hyperoptimizer,
            gradients=[gradient], params=[hyperparameter]
        )

        return hyperupdates
