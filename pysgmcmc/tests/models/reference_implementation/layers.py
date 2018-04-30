import numpy as np
import theano.tensor as T
from lasagne import init
from lasagne.layers.base import Layer, MergeLayer

from pysgmcmc.tests.optimizers.reference_implementation.utils import sharedX

def normLap(A):
    I = np.identity(A.shape[0])
    d = A.sum(axis=0)
    d = np.sqrt(1./d)
    D = np.diag(d)
    L = I - np.dot( np.dot(D,A),D )
    return L

class FixedEmbeddingLayer(Layer):
    def __init__(self, incoming, n_tasks, scale_task=1., **kwargs):
        super(FixedEmbeddingLayer, self).__init__(incoming, **kwargs)
        self.n_tasks = n_tasks
        self.scale_task = scale_task
        self.W = sharedX(np.eye(n_tasks))

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], self.n_tasks)

    def get_output_for(self, input, **kwargs):
        # this is a bit hacky but should allow us to feed
        # input in here that stems from an array containing
        # multiple inputs of different types
        #input = T.cast(input * self.scale_task, 'int32')
        input = T.iround(input * self.scale_task)
        res = self.W[input]
        return res

class TaskEmbeddingLayer(Layer):

    def __init__(self, incoming, adjacency, n_tasks, W=init.Normal(), multiplicative=False, rank1=False, scale_task=1., **kwargs):
        super(TaskEmbeddingLayer, self).__init__(incoming, **kwargs)
        self.n_tasks = n_tasks
        self.rank1 = rank1
        self.scale_task = scale_task
        if self.rank1:
            self.W = self.add_param(W, (n_tasks, 1), name='W_embed', regularizable=True)
        else:
            self.W = self.add_param(W, (n_tasks, n_tasks), name='W_embed', regularizable=True)
        if adjacency is not None:
            self.L = sharedX(np.linalg.inv(np.eye(adjacency.shape[0]) + normLap(adjacency)))
        else:
            self.L = None
        self.multiplicative = multiplicative


    def get_output_shape_for(self, input_shape):
        return (input_shape[0], self.n_tasks)

    def get_output_for(self, input, **kwargs):
        # this is a bit hacky but should allow us to feed
        # input in here that stems from an array containing
        # multiple inputs of different types
        #input = T.cast(input * self.scale_task, 'int32')
        input = T.iround(input * self.scale_task)
        if self.L is not None:
            #if self.rank1:
            W = T.dot(self.W, self.W.T)
            #else:
            #    W = self.W
            if self.multiplicative:
                #W += T.eye(self.n_tasks)
                res = (1 + W[input]) * self.L[input]
            else:
                res = W[input] + self.L[input]
        else:
            res = self.W[input]
        return res

class ElemwiseBroadcast(MergeLayer):

    def __init__(self, incomings, operator, reference=0, shuffle_target=None, axis=1, **kwargs):
        super(ElemwiseBroadcast, self).__init__(incomings, **kwargs)
        self.axis = axis
        self.shapes = [layer.output_shape for layer in incomings]
        self.reference = reference
        self.operator = operator
        self.shuffle_target = shuffle_target

    def get_output_shape_for(self, input_shape):
        return self.shapes[self.reference]
    """
        shape = [x for x in input_shape]
        max_size = 1
        for sh in self.shapes:
            if len(sh) > self.axis:
                max_size = max(max_size, sh[self.axis])
        shape[self.axis] = max_size
        return tuple(shape)
    """

    def get_output_for(self, inputs, **kwargs):
        out = None
        for i,inp in enumerate(inputs):
            if i != self.reference and self.shuffle_target is not None:
                inp = inp.dimshuffle(self.shuffle_target)
            if out is None:
                out = inp
            else:
                out = self.operator(out, inp)
        return out

class SumLayer(Layer):

    def __init__(self, incoming, axis=1, **kwargs):
        super(SumLayer, self).__init__(incoming, **kwargs)
        self.axis = axis

    def get_output_shape_for(self, input_shape):
        shape = [x for x in input_shape]
        if self.axis < len(shape) - 1:
            shape[self.axis] = 1
        else:
            shape = shape[:-1]
        return tuple(shape)

    def get_output_for(self, input, **kwargs):
        return T.sum(input, axis=self.axis, keepdims=True)


class AppendLayer(Layer):

    def __init__(self, incoming, num_units, b=init.Constant(np.log(1e-2)), **kwargs):
        super(AppendLayer, self).__init__(incoming, **kwargs)
        self.num_units = num_units
        self.b = self.add_param(b, (1, num_units), 'b', regularizable=False)
        #self.b = theano.shared(np.zeros((1, num_units)) + np.log(1e-4))

    def get_output_shape_for(self, input_shape):
        shape = [x for x in input_shape]
        shape[1] += self.num_units
        return tuple(shape)

    def get_output_for(self, input, **kwargs):
        return T.concatenate([input, T.extra_ops.repeat(self.b, input.shape[0], axis=0)], axis=1)

