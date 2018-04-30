import numpy as np
import theano
import theano.tensor as T

def shuffle(*data):
    idxs = np.random.permutation(np.arange(len(data[0])))
    if len(data) == 1:
        return [data[0][idx] for idx in idxs]
    else:
        return [[d[idx] for idx in idxs] for d in data]


def shared0s(shape, dtype=theano.config.floatX, name=None):
    return sharedX(np.zeros(shape), dtype=dtype, name=name)


def iter_data(*data, **kwargs):
    size = kwargs.get('size', 128)
    batches = len(data[0]) / size
    if len(data[0]) % size != 0:
        batches += 1
    for b in range(batches):
        start = b * size
        end = (b + 1) * size
        if len(data) == 1:
            yield data[0][start:end]
        else:
            yield tuple([d[start:end] for d in data])


def intX(X):
    return np.asarray(X, dtype=np.int32)


def floatX(X):
    return np.asarray(X, dtype=theano.config.floatX)


def sharedX(X, dtype=theano.config.floatX, name=None, broadcastable=None):
    if broadcastable:
        return theano.shared(np.asarray(X, dtype=dtype), name=name, broadcastable=broadcastable)
    else:
        return theano.shared(np.asarray(X, dtype=dtype), name=name)

def smorms3(cost, params, lrate=1e-3, eps=1e-16, gather=False):
    updates = []
    optim_params = []
    param_defaults = []
    grads = T.grad(cost, params)

    for p, grad in zip(params, grads):
        mem = sharedX(p.get_value() * 0. + 1.)
        g = sharedX(p.get_value() * 0.)
        g2 = sharedX(p.get_value() * 0.)
        if gather:
            optim_params.append(mem)
            optim_params.append(g)
            optim_params.append(g2)
            param_defaults.append(mem.get_value())
            param_defaults.append(g.get_value())
            param_defaults.append(g2.get_value())

        r_t = 1. / (mem + 1)
        g_t = (1 - r_t) * g + r_t * grad
        g2_t = (1 - r_t) * g2 + r_t * grad**2
        p_t = p - grad * T.minimum(lrate, g_t * g_t / (g2_t + eps)) / \
              (T.sqrt(g2_t + eps) + eps)
        mem_t = 1 + mem * (1 - g_t * g_t / (g2_t + eps))

        updates.append((g, g_t))
        updates.append((g2, g2_t))
        updates.append((p, p_t))
        updates.append((mem, mem_t))
    if gather:
        return updates, (optim_params, param_defaults)
    else:
        return updates

def smorms3_reset(optim_params):
    params, param_defaults = optim_params
    for p,d in zip(params, param_defaults):
        p.set_value(d)
