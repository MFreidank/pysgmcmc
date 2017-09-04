import tensorflow as tf
from pysgmcmc.tensor_utils import pdist, squareform, median


def svgd_kernel(theta, h=-1):
    n_particles, particle_dimension, *_ = theta.shape.as_list()
    sq_dist = pdist(theta)
    pairwise_dists = squareform(sq_dist) ** 2
    if h < 0:
        h = median(pairwise_dists)
        h = tf.sqrt(0.5 * h / tf.log(tf.convert_to_tensor(n_particles, dtype=theta.dtype) + 1.))

    Kxy = tf.exp(-pairwise_dists / h ** 2 / 2)
    dxkxy = -tf.matmul(Kxy, theta)
    sumkxy = tf.reduce_sum(Kxy, axis=1)

    res_multiply = tf.multiply(theta, tf.expand_dims(sumkxy, axis=1))
    dxkxy += res_multiply
    dxkxy = dxkxy / (h ** 2)
    return Kxy, dxkxy
