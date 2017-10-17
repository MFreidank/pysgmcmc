# SG-Nuts Stochastic Gradient No-U-Turn Sampler
import tensorflow as tf

from pysgmcmc.samplers.base_classes import MCMCSampler


class SGNutsSampler(MCMCSampler):
    def __init__(self, params, cost_fun, session=tf.get_default_session(),
                 dtype=tf.float64, seed=None):
        self.j = 0
        # XXX Implement rest of main algorithm 3 here

    def build_tree(self, theta, r, u, v, j, epsilon):
        # XXX Replace if/else either with tf.ops or ensure that j is incremented
        # after each iteration in `next` (the latter is preferrable)
        if j == 0:
            # base case - take one leapfrog step in direction v
            theta_prime, r_prime = self.leapfrog(theta, r, v * epsilon)

            n_prime = tf.less_equal(
                u,
                tf.exp(self.likelihood(theta_prime) - 0.5 * r_prime * r_prime)
            )
            s_prime = tf.less(
                u,
                tf.exp(
                    self.delta_max + self.likelihood(theta_prime) -
                    0.5 * r_prime * r_prime
                )
            )

            return theta_prime, r_prime, theta_prime, r_prime, theta_prime, n_prime, s_prime
        else:
            # recursion - implicitly build the left and right subtrees
            tree = self.build_tree(theta, r, u, v, j - 1, epsilon)
            theta_minus, r_minus, theta_plus, r_plus, theta_prime, n_prime, s_prime = tree
            # XXX: Replace, s_prime is tensorflow tensor, thus
            # we need to use tensorflow ops here
            if s_prime == 1:
                # XXX: Replace, v_prime is tensorflow tensor, thus
                # we need to use tensorflow ops here
                if v == -1:
                    tree = self.build_tree(
                        theta_minus, r_minus, u, v, j - 1, epsilon
                    )
                    theta_minus, r_minus, _, _, theta_prime_prime, n_prime_prime, s_prime_prime = tree

                else:
                    tree = self.build_tree(
                        theta_plus, r_plus, u, v, j - 1, epsilon
                    )
                    _, _, theta_plus, r_plus, theta_prime_prime, n_prime_prime, s_prime_prime = tree

                # XXX With probabilitiy n_prime_prime / (n_prime + n_prime_prime)
                # set theta_prime <- theta_prime_prime

                # XXX Update s_prime
                # s_prime =
                n_prime = n_prime + n_prime_prime

                return theta_minus, r_minus, theta_plus, r_plus, theta_prime, n_prime, s_prime

    def leapfrog(self, theta, r, epsilon):
        pass
