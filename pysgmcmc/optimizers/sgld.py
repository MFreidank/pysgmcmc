# vim:foldmethod=marker
# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""An optimizer module for stochastic gradient Langevin dynamics.
   Ported from https://github.com/tensorflow/tensorflow/tensorflow/contrib/bayesflow/python/ops/sgld_optimizer.py with minor modifications.

   Complete list of custom modifications:
    - removed assertion checks in __init__
    - renamed "learning_rate" -> "lr" to stay close to keras notation of optimizers
    - renamed "burnin" -> "n_burn_in_steps" to stay close to our other optimizers
    - changed default value of "burnin" to `3000` instead of `25`
    - added random `seed`
"""
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variable_scope as varscope_ops
from tensorflow.python.training import optimizer
from tensorflow.python.training import training_ops


class SGLD(optimizer.Optimizer):
    """An optimizer module for stochastic gradient Langevin dynamics.
      This implements the preconditioned Stochastic Gradient Langevin Dynamics
      optimizer [1]. The optimization variable is regarded as a sample from the
      posterior under Stochastic Gradient Langevin Dynamics with noise rescaled in
      each dimension according to RMSProp [2].
      Note: If a prior is included in the loss, it should be scaled by
      `1/num_pseudo_batches`, where num_pseudo_batches is the number of minibatches
      in the data.  I.e., it should be divided by the `num_pseudo_batches` term
      described below.
      [1]: "Preconditioned Stochastic Gradient Langevin Dynamics for Deep Neural
           Networks." Chunyuan Li, Changyou Chen, David Carlson, Lawrence Carin.
           ArXiv:1512.07666, 2015. https://arxiv.org/abs/1512.07666
      [2]: http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf
      Args:
        learning_rate: Scalar `float`-like `Tensor`. The base learning rate for the
          optimizer. Must be tuned to the specific function being minimized.
        preconditioner_decay_rate: Scalar `float`-like `Tensor`. The exponential
          decay rate of the rescaling of the preconditioner (RMSprop). (This is
          "alpha" in [1]). Should be smaller than but nearly `1` to approximate
          sampling from the posterior. (Default: `0.95`)
        num_pseudo_batches: Scalar `int`-like `Tensor`. The effective number of
          minibatches in the data set.  Trades off noise and prior with the SGD
          likelihood term. Note: Assumes the loss is taken as the mean over a
          minibatch. Otherwise if the sum was taken, divide this number by the
          batch size.  (Default: `1`)
        burnin: Scalar `int`-like `Tensor`. The number of iterations to collect
          gradient statistics to update the preconditioner before starting to draw
          noisy samples. (Default: `25`)
        diagonal_bias: Scalar `float`-like `Tensor`. Term added to the diagonal of
          the preconditioner to prevent the preconditioner from degenerating.
          (Default: `1e-8`)
        name: Python `str` describing ops managed by this function.
          (Default: `"SGLDOptimizer"`)
        variable_scope: Variable scope used for calls to `tf.get_variable`.
          If `None`, a new variable scope is created using name
          `ops.get_default_graph().unique_name(name or default_name)`.
      Raises:
        InvalidArgumentError: If preconditioner_decay_rate is a `Tensor` not in
          `(0,1]`.
      """
    def __init__(self,
                 lr,
                 preconditioner_decay_rate=0.95,
                 num_pseudo_batches=1,
                 burn_in_steps=3000,
                 diagonal_bias=1e-8,
                 seed=None,
                 name=None,
                 variable_scope=None):
        default_name = 'SGLD'
        self.seed = seed
        with ops.name_scope(name, default_name, [
                lr, preconditioner_decay_rate, num_pseudo_batches, burn_in_steps,
                diagonal_bias
        ]):
            if variable_scope is None:
                var_scope_name = ops.get_default_graph().unique_name(
                    name or default_name
                )
                with varscope_ops.variable_scope(var_scope_name) as scope:
                    self._variable_scope = scope
            else:
                self._variable_scope = variable_scope

            self._preconditioner_decay_rate = ops.convert_to_tensor(
                preconditioner_decay_rate, name='preconditioner_decay_rate'
            )
            self._num_pseudo_batches = ops.convert_to_tensor(
                num_pseudo_batches, name='num_pseudo_batches'
            )
            self._burnin = ops.convert_to_tensor(burn_in_steps, name='burnin')
            self._diagonal_bias = ops.convert_to_tensor(
                diagonal_bias, name='diagonal_bias'
            )
            self._learning_rate = ops.convert_to_tensor(
                lr, name='learning_rate'
            )

            with tf.variable_scope(self._variable_scope):
                self._counter = tf.get_variable(
                    'counter', initializer=0, trainable=False
                )

            super(SGLD, self).__init__(use_locking=False, name=name or default_name)

    def _create_slots(self, var_list):
        for v in var_list:
            init_rms = tf.ones_initializer(dtype=v.dtype)
            self._get_or_make_slot_with_initializer(
                v, init_rms, v.get_shape(), v.dtype, 'rms', self._name
            )

    def _prepare(self):
        # We need to put the conversion and check here because a user will likely
        # want to decay the learning rate dynamically.
        self._learning_rate_tensor = tf.convert_to_tensor(self._learning_rate, name='learning_rate_tensor')
        self._decay_tensor = tf.convert_to_tensor(
            self._preconditioner_decay_rate, name='preconditioner_decay_rate')

        super(SGLD, self)._prepare()

    def _apply_dense(self, grad, var):
        rms = self.get_slot(var, 'rms')

        with tf.control_dependencies([
                self._update_momentum(rms, grad, math_ops.cast(self._decay_tensor, var.dtype.base_dtype))]):
            new_grad = self._apply_noisy_update(rms, grad)

        return training_ops.apply_gradient_descent(
            var,
            math_ops.cast(self._learning_rate_tensor, var.dtype.base_dtype),
            new_grad,
            use_locking=self._use_locking).op

    def _apply_sparse(self, grad, var):
        rms = self.get_slot(var, 'rms')

        with tf.control_dependencies([
                self._update_momentum(rms, grad, math_ops.cast(self._decay_tensor, var.dtype.base_dtype))]):
            new_grad = self._apply_noisy_update(rms, grad)

        return tf.apply_gradient_descent(
            var,
            tf.cast(self._learning_rate_tensor, var.dtype.base_dtype),
            new_grad,
            use_locking=self._use_locking).op

    @property
    def variable_scope(self):
        """Variable scope of all calls to `tf.get_variable`."""
        return self._variable_scope

    def _apply_noisy_update(self, mom, grad):
        # Compute and apply the gradient update following
        # preconditioned Langevin dynamics
        stddev = tf.where(
            tf.squeeze(self._counter > self._burnin),
            tf.cast(tf.rsqrt(self._learning_rate), grad.dtype),
            tf.zeros([], grad.dtype))

        preconditioner = tf.rsqrt(
            mom + tf.cast(self._diagonal_bias, grad.dtype))
        return (
            0.5 * preconditioner * grad * tf.cast(self._num_pseudo_batches, grad.dtype) +
            tf.random_normal(tf.shape(grad), 1.0, dtype=grad.dtype, seed=self.seed) *
            stddev * tf.sqrt(preconditioner))

    def _update_momentum(self, mom, grad, decay):
        # Keep an exponentially weighted moving average of squared gradients.
        # Not thread safe
        return mom.assign_add((1.0 - decay) * (tf.square(grad) - mom))
