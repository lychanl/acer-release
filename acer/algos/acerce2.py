from typing import Optional, List, Union, Dict, Tuple
import gym
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

from algos.base import BaseActor, CategoricalActor, Critic
from algos.acerce import ACERCE


class ACERCE2(ACERCE):

    @tf.function(experimental_relax_shapes=True)
    def _learn_from_experience_batch(self, obs, obs_next, actions, old_policies,
                                     rewards, first_obs, first_actions, dones, lengths):
        batches_indices = tf.RaggedTensor.from_row_lengths(values=tf.range(tf.reduce_sum(lengths)), row_lengths=lengths)
        values = tf.squeeze(self._critic.value(obs))
        rho = tf.squeeze(self._explorer.value(obs))
        values_next = tf.squeeze(self._critic.value(obs_next)) * (1.0 - tf.cast(dones, tf.dtypes.float32))
        policies, log_policies = tf.split(self._actor.prob(obs, actions), 2, axis=0)
        policies, log_policies = tf.squeeze(policies), tf.squeeze(log_policies)
        indices = tf.expand_dims(batches_indices, axis=2)

        # flat tensor
        policies_ratio = tf.math.divide(policies, old_policies)
        # ragged tensor divided into batches
        policies_ratio_batches = tf.squeeze(tf.gather(policies_ratio, indices), axis=2)

        # cumprod and cumsum do not work on ragged tensors, we transform them into tensors
        # padded with 0 and then apply boolean mask to retrieve original ragged tensor
        batch_mask = tf.sequence_mask(policies_ratio_batches.row_lengths())
        policies_ratio_product = tf.math.cumprod(policies_ratio_batches.to_tensor(), axis=1)

        truncated_densities = tf.ragged.boolean_mask(
            tf.tanh(policies_ratio_product / self._b) * self._b,
            batch_mask
        )

        gamma_coeffs_batches = tf.ones_like(policies_ratio_batches).to_tensor() * self._gamma
        gamma_coeffs = tf.ragged.boolean_mask(
            tf.math.cumprod(gamma_coeffs_batches, axis=1, exclusive=True),
            batch_mask
        ).flat_values

        # flat tensors
        d_coeffs = gamma_coeffs * (rewards + self._gamma * values_next - values) * truncated_densities.flat_values
        # ragged
        d_coeffs_batches = tf.gather_nd(d_coeffs, tf.expand_dims(indices, axis=2))
        # final summation over original batches
        d = tf.stop_gradient(tf.reduce_sum(d_coeffs_batches, axis=1))

        mu = self._actor.act_deterministic(obs)
        actions_diff = tf.expand_dims(actions - mu, axis=1)
        h_i = tf.matmul(tf.matmul(actions_diff, self._c_inverse), tf.linalg.matrix_transpose(actions_diff))
        # exp_rho_h = tf.tanh(tf.expand_dims(tf.squeeze(h_i) * tf.exp(-rho), axis=1) / self._b) * self._b
        exp_rho_h = tf.expand_dims(tf.squeeze(h_i) * tf.exp(-rho), axis=1)

        kappa = tf.math.log(self._n) - 0.5 * rho - 0.5

        with tf.name_scope('actor'):
            tf.summary.scalar('forced_ratio', tf.reduce_mean(tf.exp(policies - tf.squeeze(kappa))), self._tf_time_step)

        exploration_gain = (0.5 * exp_rho_h - 0.5 + tf.math.log(self._alpha)) * exp_rho_h

        self._backward_pass(first_obs, first_actions, d, exploration_gain, obs)

        _, new_log_policies = tf.split(self._actor.prob(obs, actions), 2, axis=0)
        new_log_policies = tf.squeeze(new_log_policies)
        approx_kl = tf.reduce_mean(policies - new_log_policies)
        with tf.name_scope('actor'):
            tf.summary.scalar('sample_approx_kl_divergence', approx_kl, self._tf_time_step)