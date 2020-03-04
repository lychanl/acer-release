from typing import Optional, Tuple

import gym
import tensorflow as tf
import numpy as np

from algos.acer import ACER


class RepresentativeACER(ACER):
    
    def __init__(self, observations_space: gym.Space, actions_space: gym.Space, actor_layers: Optional[Tuple[int]],
                 critic_layers: Optional[Tuple[int]], *args, **kwargs):
        super().__init__(observations_space, actions_space, actor_layers, critic_layers, *args, **kwargs)
        if self._is_discrete:
            raise NotImplementedError('Discrete actions are not supported yet.')

    @tf.function(experimental_relax_shapes=True)
    def _learn_from_experience_batch(self, obs, obs_next, actions, old_policies,
                                     rewards, first_obs, first_actions, dones, lengths):
        batches_indices = tf.RaggedTensor.from_row_lengths(values=tf.range(tf.reduce_sum(lengths)), row_lengths=lengths)
        values = tf.squeeze(self._critic.value(obs))
        values_next = tf.squeeze(self._critic.value(obs_next)) * (1.0 - tf.cast(dones, tf.dtypes.float32))
        policies, log_policies = tf.split(self._actor.prob(obs, actions), 2, axis=0)
        policies, log_policies = tf.squeeze(policies), tf.squeeze(log_policies)
        indices = tf.expand_dims(batches_indices, axis=2)

        cov_matrix = tf.linalg.diag(tf.repeat(tf.square(tf.exp(self._actor.log_std)), self._actions_space.shape[0]))
        kappa_coeff = ((2 * np.pi) ** self._actions_space.shape[0]) * tf.linalg.det(cov_matrix)
        kappas = kappa_coeff * tf.square(old_policies)
        policy_max = 1 / tf.sqrt(kappa_coeff)
        # flat tensor
        policies_ratio = tf.math.divide(policies, policy_max)
        policies_power = tf.math.pow(policies_ratio, kappas)

        # ragged tensor divided into batches
        policies_power_batches = tf.squeeze(tf.gather(policies_power, indices), axis=2)

        # cumprod and cumsum do not work on ragged tensors, we transform them into tensors
        # padded with 0 and then apply boolean mask to retrieve original ragged tensor
        batch_mask = tf.sequence_mask(policies_power_batches.row_lengths())
        policies_power_product_batches = tf.ragged.boolean_mask(
            tf.math.cumprod(policies_power_batches.to_tensor(), axis=1),
            batch_mask
        )

        gamma_coeffs_batches = tf.ones_like(policies_power_batches).to_tensor() * self._gamma
        gamma_coeffs = tf.ragged.boolean_mask(
            tf.math.cumprod(gamma_coeffs_batches, axis=1, exclusive=True),
            batch_mask
        ).flat_values

        # flat tensors
        d_coeffs = gamma_coeffs * (rewards + self._gamma * values_next - values) \
            * policies_power_product_batches.flat_values
        # ragged
        d_coeffs_batches = tf.gather_nd(d_coeffs, tf.expand_dims(indices, axis=2))
        # final summation over original batches
        d = tf.stop_gradient(tf.reduce_sum(d_coeffs_batches, axis=1))

        self._backward_pass(first_obs, first_actions, d)

        _, new_log_policies = tf.split(self._actor.prob(obs, actions), 2, axis=0)
        new_log_policies = tf.squeeze(new_log_policies)
        approx_kl = tf.reduce_mean(policies - new_log_policies)
        with tf.name_scope('actor'):
            tf.summary.scalar('sample_approx_kl_divergence', approx_kl, self._tf_time_step)