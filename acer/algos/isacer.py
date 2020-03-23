"""
BaseActor-Critic with Experience Replay with importance sampling modifications
"""
import sys
from typing import Optional, List, Union, Dict, Tuple
# import os
# os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import gym
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

import scipy

import utils
from algos.acer import ACER


class ISACER(ACER):
    def __init__(self, observations_space: gym.Space, actions_space: gym.Space, actor_layers: Optional[Tuple[int]],
                 critic_layers: Optional[Tuple[int]], lam: float = 0.1, b: float = 3, is_alpha: float = 0.1, *args, **kwargs):
        """BaseActor-Critic with Experience Replay and importance sampling modifications

        TODO: finish docstrings
        """
        super().__init__(observations_space, actions_space, actor_layers, critic_layers, *args, **kwargs)

        std = np.exp(self._actor.log_std.numpy()[0])

        self.alpha = is_alpha
        dim = len(actions_space.shape)
        self.nb = np.pi ** (dim / 2) / scipy.special.gamma(dim) * (std * self.alpha) ** dim
        print(self.nb)

    @tf.function(experimental_relax_shapes=True)
    def _learn_from_experience_batch(self, obs, obs_next, actions, old_policies,
                                     rewards, first_obs, first_actions, dones, lengths):
        """Backward pass with single batch of experience.

        Every experience replay requires sequence of experiences with random length, thus we have to use
        ragged tensors here.

        See Equation (8) and Equation (9) in the paper (1).
        """

        NO_CUMULATIVE = False
        NO_DIVISION = True
        NO_DIVISION_CLIP = True
        NB = True
        SIMPLE_GAMMA = False
        DIVIDE_BY_MAX = False

        batches_indices = tf.RaggedTensor.from_row_lengths(values=tf.range(tf.reduce_sum(lengths)), row_lengths=lengths)
        values = tf.squeeze(self._critic.value(obs))
        values_next = tf.squeeze(self._critic.value(obs_next)) * (1.0 - tf.cast(dones, tf.dtypes.float32))
        policies, log_policies = tf.split(self._actor.prob(obs, actions), 2, axis=0)
        policies, log_policies = tf.squeeze(policies), tf.squeeze(log_policies)
        indices = tf.expand_dims(batches_indices, axis=2)

        # flat tensor
        policies_ratio = tf.math.multiply(policies, self.nb) if NB else policies if NO_DIVISION else tf.math.divide(policies, old_policies)
        
        # ragged tensor divided into batches
        policies_ratio_batches = tf.squeeze(tf.gather(policies_ratio, indices), axis=2)

        # cumprod and cumsum do not work on ragged tensors, we transform them into tensors
        # padded with 0 and then apply boolean mask to retrieve original ragged tensor
        batch_mask = tf.sequence_mask(policies_ratio_batches.row_lengths())
        policies_ratio_product =  tf.math.cumprod(policies_ratio_batches.to_tensor(), axis=1)

        truncated_densities = tf.ragged.boolean_mask(
            policies_ratio_product if NO_DIVISION_CLIP else tf.minimum(policies_ratio_product, self._b),
            batch_mask
        )
        gamma_coeffs_batches = tf.ones_like(truncated_densities).to_tensor() * self._gamma
        gamma_coeffs = tf.ragged.boolean_mask(
            tf.math.cumprod(gamma_coeffs_batches, axis=1, exclusive=True),
            batch_mask
        ).flat_values

        # flat tensors
        if NO_CUMULATIVE:
            d_coeffs = (rewards + self._gamma * values_next - values) * policies_ratio_batches.flat_values
        else:
            d_coeffs = (self._gamma if SIMPLE_GAMMA else gamma_coeffs) * (rewards + self._gamma * values_next - values) * truncated_densities.flat_values
        # ragged
        d_coeffs_batches = tf.gather_nd(d_coeffs, tf.expand_dims(indices, axis=2))
        # final summation over original batches
        if NO_CUMULATIVE:
            d = tf.stop_gradient(tf.reduce_mean(d_coeffs_batches))
        else:
            d = tf.stop_gradient(tf.reduce_sum(d_coeffs_batches, axis=1))

        self._backward_pass(first_obs, first_actions, d)

        _, new_log_policies = tf.split(self._actor.prob(obs, actions), 2, axis=0)
        new_log_policies = tf.squeeze(new_log_policies)
        approx_kl = tf.reduce_mean(policies - new_log_policies)
        with tf.name_scope('actor'):
            tf.summary.scalar('mean_policies_ratio', tf.reduce_mean(policies_ratio), self._tf_time_step)
            tf.summary.scalar('sample_approx_kl_divergence', approx_kl, self._tf_time_step)

    def _fetch_offline_batch(self) -> List[Dict[str, Union[np.array, list]]]:
        trajectory_lens = [1 for _ in range(self._num_parallel_envs)]
        batch = []
        [batch.extend(self._memory.get(trajectory_lens)) for _ in range(self._batches_per_env)]
        return batch