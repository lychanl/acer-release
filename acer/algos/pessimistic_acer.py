"""
BaseActor-Critic with Experience Replay and pessimistic Critic algorithm.
"""
import sys
from typing import Optional, List, Union, Dict, Tuple
# import os
# os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import gym
import tensorflow as tf
import numpy as np

import utils
from algos.base import BaseACERAgent, BaseActor, CategoricalBaseActor, GaussianBaseActor, BaseCritic


class PessimisticCritic(BaseCritic):

    def __init__(self, observations_space: gym.Space, layers: Optional[Tuple[int]], tf_time_step: tf.Variable, *args,
                 **kwargs):
        super().__init__(observations_space, layers, tf_time_step, *args, **kwargs)

        self._v = tf.keras.layers.Dense(2, kernel_initializer=utils.normc_initializer())

    def values(self, observations: tf.Tensor,  **kwargs) -> Tuple[tf.Tensor, tf.Tensor]:
        """Calculates lower bound of the value function and value function approximation

        Args:
            observations: batch [batch_size, observations_dim] of observations vectors

        Returns:
            Tuple of two Tensors: ([batch_size, 1], [batch_size, 1]) with value function lower bound
                and value function approximations

        """
        x = self._hidden_layers[0](observations)

        for layer in self._hidden_layers[1:]:
            x = layer(x)

        v = self._v(x)

        return v

    def loss(self, observations: np.array, d1: tf.Tensor, d2: tf.Tensor) -> tf.Tensor:
        """Computes Critic's loss.

        Args:
            observations: batch [batch_size, observations_dim] of observations vectors
            d1: update coefficient of value function lower bound approximation
            d2: update coefficient of value function approximation
        """
        value = self.values(observations)

        loss = tf.reduce_mean(-tf.math.multiply(value, tf.concat([d1, d2], axis=1)))

        with tf.name_scope('critic'):
            tf.summary.scalar('batch_loss_value', loss, step=self._tf_time_step)
            v1, v2 = tf.split(value, 2, axis=1)
            tf.summary.scalar('batch_v1_mean', tf.reduce_mean(v1), step=self._tf_time_step)
            tf.summary.scalar('batch_v2_mean', tf.reduce_mean(v2), step=self._tf_time_step)
        return loss


class PACER(BaseACERAgent):
    def __init__(self, observations_space: gym.Space, actions_space: gym.Space, actor_layers: Optional[Tuple[int]],
                 critic_layers: Optional[Tuple[int]], rho: float = 0.1, b: float = 3, *args, **kwargs):
        """BaseActor-Critic with Experience Replay and pessimistic Critic

        TODO: finish docstrings
        """

        super().__init__(observations_space, actions_space, actor_layers, critic_layers, *args, **kwargs)
        self._rho = rho
        self._b = b

    def _init_actor(self) -> BaseActor:
        if self._is_discrete:
            return CategoricalBaseActor(
                self._observations_space, self._actions_space, self._actor_layers,
                self._actor_beta_penalty, self._tf_time_step
            )
        else:
            return GaussianBaseActor(
                self._observations_space, self._actions_space, self._actor_layers,
                self._actor_beta_penalty, self._actions_bound, self._std, self._tf_time_step
            )

    def _init_critic(self) -> BaseCritic:
        return PessimisticCritic(self._observations_space, self._critic_layers, self._tf_time_step)

    def learn(self):
        """
        Performs experience replay learning. Experience trajectory is sampled from every replay buffer once, thus
        single backwards pass batch consists of 'num_parallel_envs' trajectories.

        Every call executes N of backwards passes, where: N = min(c0 * time_step / num_parallel_envs, c).
        That means at the beginning experience replay intensity increases linearly with number of samples
        collected till c value is reached.
        """
        experience_replay_iterations = min([round(self._c0 * self._time_step / self._num_parallel_envs), self._c])

        for batch in self._data_loader.take(experience_replay_iterations):
            self._learn_from_experience_batch(*batch)

    @tf.function(experimental_relax_shapes=True)
    def _learn_from_experience_batch(self, obs, obs_next, actions, old_policies,
                                     rewards, first_obs, first_actions, dones, lengths):
        """Backward pass with single batch of experience.

        Every experience replay requires sequence of experiences with random length, thus we have to use
        ragged tensors here.

        See Equation (8) and Equation (9) in the paper (1).
        """
        batches_indices = tf.RaggedTensor.from_row_lengths(values=tf.range(tf.reduce_sum(lengths)), row_lengths=lengths)

        values_low, values = tf.split(self._critic.values(obs), 2, axis=1)
        values_low_next, values_next = tf.split(self._critic.values(obs_next), 2, axis=1)
        values_low, values = tf.squeeze(values_low), tf.squeeze(values)
        values_low_next, values_next = tf.squeeze(values_low_next), tf.squeeze(values_next)
        values_low_next = values_low_next * (1.0 - tf.cast(dones, tf.dtypes.float32))
        values_next = values_next * (1.0 - tf.cast(dones, tf.dtypes.float32))

        policies = tf.squeeze(self._actor.prob(obs, actions))
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
            tf.minimum(policies_ratio_product, self._b),
            batch_mask
        )
        # matrices filled with and padded with 0's
        ones = tf.ones_like(truncated_densities).to_tensor()
        gamma_coeffs_batches = ones * self._gamma
        gamma_coeffs = tf.ragged.boolean_mask(
            tf.math.cumprod(gamma_coeffs_batches, axis=1, exclusive=True),
            batch_mask
        ).flat_values

        # flat tensors
        d1_coeffs = gamma_coeffs * (rewards + self._gamma * values_low_next - values_low)
        # ragged
        d1_coeffs_batches = tf.gather_nd(d1_coeffs, tf.expand_dims(indices, axis=2))
        # final summation over original batches
        d1 = tf.stop_gradient(tf.reduce_sum(d1_coeffs_batches, axis=1))

        # K + 1 duplicated K + 1 times
        K = tf.cast(tf.repeat(lengths, lengths), tf.float32)
        # K - k
        k1 = tf.ragged.boolean_mask(
            tf.math.cumsum(ones, axis=1, reverse=True, exclusive=True),
            batch_mask
        ).flat_values
        # k + 1
        k2 = tf.ragged.boolean_mask(
            tf.math.cumsum(ones, axis=1),
            batch_mask
        ).flat_values
        # K + 1 - k
        k3 = k1 + 1
        # k
        k4 = k2 - 1
        z_coeffs = gamma_coeffs * (rewards + (
                    self._gamma * (k1 * values_low_next + k2 * values_next) - (k3 * values_low + k4 * values)) / K)
        d2_coeffs = z_coeffs * truncated_densities.flat_values
        d2_coeffs_batches = tf.gather_nd(d2_coeffs, tf.expand_dims(indices, axis=2))
        d2 = tf.stop_gradient(tf.reduce_sum(d2_coeffs_batches, axis=1))

        self._backward_pass(first_obs, first_actions, d1, d2)

    def _backward_pass(self, observations: tf.Tensor, actions: tf.Tensor, d1: tf.Tensor, d2: tf.Tensor):
        """Performs backward pass in BaseActor's and Critic's networks

        Args:
            observations: batch [batch_size, observations_dim] of observations vectors
            actions: batch [batch_size, actions_dim] of actions vectors
            d1: batch [batch_size, observations_dim] of gradient update coefficients (value lower bound)
            d2: batch [batch_size, observations_dim] of gradient update coefficients (value)
        """
        with tf.GradientTape() as tape:
            loss = self._actor.loss(observations, actions, d2)
        grads = tape.gradient(loss, self._actor.trainable_variables)
        gradients = zip(grads, self._actor.trainable_variables)

        self._actor_optimizer.apply_gradients(gradients)

        with tf.GradientTape() as tape:
            loss = self._critic.loss(observations, d1, d2)
        grads = tape.gradient(loss, self._critic.trainable_variables)
        gradients = zip(grads, self._critic.trainable_variables)

        self._critic_optimizer.apply_gradients(gradients)

    def _fetch_offline_batch(self) -> List[Dict[str, Union[np.array, list]]]:
        trajectory_lens = [np.random.geometric(self._rho) + 1 for _ in range(self._num_parallel_envs)]
        batch = []
        [batch.extend(self._memory.get(trajectory_lens)) for _ in range(self._batches_per_env)]
        return batch
