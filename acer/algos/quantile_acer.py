"""
BaseActor-Critic with Experience Replay with quantile critic.
"""
import sys
from typing import Optional, List, Union, Dict, Tuple
# import os
# os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import gym
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

import utils
from algos.base import BaseACERAgent, BaseActor, CategoricalActor, GaussianActor, BaseCritic


class QuantileCritic(BaseCritic):

    def __init__(self, observations_space: gym.Space, layers: Optional[Tuple[int]], tf_time_step: tf.Variable, outputs: int, 
                 kappa: float, *args, **kwargs):
        """ 
        Args:
            atoms: number of quantiles to be estimated
            kappa: the parameter of hubile loss, use 0 for regular quantile loss 
        """
        super().__init__(observations_space, layers, tf_time_step, *args, **kwargs)

        self._outputs = outputs
        self._kappa = kappa
        self._v = tf.keras.layers.Dense(outputs, kernel_initializer=utils.normc_initializer())

    def values(self, observations: tf.Tensor,  **kwargs) -> tf.Tensor:
        """Calculates quantiles' values of value function

        Args:
            observations: batch [batch_size, observations_dim] of observations vectors

        Returns:
            Tensor of [batch_size, atoms]

        """
        x = self._hidden_layers[0](observations)

        for layer in self._hidden_layers[1:]:
            x = layer(x)

        v = self._v(x)

        return v

    def loss(self, observations: np.array, d: tf.Tensor) -> tf.Tensor:
        """Computes Critic's loss.

        Args:
            observations: batch [batch_size, observations_dim] of observations vectors
            d: update coefficient of value function lower bound approximation
        """
        value = self.values(observations)

        loss = tf.reduce_sum(-tf.math.multiply(value, d))

        if self._outputs > 1:
            value_lower = tf.slice(value, [0, 0], [-1, self._outputs - 1])
            value_higher = tf.slice(value, [0, 1], [-1, self._outputs - 1])

            correct_order = value_lower < value_higher

        with tf.name_scope('critic'):
            tf.summary.scalar('batch_loss_value', loss, step=self._tf_time_step)
            tf.summary.scalar('batch_v_mean', tf.reduce_mean(value), self._tf_time_step)
            if self._outputs > 1:
                tf.summary.scalar('correct_quantile_order', tf.reduce_mean(tf.cast(correct_order, tf.float32)), self._tf_time_step)
                tf.summary.scalar('batch_1st_q', tf.reduce_mean(tf.slice(value, [0, 0], [-1, 1])), self._tf_time_step)
                tf.summary.scalar('batch_middle_q', tf.reduce_mean(tf.slice(value, [0, self._outputs // 2], [-1, 1])), self._tf_time_step)
                tf.summary.scalar('batch_last_q', tf.reduce_mean(tf.slice(value, [0, self._outputs - 1], [-1, 1])), self._tf_time_step)
                tf.summary.scalar('batch_1st_q_d', tf.reduce_mean(tf.slice(d, [0, 0], [-1, 1])), self._tf_time_step)
                tf.summary.scalar('batch_middle_q_d', tf.reduce_mean(tf.slice(d, [0, self._outputs // 2], [-1, 1])), self._tf_time_step)
                tf.summary.scalar('batch_last_q_d', tf.reduce_mean(tf.slice(d, [0, self._outputs - 1], [-1, 1])), self._tf_time_step)
        return loss


class QACER(BaseACERAgent):
    def __init__(self, observations_space: gym.Space, actions_space: gym.Space, actor_layers: Optional[Tuple[int]],
                 critic_layers: Optional[Tuple[int]], lam: float = 0.1, b: float = 3, atoms: int = 50, kappa: float = 0.,
                 border_atoms=False, *args, **kwargs):
        """BaseActor-Critic with Experience Replay and pessimistic Critic

        TODO: finish docstrings
        """
        self._lam = lam
        self._b = b
        self._atoms = atoms
        self._kappa = kappa
        self._border_atoms = border_atoms

        super().__init__(observations_space, actions_space, actor_layers, critic_layers, *args, **kwargs)

    def _init_actor(self) -> BaseActor:
        if self._is_discrete:
            return CategoricalActor(
                self._observations_space, self._actions_space, self._actor_layers,
                self._actor_beta_penalty, self._tf_time_step
            )
        else:
            return GaussianActor(
                self._observations_space, self._actions_space, self._actor_layers,
                self._actor_beta_penalty, self._actions_bound, self._std, self._tf_time_step
            )

    def _init_critic(self) -> BaseCritic:
        outputs = self._atoms + 1 if self._border_atoms else self._atoms
        return QuantileCritic(self._observations_space, self._critic_layers, self._tf_time_step, outputs, self._kappa)

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
        obs = self._process_observations(obs)
        obs_next = self._process_observations(obs_next)
        rewards = self._process_rewards(rewards)

        batches_indices = tf.RaggedTensor.from_row_lengths(values=tf.range(tf.reduce_sum(lengths)), row_lengths=lengths)
        values = self._critic.value(obs)
        values_next = self._critic.value(obs_next) * tf.expand_dims((1.0 - tf.cast(dones, tf.dtypes.float32)), axis=1)
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
            tf.minimum(policies_ratio_product, self._b),
            batch_mask
        )
        gamma_coeffs_batches = tf.ones_like(truncated_densities).to_tensor() * self._gamma
        gamma_coeffs = tf.ragged.boolean_mask(
            tf.math.cumprod(gamma_coeffs_batches, axis=1, exclusive=True),
            batch_mask
        ).flat_values

        u = tf.reshape(rewards, (-1, 1, 1)) + self._gamma * tf.expand_dims(values_next, 2) - tf.expand_dims(values, 1)

        with tf.name_scope('critic'):
            tf.summary.scalar('batch_mean_temporal_difference', tf.reduce_mean(u), step=self._tf_time_step)

        if self._border_atoms:
            tau = tf.range(0., self._atoms + 1, 1.) / self._atoms
        else:
            tau = tf.range(1., 2 * self._atoms, 2.) / (2 * self._atoms)
        tau = tf.reshape(tau, (1, 1, -1))
        
        if self._kappa < 0.:
            loss1_gradient_parts = u
        elif self._kappa > 0.:
            loss1_gradient_parts = tf.abs(tau - tf.cast(tf.less(u, 0), tf.float32)) * tf.where(tf.abs(u) < self._kappa, u / self._kappa, tf.sign(u))
        else:
            loss1_gradient_parts = (tau - tf.cast(tf.less(u, 0), tf.float32))
        loss1_gradient = tf.reduce_mean(loss1_gradient_parts, axis=1)
        
        # flat tensors
        d1_coeffs = tf.expand_dims(gamma_coeffs, axis=1) * loss1_gradient * tf.expand_dims(truncated_densities.flat_values, axis=1)
        # ragged
        d1_coeffs_batches = tf.gather_nd(d1_coeffs, indices)
        # final summation over original batches
        d1 = tf.stop_gradient(tf.reduce_sum(d1_coeffs_batches, axis=1))

        td2 = tf.reduce_mean(u, axis=(1, 2))
        # flat tensors
        d2_coeffs = gamma_coeffs * td2 * truncated_densities.flat_values
        # ragged
        d2_coeffs_batches = tf.gather_nd(d2_coeffs, tf.expand_dims(indices, axis=2))
        # final summation over original batches
        d2 = tf.stop_gradient(tf.reduce_sum(d2_coeffs_batches, axis=1))

        self._backward_pass(first_obs, first_actions, d1, d2)

        _, new_log_policies = tf.split(self._actor.prob(obs, actions), 2, axis=0)
        new_log_policies = tf.squeeze(new_log_policies)
        approx_kl = tf.reduce_mean(policies - new_log_policies)
        with tf.name_scope('actor'):
            tf.summary.scalar('sample_approx_kl_divergence', approx_kl, self._tf_time_step)

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
            loss = self._critic.loss(observations, d1)
        grads = tape.gradient(loss, self._critic.trainable_variables)
        gradients = zip(grads, self._critic.trainable_variables)

        self._critic_optimizer.apply_gradients(gradients)

    def _fetch_offline_batch(self) -> List[Dict[str, Union[np.array, list]]]:
        trajectory_lens = [np.random.geometric(1 - self._lam) + 1 for _ in range(self._num_parallel_envs)]
        batch = []
        [batch.extend(self._memory.get(trajectory_lens)) for _ in range(self._batches_per_env)]
        return batch
