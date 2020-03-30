from collections import deque
from typing import Optional, List, Union, Dict, Tuple
# import os
# os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import gym
import tensorflow_probability as tfp
import tensorflow as tf
import numpy as np

from algos.acerac import NoiseGaussianActor
from algos.base import BaseActor, Critic, BaseACERAgent
from replay_buffer import BufferFieldSpec, MultiPrevReplayBuffer


def get_lambda_1(n, alpha):
    lam = np.zeros(shape=(n + 1, n + 1), dtype=np.float32)
    for i in range(n + 1):
        for j in range(i, n + 1):
            lam[i][j] = lam[j][i] = alpha ** abs(i - j) - alpha ** (i + j + 2)
    return lam


def get_lambda_0(n, alpha):
    lam = np.zeros(shape=(n + 1, n + 1), dtype=np.float32)
    for i in range(n + 1):
        for j in range(i, n + 1):
            lam[i][j] = lam[j][i] = alpha ** abs(i - j)
    return lam


def k_prod(x, y):
    operator_1 = tf.linalg.LinearOperatorFullMatrix(x)
    operator_2 = tf.linalg.LinearOperatorFullMatrix(y)
    prod = tf.linalg.LinearOperatorKronecker([operator_1, operator_2]).to_dense()
    return prod


class ACERAC2(BaseACERAgent):
    def __init__(self, observations_space: gym.Space, actions_space: gym.Space, actor_layers: Optional[Tuple[int]],
                 critic_layers: Optional[Tuple[int]], b: float = 3, tau: int = 2, *args, **kwargs):

        self._tau = tau
        self._alpha = 1 - (1 / tau)
        super().__init__(observations_space, actions_space, actor_layers, critic_layers, *args, **kwargs)
        self._b = b

        self._cov_matrix = tf.linalg.diag(tf.square(tf.exp(self._actor.log_std)))

        self._lam0_c_prod_invs = []
        self._lam1_c_prod_invs = []

        for i in range(0, self._tau):
            lam0 = get_lambda_0(i, self._alpha)
            lam1 = get_lambda_1(i, self._alpha)
            lam0_c_prod = k_prod(lam0, self._cov_matrix)
            lam1_c_prod = k_prod(lam1, self._cov_matrix)
            inv0 = tf.linalg.inv(lam0_c_prod)
            inv1 = tf.linalg.inv(lam1_c_prod)
            self._lam0_c_prod_invs.append(inv0.numpy())
            self._lam1_c_prod_invs.append(inv1.numpy())

        self._lam1_c_prod_invs = tf.ragged.constant(self._lam1_c_prod_invs)
        self._lam0_c_prod_invs = tf.ragged.constant(self._lam0_c_prod_invs)

        self._data_loader = tf.data.Dataset.from_generator(
            self._experience_replay_generator,
            (tf.dtypes.float32, tf.dtypes.float32, tf.dtypes.float32, self._actor.action_dtype, tf.dtypes.float32,
             tf.dtypes.float32, tf.dtypes.bool, tf.dtypes.int32, tf.dtypes.int32, tf.dtypes.float32, tf.dtypes.float32,
             tf.dtypes.float32, self._actor.action_dtype, self._actor.action_dtype, tf.dtypes.float32,
             tf.dtypes.float32, tf.dtypes.int32)
        ).prefetch(2)

    def _init_replay_buffer(self, memory_size: int):
        if type(self._actions_space) == gym.spaces.Discrete:
            actions_shape = (1,)
        else:
            actions_shape = self._actions_space.shape

        self._memory = MultiPrevReplayBuffer(
            action_spec=BufferFieldSpec(shape=actions_shape, dtype=self._actor.action_dtype_np),
            obs_spec=BufferFieldSpec(shape=self._observations_space.shape, dtype=self._observations_space.dtype),
            max_size=memory_size,
            num_buffers=self._num_parallel_envs
        )

    def _init_actor(self) -> BaseActor:
        if self._is_discrete:
            raise NotImplementedError
        else:
            return NoiseGaussianActor(
                self._observations_space, self._actions_space, self._actor_layers,
                self._actor_beta_penalty, self._actions_bound, self._tau, self._alpha, self._num_parallel_envs,
                'autocor', self._std, self._tf_time_step
            )

    def _init_critic(self) -> Critic:
        return Critic(self._observations_space, self._critic_layers, self._tf_time_step)

    def save_experience(self, steps: List[
        Tuple[Union[int, float, list], np.array, np.array, np.array, bool, bool]
    ]):
        super().save_experience(steps)
        ends = np.array([[step[5]] for step in steps])
        self._actor.update_ends(ends)

    def learn(self):
        experience_replay_iterations = min([round(self._c0 * self._time_step / self._num_parallel_envs), self._c])

        for batch in self._data_loader.take(experience_replay_iterations):
            self._learn_from_experience_batch(*batch)

    @tf.function(experimental_relax_shapes=True)
    def _learn_from_experience_batch(self, obs, obs_next, obs_first, actions, old_policies,
                                     rewards, dones, lengths, big_batches_lengths, is_prev_noise, is_prev_noise_batch,
                                     prev_obs, prev_actions, prev_means, obs_gradient, obs_first_gradient,
                                     lengths_gradients):

        batches_indices = tf.RaggedTensor.from_row_lengths(values=tf.range(tf.reduce_sum(lengths)), row_lengths=lengths)
        indices = tf.expand_dims(batches_indices, axis=2)

        actions_flatten = tf.squeeze(tf.gather_nd(actions, tf.expand_dims(indices, axis=2)), axis=2).merge_dims(1, 2)
        old_means_flatten = tf.squeeze(tf.gather_nd(old_policies, tf.expand_dims(indices, axis=2)), axis=2).merge_dims(1, 2)
        rewards_flatten = tf.squeeze(tf.gather_nd(rewards, tf.expand_dims(indices, axis=2)), axis=-1)
        current_prev_means = self._actor.act_deterministic(prev_obs)
        values_next = tf.squeeze(self._critic.value(obs_next)) * (1.0 - tf.cast(dones, tf.dtypes.float32))
        values = tf.squeeze(self._critic.value(obs_first))
        means = self._actor.act_deterministic(obs)

        c_invs_0 = tf.gather(self._lam0_c_prod_invs, lengths - 1).to_tensor()
        c_invs_1 = tf.gather(self._lam1_c_prod_invs, lengths - 1).to_tensor()

        is_prev_noise_batch_mask = tf.expand_dims(tf.expand_dims(is_prev_noise_batch, 1), 2)
        c_invs = c_invs_1 * is_prev_noise_batch_mask + c_invs_0 * (1 - is_prev_noise_batch_mask)

        batch_mask = tf.sequence_mask(rewards_flatten.row_lengths())
        alpha_coeffs_batches = tf.ones_like(rewards_flatten).to_tensor() * self._alpha
        alpha_coeffs = tf.expand_dims(
            tf.ragged.boolean_mask(
                tf.math.cumprod(alpha_coeffs_batches, axis=1),
                batch_mask
            ).flat_values,
            1
        )
        is_prev_noise_mask = tf.expand_dims(is_prev_noise, 1)
        mu = alpha_coeffs * (prev_actions - prev_means) * is_prev_noise_mask
        eta = alpha_coeffs * (prev_actions - current_prev_means) * is_prev_noise_mask
        mu_flatten = tf.squeeze(tf.gather_nd(mu, tf.expand_dims(indices, axis=2)), axis=2).merge_dims(1, 2)
        eta_flatten = tf.squeeze(tf.gather_nd(eta, tf.expand_dims(indices, axis=2)), axis=2).merge_dims(1, 2)

        big_batches_indices = tf.expand_dims(tf.RaggedTensor.from_row_lengths(
            values=tf.range(tf.reduce_sum(big_batches_lengths)), row_lengths=big_batches_lengths
        ), 2)
        batches_indices_gradients = tf.expand_dims(tf.RaggedTensor.from_row_lengths(
            values=tf.range(tf.reduce_sum(lengths_gradients)), row_lengths=lengths_gradients
        ), 2)

        with tf.GradientTape(persistent=True) as tape:
            means_gradient = self._actor.act_deterministic(obs_gradient)
            values_gradient = tf.squeeze(self._critic.value(obs_first_gradient))
            means_flatten = tf.squeeze(tf.gather_nd(means, tf.expand_dims(indices, axis=2)), axis=2).merge_dims(1, 2)
            actions_mu_diff_current = tf.expand_dims((actions_flatten - means_flatten - eta_flatten).to_tensor(), 1)
            actions_mu_diff_old = tf.expand_dims((actions_flatten - old_means_flatten - mu_flatten).to_tensor(), 1)
            exp_current = tf.matmul(tf.matmul(actions_mu_diff_current, c_invs), tf.transpose(actions_mu_diff_current, [0, 2, 1]))
            exp_old = tf.matmul(tf.matmul(actions_mu_diff_old, c_invs), tf.transpose(actions_mu_diff_old, [0, 2, 1]))
            density_ratio = tf.squeeze(tf.exp(-0.5 * exp_current + 0.5 * exp_old))
            truncated_density = tf.tanh(density_ratio / self._b) * self._b

        #
            batch_mask = tf.sequence_mask(rewards_flatten.row_lengths())
            gamma_coeffs_batches = tf.ones_like(rewards_flatten).to_tensor() * self._gamma
            gamma_coeffs = tf.ragged.boolean_mask(
                tf.math.cumprod(gamma_coeffs_batches, axis=1, exclusive=True),
                batch_mask
            )
            td_return = tf.reduce_sum(rewards_flatten * gamma_coeffs.with_row_splits_dtype(tf.int32), axis=1)
            d = truncated_density * (-values
                                     + td_return
                                     + tf.pow(self._gamma, tf.cast(lengths, tf.float32)) * values_next)
            means_gradient_batches = tf.squeeze(
                tf.gather_nd(means_gradient, tf.expand_dims(batches_indices_gradients, axis=2)),
                axis=2
            ).to_tensor()
            means_gradient_batches = tf.reshape(means_gradient_batches, shape=[tf.shape(means_gradient_batches)[0], -1])

            c_mu = tf.matmul(c_invs, tf.transpose(actions_mu_diff_current, [0, 2, 1]))
            c_mu_d = c_mu * tf.expand_dims(tf.expand_dims(d, axis=1), 2)
            c_mu_d_batches = tf.squeeze(
                tf.gather_nd(c_mu_d, big_batches_indices).to_tensor()
            )

            c_mu_sum = tf.stop_gradient(tf.reduce_sum(c_mu_d_batches, axis=1) / tf.expand_dims(tf.cast(big_batches_lengths, tf.float32), 1))
            d_batches = tf.squeeze(
                tf.gather_nd(d, big_batches_indices).to_tensor()
            )
            d_batches = tf.stop_gradient(tf.reduce_sum(d_batches, axis=1))

            bounds_penalty = tf.scalar_mul(
                    self._actor._beta_penalty,
                    tf.square(tf.maximum(0.0, tf.abs(means_gradient) - self._actions_bound))
            )

            bounds_penalty = tf.reduce_mean(
                tf.reduce_sum(
                    bounds_penalty,
                    axis=1
                ),
                0
            )
            actor_loss = tf.matmul(tf.expand_dims(means_gradient_batches, axis=1), tf.expand_dims(c_mu_sum, axis=2))
            actor_loss = -tf.reduce_mean(tf.squeeze(actor_loss)) + bounds_penalty
            critic_loss = -tf.reduce_mean(values_gradient * d_batches / tf.expand_dims(tf.cast(big_batches_lengths, tf.float32), 1))

        grads_actor = tape.gradient(actor_loss, self._actor.trainable_variables)
        grads_var_actor = zip(grads_actor, self._actor.trainable_variables)
        self._actor_optimizer.apply_gradients(grads_var_actor)

        with tf.name_scope('actor'):
            tf.summary.scalar(f'batch_actor_loss', actor_loss, step=self._tf_time_step)
            tf.summary.scalar(f'batch_bounds_penalty', bounds_penalty, step=self._tf_time_step)

        grads_critic = tape.gradient(critic_loss, self._critic.trainable_variables)
        grads_var_critic = zip(grads_critic, self._critic.trainable_variables)
        self._critic_optimizer.apply_gradients(grads_var_critic)

        with tf.name_scope('critic'):
            tf.summary.scalar(f'batch_critic_loss', critic_loss, step=self._tf_time_step)
            tf.summary.scalar(f'batch_value_mean', tf.reduce_mean(values), step=self._tf_time_step)

    def _fetch_offline_batch(self) -> List[Tuple[Dict[str, Union[np.array, list]], int]]:
        trajectory_lens = [[self._tau for _ in range(self._num_parallel_envs)] for _ in range(self._batches_per_env)]
        batch = []
        [batch.extend(self._memory.get(trajectories)) for trajectories in trajectory_lens]
        return batch

    def _experience_replay_generator(self):
        while True:
            offline_batches = self._fetch_offline_batch()

            obs, obs_next, obs_first, actions, policies, rewards, dones = [], [], [], [], [], [], []
            big_batches_lengths = []
            lengths = []
            is_prev_noise = []
            is_prev_noise_batch = []
            prev_obs = []
            prev_actions = []
            prev_policies = []
            obs_gradient = []
            obs_first_gradient = []
            lengths_gradient = []

            for batch, first_index in offline_batches:
                batch_size = len(batch['observations'])
                big_batch_len = 0
                obs_first_gradient.append(batch['observations'][first_index])
                obs_gradient.append(batch['observations'][first_index: ])
                lengths_gradient.append(len(obs_gradient[-1]))

                for i in range(first_index + 1, batch_size + 1):
                    obs.append(batch['observations'][first_index: i])
                    obs_first.append(batch['observations'][first_index])
                    obs_next.append(batch['next_observations'][i - 1])
                    actions.append(batch['actions'][first_index: i])
                    policies.append(batch['policies'][first_index: i])
                    rewards.append(batch['rewards'][first_index: i])
                    dones.append(batch['dones'][i - 1])
                    lengths.append(len(obs[-1]))
                    is_prev_noise_batch.append(float(first_index != 0))
                    is_prev_noise.append([float(first_index != 0) for _ in range(len(obs[-1]))])
                    big_batch_len += 1

                    prev_obs.append([batch['observations'][0] for _ in range(len(obs[-1]))])
                    prev_actions.append([batch['actions'][0] for _ in range(len(obs[-1]))])
                    prev_policies.append([batch['policies'][0] for _ in range(len(obs[-1]))])
                big_batches_lengths.append(big_batch_len)

            obs = np.concatenate(obs, axis=0)
            obs_first = np.stack(obs_first)
            obs_next = np.stack(obs_next)
            obs_gradient = np.concatenate(obs_gradient, axis=0)
            obs_first_gradient = np.stack(obs_first_gradient)
            dones = np.stack(dones)
            is_prev_noise = np.concatenate(is_prev_noise, axis=0)
            is_prev_noise_batch = np.stack(is_prev_noise_batch)
            rewards = np.concatenate(rewards, axis=0)
            actions = np.concatenate(actions, axis=0)
            policies = np.concatenate(policies, axis=0)
            prev_obs = np.concatenate(prev_obs, axis=0)
            prev_actions = np.concatenate(prev_actions, axis=0)
            prev_policies = np.concatenate(prev_policies, axis=0)

            obs = self._process_observations(obs)
            obs_next = self._process_observations(obs_next)
            obs_first = self._process_observations(obs_first)
            obs_gradient = self._process_observations(obs_gradient)
            obs_first_gradient = self._process_observations(obs_first_gradient)
            prev_obs = self._process_observations(prev_obs)
            rewards_flatten = self._process_rewards(rewards)

            yield (
                obs,
                obs_next,
                obs_first,
                actions,
                policies,
                rewards_flatten,
                dones,
                lengths,
                big_batches_lengths,
                is_prev_noise,
                is_prev_noise_batch,

                prev_obs,
                prev_actions,
                prev_policies,

                obs_gradient,
                obs_first_gradient,
                lengths_gradient
            )