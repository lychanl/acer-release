from collections import deque
from typing import Optional, List, Union, Dict, Tuple
# import os
# os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import gym
import tensorflow_probability as tfp
import tensorflow as tf
import numpy as np

from algos.base import GaussianActor, BaseActor, Critic, BaseACERAgent
from replay_buffer import WindowReplayBuffer, MultiReplayBuffer, BufferFieldSpec


class OldACERAC(BaseACERAgent):
    def __init__(self, observations_space: gym.Space, actions_space: gym.Space, actor_layers: Optional[Tuple[int]],
                 critic_layers: Optional[Tuple[int]], lam: float = 0.1, b: float = 3, tau: int = 2, alpha: float = 0.8,
                 noise_type: str = 'mean', *args, **kwargs):

        self._tau = tau
        self._alpha = alpha
        self._noise_type = noise_type
        super().__init__(observations_space, actions_space, actor_layers, critic_layers, *args, **kwargs)
        self._lam = lam
        self._b = b

        self._cov_matrix = tf.linalg.diag(tf.square(tf.exp(self._actor.log_std)))

        self._c_invs = []
        self._c_dets = []

        for i in range(1, int((self._tau * 2)) + 2):
            if self._noise_type == 'mean':
                window_range = tf.range(i)
                toeplitz_mat = tf.cast(tf.linalg.LinearOperatorToeplitz(window_range, window_range).to_dense(), tf.float32)
                Lambda = tf.maximum((self._tau - toeplitz_mat) / self._tau, 0)
            else:
                window_range = tf.range(i)
                ones = tf.ones(shape=[i, i]) * self._alpha
                toeplitz_mat = tf.cast(tf.linalg.LinearOperatorToeplitz(window_range, window_range).to_dense(),
                                       tf.float32)
                Lambda = tf.pow(ones, toeplitz_mat)
            operator_1 = tf.linalg.LinearOperatorFullMatrix(Lambda)
            operator_2 = tf.linalg.LinearOperatorFullMatrix(self._cov_matrix)
            c = tf.linalg.LinearOperatorKronecker([operator_1, operator_2]).to_dense()
            self._c_dets.append((tf.linalg.det(c) + 1e-20).numpy())
            c_inv = tf.linalg.inv(c)
            self._c_invs.append(c_inv.numpy())

        self._c_dets = tf.constant(self._c_dets)
        self._c_invs = tf.ragged.constant(self._c_invs)

        self._data_loader = tf.data.Dataset.from_generator(
            self._experience_replay_generator,
            (tf.dtypes.float32, tf.dtypes.float32, self._actor.action_dtype, tf.dtypes.float32, tf.dtypes.float32,
             tf.dtypes.float32, tf.dtypes.bool, tf.dtypes.int32, tf.dtypes.int32,
             tf.dtypes.int32, tf.dtypes.float32, tf.dtypes.float32, tf.dtypes.int32)
        ).prefetch(2)

    def _init_replay_buffer(self, memory_size: int):
        if type(self._actions_space) == gym.spaces.Discrete:
            actions_shape = (1,)
        else:
            actions_shape = self._actions_space.shape

        self._memory = MultiReplayBuffer(
            action_spec=BufferFieldSpec(shape=actions_shape, dtype=self._actor.action_dtype_np),
            obs_spec=BufferFieldSpec(shape=self._observations_space.shape, dtype=self._observations_space.dtype),
            max_size=memory_size,
            num_buffers=self._num_parallel_envs,
            buffer_class=WindowReplayBuffer
        )

    def _init_actor(self) -> BaseActor:
        if self._is_discrete:
            raise NotImplementedError
        else:
            return NoiseGaussianActor(
                self._observations_space, self._actions_space, self._actor_layers,
                self._actor_beta_penalty, self._actions_bound, self._tau, self._alpha, self._num_parallel_envs,
                self._noise_type, self._std, self._tf_time_step
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
    def _learn_from_experience_batch(self, obs, obs_next, actions, old_policies,
                                     rewards, middle_obs, dones, lengths,
                                     rewards_lengths, big_batches_lengths, obs_gradient, obs_gradient_middle,
                                     lengths_gradient):
        c_invs = tf.gather(self._c_invs, lengths - 1).to_tensor()

        batches_indices = tf.RaggedTensor.from_row_lengths(
            values=tf.range(tf.reduce_sum(lengths)), row_lengths=lengths
        )
        batches_indices_gradients = tf.expand_dims(tf.RaggedTensor.from_row_lengths(
            values=tf.range(tf.reduce_sum(lengths_gradient)), row_lengths=lengths_gradient
        ), 2)
        big_batches_indices = tf.expand_dims(tf.RaggedTensor.from_row_lengths(
            values=tf.range(tf.reduce_sum(big_batches_lengths)), row_lengths=big_batches_lengths
        ), 2)

        values_middle = tf.squeeze(self._critic.value(middle_obs))
        batch_indices_rewards = tf.RaggedTensor.from_row_lengths(values=tf.range(tf.reduce_sum(rewards_lengths)), row_lengths=rewards_lengths)
        values_last = tf.squeeze(self._critic.value(obs_next)) * (1.0 - tf.cast(dones, tf.dtypes.float32))
        indices = tf.expand_dims(batches_indices, axis=2)
        rewards_indices = tf.expand_dims(batch_indices_rewards, axis=2)

        actions_flatten = tf.squeeze(tf.gather_nd(actions, tf.expand_dims(indices, axis=2)), axis=2).merge_dims(1, 2)
        old_means_flatten = tf.squeeze(tf.gather_nd(old_policies, tf.expand_dims(indices, axis=2)), axis=2).merge_dims(1, 2)
        rewards_flatten = tf.squeeze(tf.gather_nd(rewards, tf.expand_dims(rewards_indices, axis=2)), axis=-1)

        with tf.GradientTape(persistent=True) as tape:
            means = self._actor.act_deterministic(obs)
            means_gradient = self._actor.act_deterministic(obs_gradient)
            values_gradient = tf.squeeze(self._critic.value(obs_gradient_middle))

            means_flatten = tf.squeeze(tf.gather_nd(means, tf.expand_dims(indices, axis=2)), axis=2).merge_dims(1, 2)
            actions_mu_diff_current = tf.expand_dims((actions_flatten - means_flatten).to_tensor(), 1)
            actions_mu_diff_old = tf.expand_dims((actions_flatten - old_means_flatten).to_tensor(), 1)
            exp_current = tf.matmul(tf.matmul(actions_mu_diff_current, c_invs), tf.transpose(actions_mu_diff_current, [0, 2, 1]))
            exp_old = tf.matmul(tf.matmul(actions_mu_diff_old, c_invs), tf.transpose(actions_mu_diff_old, [0, 2, 1]))
            density_ratio = tf.squeeze(tf.exp(-0.5 * exp_current + 0.5 * exp_old))
            truncated_density = tf.tanh(density_ratio / self._b) * self._b

            batch_mask = tf.sequence_mask(rewards_flatten.row_lengths())
            gamma_coeffs_batches = tf.ones_like(rewards_flatten).to_tensor() * self._gamma
            gamma_coeffs = tf.ragged.boolean_mask(
                tf.math.cumprod(gamma_coeffs_batches, axis=1, exclusive=True),
                batch_mask
            )
            td_return = tf.reduce_sum(rewards_flatten * gamma_coeffs.with_row_splits_dtype(tf.int32), axis=1)
            d = truncated_density * (-values_middle
                                     + td_return
                                     + tf.pow(self._gamma, tf.cast(rewards_lengths, tf.float32)) * values_last)

            means_gradient_batches = tf.squeeze(
                tf.gather_nd(means_gradient, tf.expand_dims(batches_indices_gradients, axis=2)),
                axis=2
            ).to_tensor()
            means_gradient_batches = tf.reshape(means_gradient_batches, shape=[tf.shape(means_gradient_batches)[0], -1])

            c_mu = tf.matmul(c_invs, tf.transpose(actions_mu_diff_current, [0, 2, 1]))
            c_mu_di = c_mu * tf.expand_dims(tf.expand_dims(d, axis=1), 2)
            c_mu_di_batches = tf.squeeze(
                tf.gather_nd(c_mu_di, big_batches_indices).to_tensor()
            )
            c_mu_dim_sum = tf.stop_gradient(tf.reduce_sum(c_mu_di_batches, axis=1) / self._tau)

            d_batches = tf.squeeze(
                tf.gather_nd(d, big_batches_indices).to_tensor()
            )
            d_batches = tf.stop_gradient(tf.reduce_sum(d_batches, axis=1))

            bounds_penalty = tf.scalar_mul(
                    self._actor.beta_penalty,
                    tf.square(tf.maximum(0.0, tf.abs(means_gradient) - self._actions_bound))
            )

            bounds_penalty = tf.reduce_mean(
                tf.reduce_sum(
                    bounds_penalty,
                    axis=1
                ),
                0
            )
            actor_loss = tf.matmul(tf.expand_dims(means_gradient_batches, axis=1), tf.expand_dims(c_mu_dim_sum, axis=2))
            actor_loss = -tf.reduce_mean(tf.squeeze(actor_loss)) + bounds_penalty
            critic_loss = -tf.reduce_mean(values_gradient * d_batches / self._tau)

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
            tf.summary.scalar(f'batch_value_mean', tf.reduce_mean(values_middle), step=self._tf_time_step)

    def _fetch_offline_batch(self) -> List[Tuple[Dict[str, Union[np.array, list]], int]]:
        trajectory_lens = [[self._tau - 1 for _ in range(self._num_parallel_envs)] for _ in range(self._batches_per_env)]
        batch = []
        [batch.extend(self._memory.get(trajectories)) for trajectories in trajectory_lens]
        return batch

    def _experience_replay_generator(self):
        while True:
            offline_batches = self._fetch_offline_batch()

            obs, obs_next, actions, policies, rewards, dones, lengths, rewards_lengths, c, c_inv = [], [], [], [], [], [], [], [], [], []
            obs_gradient, actions_gradient, obs_gradient_middle, lengths_gradient = [], [], [], []
            middle_obs, middle_actions = [], []

            big_batches_lengths = []

            for batch, middle_index in offline_batches:
                batch_size = len(batch['observations'])
                big_batch_len = 1
                k = 1

                obs_gradient.append(batch['observations'])
                obs_gradient_middle.append([batch['observations'][middle_index]])
                actions_gradient.append(batch['actions'])
                lengths_gradient.append(len(obs_gradient[-1]))

                obs.append([batch['observations'][middle_index]])
                obs_next.append(batch['next_observations'][middle_index])
                actions.append([batch['actions'][middle_index]])
                policies.append([batch['policies'][middle_index]])
                rewards.append([batch['rewards'][middle_index]])
                dones.append(batch['dones'][middle_index])
                lengths.append(1)
                rewards_lengths.append(1)
                middle_obs.append(batch['observations'][middle_index])
                middle_actions.append(batch['actions'][middle_index])
                while 1:
                    start = max([0, middle_index - k])
                    end = min([middle_index + k + 1, batch_size])
                    obs.append(batch['observations'][start: end])
                    obs_next.append(batch['next_observations'][end - 1])
                    actions.append(batch['actions'][start: end])
                    policies.append(batch['policies'][start: end])
                    rewards.append(batch['rewards'][middle_index: end])
                    dones.append(batch['dones'][end - 1])
                    lengths.append(len(obs[-1]))
                    big_batch_len += 1
                    rewards_lengths.append(len(rewards[-1]))
                    middle_obs.append(batch['observations'][middle_index])
                    middle_actions.append(batch['actions'][middle_index])
                    if start == 0 and end == batch_size:
                        break
                    k += 1
                big_batches_lengths.append(big_batch_len)

            obs = np.concatenate(obs, axis=0)
            obs_next = np.stack(obs_next)
            dones = np.stack(dones)
            rewards = np.concatenate(rewards, axis=0)
            actions = np.concatenate(actions, axis=0)
            obs_gradient = np.concatenate(obs_gradient, axis=0)
            obs_gradient_middle = np.concatenate(obs_gradient_middle, axis=0)
            policies = np.concatenate(policies, axis=0)
            middle_obs = np.stack(middle_obs)

            obs_flatten = self._process_observations(obs)
            obs_gradient = self._process_observations(obs_gradient)
            obs_gradient_middle = self._process_observations(obs_gradient_middle)
            obs_next_flatten = self._process_observations(obs_next)
            rewards_flatten = self._process_rewards(rewards)

            yield (
                obs_flatten,
                obs_next_flatten,
                actions,
                policies,
                rewards_flatten,
                middle_obs,
                dones,
                lengths,
                rewards_lengths,
                big_batches_lengths,

                obs_gradient,
                obs_gradient_middle,
                lengths_gradient
            )