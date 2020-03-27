from collections import deque
from typing import Optional, List, Union, Dict, Tuple
# import os
# os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import gym
import tensorflow_probability as tfp
import tensorflow as tf
import numpy as np


from algos.acerac_single import ACERACSingle
from replay_buffer import MultiWindowReplayBuffer, BufferFieldSpec


class ACERAC(ACERACSingle):
    def __init__(self, observations_space: gym.Space, actions_space: gym.Space, actor_layers: Optional[Tuple[int]],
                 critic_layers: Optional[Tuple[int]], lam: float = 0.1, b: float = 3, tau: int = 2, *args, **kwargs):

        super().__init__(observations_space, actions_space, actor_layers, critic_layers, lam, b, tau, *args, **kwargs)

    def _init_replay_buffer(self, memory_size: int):
        if type(self._actions_space) == gym.spaces.Discrete:
            actions_shape = (1, )
        else:
            actions_shape = self._actions_space.shape

        self._memory = MultiWindowReplayBuffer(
            action_spec=BufferFieldSpec(shape=actions_shape, dtype=self._actor.action_dtype_np),
            obs_spec=BufferFieldSpec(shape=self._observations_space.shape, dtype=self._observations_space.dtype),
            max_size=memory_size,
            num_buffers=self._num_parallel_envs
        )

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
                                     rewards, middle_obs, middle_actions, dones, lengths, rewards_lengths):

        c_det = tf.gather(self._c_dets, lengths - 1)
        c_invs = tf.gather(self._c_invs, lengths - 1).to_tensor()

        batches_indices = tf.RaggedTensor.from_row_lengths(values=tf.range(tf.reduce_sum(lengths)), row_lengths=lengths)
        batch_indices_rewards = tf.RaggedTensor.from_row_lengths(values=tf.range(tf.reduce_sum(rewards_lengths)), row_lengths=rewards_lengths)
        values_last = tf.squeeze(self._critic.value(obs_next)) * (1.0 - tf.cast(dones, tf.dtypes.float32))
        policies, _ = tf.split(self._actor.prob(obs, actions), 2, axis=0)
        indices = tf.expand_dims(batches_indices, axis=2)
        rewards_indices = tf.expand_dims(batch_indices_rewards, axis=2)

        actions_flatten = tf.squeeze(tf.gather_nd(actions, tf.expand_dims(indices, axis=2)), axis=2).merge_dims(1, 2)
        old_means_flatten = tf.squeeze(tf.gather_nd(old_policies, tf.expand_dims(indices, axis=2)), axis=2).merge_dims(1, 2)
        rewards_flatten = tf.squeeze(tf.gather_nd(rewards, tf.expand_dims(rewards_indices, axis=2)), axis=-1)

        with tf.GradientTape(persistent=True) as tape:
            print()
            means = self._actor.act_deterministic(obs)
            values_middle = tf.squeeze(self._critic.value(middle_obs))
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
            d = truncated_density * (-tf.squeeze(values_middle) + td_return + tf.pow(self._gamma, tf.cast(rewards_lengths, tf.float32)) * values_last)

            density = (1 / tf.sqrt((tf.pow(np.pi, tf.cast(means_flatten.row_lengths(), tf.float32)) * c_det))) * tf.squeeze(tf.exp(-0.5 * exp_current))
            bounds_penalty = tf.reduce_mean(
                tf.scalar_mul(
                    self._actor._beta_penalty,
                    tf.square(tf.maximum(0.0, tf.abs(means) - self._actions_bound))
                )
            )
            actor_loss = tf.reduce_mean(-tf.math.log(density + 1e-20) * tf.stop_gradient(d)) + bounds_penalty
            critic_loss = -tf.reduce_mean(values_middle * tf.stop_gradient(d))

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
        trajectory_lens = [[np.random.randint(0, self._tau) for _ in range(self._num_parallel_envs)] for _ in range(self._batches_per_env)]
        batch = []
        [batch.extend(self._memory.get(trajectories)) for trajectories in trajectory_lens]
        return batch

    def _experience_replay_generator(self):
        while True:
            offline_batches = self._fetch_offline_batch()

            obs, obs_next, actions, policies, rewards, dones, lengths, rewards_lengths, c, c_inv = [], [], [], [], [], [], [], [], [], []
            middle_obs, middle_actions = [], []
            for batch, i in offline_batches:
                obs.append(batch['observations'])
                obs_next.append(batch['next_observations'][batch['next_observations'].shape[0] - 1])
                actions.append(batch['actions'])
                policies.append(batch['policies'])
                rewards.append(batch['rewards'][i:])
                dones.append(batch['dones'][batch['observations'].shape[0] - 1])
                lengths.append(len(batch['observations']))
                rewards_lengths.append(len(rewards[-1]))
                middle_obs.append(batch['observations'][i])
                middle_actions.append(batch['actions'][i])

            obs = np.concatenate(obs, axis=0)
            obs_next = np.stack(obs_next)
            dones = np.stack(dones)
            rewards = np.concatenate(rewards, axis=0)
            actions = np.concatenate(actions, axis=0)
            policies = np.concatenate(policies, axis=0)
            middle_actions = np.stack(middle_actions)
            middle_obs = np.stack(middle_obs)
            obs_flatten = self._process_observations(obs)
            obs_next_flatten = self._process_observations(obs_next)
            rewards_flatten = self._process_rewards(rewards)

            yield (
                obs_flatten,
                obs_next_flatten,
                actions,
                policies,
                rewards_flatten,
                middle_obs,
                middle_actions,
                dones,
                lengths,
                rewards_lengths
            )