"""
BaseActor-Critic with Experience Replay algorithm.
Implements the algorithm from:

(1)
Wawrzyński P, Tanwani AK. Autonomous reinforcement learning with experience replay.
Neural Networks : the Official Journal of the International Neural Network Society.
2013 May;41:156-167. DOI: 10.1016/j.neunet.2012.11.007.

(2)
Wawrzyński, Paweł. "Real-time reinforcement learning by sequential actor–critics
and experience replay." Neural Networks 22.10 (2009): 1484-1497.
"""
from typing import Optional, List, Union, Dict, Tuple
import gym
import tensorflow as tf
import numpy as np

from algos.base import BaseACERAgent, BaseActor, CategoricalActor, GaussianActor, Critic
from replay_buffer import BufferFieldSpec, VecReplayBuffer, MultiReplayBuffer


class FastACER(BaseACERAgent):
    def __init__(self, observations_space: gym.Space, actions_space: gym.Space, actor_layers: Optional[Tuple[int]],
                 critic_layers: Optional[Tuple[int]], n: int = 2, b: float = 3, no_truncate: bool = True, *args, **kwargs):
        """BaseActor-Critic with Experience Replay

        TODO: finish docstrings
        """

        super().__init__(observations_space, actions_space, actor_layers, critic_layers, *args, **kwargs)
        self._n = n
        self._b = b
        self._truncate = not no_truncate

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

    def _init_replay_buffer(self, memory_size: int, policy_spec: BufferFieldSpec = None):
        if type(self._actions_space) == gym.spaces.Discrete:
            self._actions_shape = (1, )
        else:
            self._actions_shape = self._actions_space.shape

        self._memory = MultiReplayBuffer(
            buffer_class=VecReplayBuffer,
            action_spec=BufferFieldSpec(shape=self._actions_shape, dtype=self._actor.action_dtype_np),
            obs_spec=BufferFieldSpec(shape=self._observations_space.shape, dtype=self._observations_space.dtype),
            max_size=memory_size,
            policy_spec=policy_spec,
            num_buffers=self._num_parallel_envs
        )

    def _init_critic(self) -> Critic:
        return Critic(self._observations_space, self._critic_layers, self._tf_time_step)

    def learn(self):
        """
        Performs experience replay learning. Experience trajectory is sampled from every replay buffer once, thus
        single backwards pass batch consists of 'num_parallel_envs' trajectories.

        Every call executes N of backwards passes, where: N = min(c0 * time_step / num_parallel_envs, c).
        That means at the beginning experience replay intensity increases linearly with number of samples
        collected till c value is reached.
        """
        if self._time_step > self._learning_starts:
            experience_replay_iterations = min([round(self._c0 * self._time_step), self._c])
            
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
        
        """
        flat_obs = tf.reshape(obs, (-1, *self._observations_space.shape))

        policies, _ = self._actor.prob(
            flat_obs,
            tf.reshape(actions, (-1, *self._actions_shape)),
        )
        policies = tf.reshape(policies, (self._batch_size, self._n))
        """
        policies, _ = self._actor.prob(obs, actions)

        mask = tf.sequence_mask(lengths, maxlen=self._n, dtype=tf.float32)

        td = self._calculate_td(obs, obs_next, rewards, lengths, dones, mask)
        truncated_density = self._calculate_truncated_density(policies, old_policies, mask)

        with tf.name_scope('density'):
            tf.summary.scalar('density_mean', tf.reduce_mean(truncated_density), step=self._tf_time_step)
            tf.summary.scalar('density_std', tf.math.reduce_std(truncated_density), step=self._tf_time_step)

        d = tf.stop_gradient(td * truncated_density)

        self._actor_backward_pass(first_obs, first_actions, d)
        self._critic_backward_pass(first_obs, d)

    @tf.function(experimental_relax_shapes=True)
    def _calculate_truncated_density(self, policies, old_policies, mask):
        policies_masked = policies * mask + (1 - mask) * tf.ones_like(policies)
        old_policies_masked = old_policies * mask + (1 - mask) * tf.ones_like(old_policies)

        policies_ratio = policies_masked / old_policies_masked
        policies_ratio_prod = tf.reduce_prod(policies_ratio, axis=-1, keepdims=True)

        if self._truncate:
            return tf.tanh(policies_ratio_prod / self._b) * self._b
        else:
            return policies_ratio_prod

    @tf.function(experimental_relax_shapes=True)
    def _calculate_td(self, obs, obs_next, rewards, lengths, dones, mask):
        dones_mask = 1 - tf.cast(
            (tf.expand_dims(tf.range(1, self._n + 1), 0) == tf.expand_dims(lengths, 1)) & tf.expand_dims(dones, 1),
            tf.float32
        )

        """
        all_obs = tf.concat([obs, tf.expand_dims(obs_next, 1)], axis=1)
        values_o = tf.reshape(
            self._critic.value(tf.reshape(all_obs, (-1, *self._observations_space.shape))),
            (self._batch_size, self._n + 1)
        )
        """

        values = tf.squeeze(self._critic.value(tf.concat([obs, tf.expand_dims(obs_next, 1)], axis=1)), axis=2)
        
        values_first = values[:,:-1]
        values_next = values[:,1:] * dones_mask

        gamma_coeffs_masked = tf.expand_dims(tf.pow(self._gamma, tf.range(1., self._n + 1)), axis=0) * mask

        td_parts = rewards + self._gamma * values_next - values_first
        td_rewards = tf.reduce_sum(td_parts * gamma_coeffs_masked, axis=1, keepdims=True)

        return td_rewards

        #  values_next_discounted = tf.expand_dims(tf.pow(self._gamma, tf.cast(lengths, tf.float32)), 1) * values_next

        #  return (-values_first + td_rewards + values_next_discounted)

    def _actor_backward_pass(self, observations: tf.Tensor, actions: tf.Tensor, d: tf.Tensor):
        with tf.GradientTape() as tape:
            loss = self._actor.loss(observations, actions, d)
        grads = tape.gradient(loss, self._actor.trainable_variables)
        if self._gradient_norm is not None:
            grads = self._clip_gradient(grads, self._actor_gradient_norm_median, 'actor')
        gradients = zip(grads, self._actor.trainable_variables)

        self._actor_optimizer.apply_gradients(gradients)

    def _critic_backward_pass(self, observations: tf.Tensor, d: tf.Tensor):
        with tf.GradientTape() as tape:
            loss = self._critic.loss(observations, d)
        grads = tape.gradient(loss, self._critic.trainable_variables)
        if self._gradient_norm is not None:
            grads = self._clip_gradient(grads, self._critic_gradient_norm_median, 'critic')
        gradients = zip(grads, self._critic.trainable_variables)

        self._critic_optimizer.apply_gradients(gradients)

    def _fetch_offline_batch(self) -> List[Dict[str, Union[np.array, list]]]:
        return self._memory.get_vec(self._batches_per_env, self._n)

    def _experience_replay_generator(self):
        """Generates trajectories batches. All tensors are padded with zeros to match self._n number of
        experience tuples in a single trajectory.
        Trajectories are returned in shape [batch, self._n, <obs/actions/etc shape>]
        """
        while True:
            offline_batches, lens = self._fetch_offline_batch()
            
            lengths = lens
            obs = offline_batches['observations']
            obs_next = offline_batches['next_observations'][np.arange(self._batch_size),lens - 1]
            actions = offline_batches['actions']
            rewards = offline_batches['rewards']
            policies = offline_batches['policies']
            dones = offline_batches['dones'][np.arange(self._batch_size),lens - 1]

            yield (
                obs,
                obs_next,
                actions,
                policies[:,:,0],
                rewards,
                obs[:,0],
                actions[:,0],
                dones,
                lengths
            )
