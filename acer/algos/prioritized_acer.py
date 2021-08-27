import gym
import numpy as np
import tensorflow as tf

from algos.fast_acer import FastACER
from replay_buffer import BufferFieldSpec, MultiPrioritizedReplayBuffer


class PrioritizedACER(FastACER):
    def __init__(self, *args, levels=2, block_size=256, update_blocks=1, **kwargs) -> None:
        self._levels = levels
        self._block_size = block_size
        self._update_blocks = update_blocks

        super().__init__(*args, **kwargs, additional_buffer_types=(tf.dtypes.float32,))
        
        self._priority_update_loader = tf.data.Dataset.from_generator(
            self._priorities_update_generator,
            (tf.dtypes.float32, self._actor.action_dtype, tf.dtypes.float32, tf.dtypes.int32)
        ).prefetch(1)

    def learn(self):
        """
        Performs experience replay learning. Experience trajectory is sampled from every replay buffer once, thus
        single backwards pass batch consists of 'num_parallel_envs' trajectories.

        Every call executes N of backwards passes, where: N = min(c0 * time_step / num_parallel_envs, c).
        That means at the beginning experience replay intensity increases linearly with number of samples
        collected till c value is reached.
        """


        if self._time_step > self._learning_starts:
            if self._memory.should_update_block():
                for block in self._priority_update_loader.take(self._update_blocks):
                    priorities = self._calculate_priorities(*block)
                    self._memory.update_block(priorities.numpy())
                    
                    with tf.name_scope('priorities'):
                        tf.summary.scalar('priorities_mean', tf.reduce_mean(priorities), step=self._tf_time_step)
                        tf.summary.scalar('priorities_std', tf.math.reduce_std(priorities), step=self._tf_time_step)

            experience_replay_iterations = min([round(self._c0 * self._time_step), self._c])
            
            for batch in self._data_loader.take(experience_replay_iterations):
                self._learn_from_experience_batch(*batch)

    def _init_replay_buffer(self, memory_size: int, policy_spec: BufferFieldSpec = None):
        if type(self._actions_space) == gym.spaces.Discrete:
            self._actions_shape = (1, )
        else:
            self._actions_shape = self._actions_space.shape

        self._memory = MultiPrioritizedReplayBuffer(
            action_spec=BufferFieldSpec(shape=self._actions_shape, dtype=self._actor.action_dtype_np),
            obs_spec=BufferFieldSpec(shape=self._observations_space.shape, dtype=self._observations_space.dtype),
            max_size=memory_size,
            policy_spec=policy_spec,
            num_buffers=self._num_parallel_envs,
            levels=self._levels, block=self._block_size
        )

    @tf.function(experimental_relax_shapes=True)
    def _learn_from_experience_batch(self, obs, obs_next, actions, old_policies,
                                     rewards, first_obs, first_actions, dones, lengths,
                                     priorities):

        obs = self._process_observations(obs)
        obs_next = self._process_observations(obs_next)
        rewards = self._process_rewards(rewards)
        
        policies, _ = self._actor.prob(obs, actions)

        mask = tf.sequence_mask(lengths, maxlen=self._n, dtype=tf.float32)

        td = self._calculate_td(obs, obs_next, rewards, lengths, dones, mask)
        truncated_density = self._calculate_density(policies, old_policies, mask) * tf.reshape(priorities, (-1, 1))

        with tf.name_scope('density'):
            tf.summary.scalar('density_mean', tf.reduce_mean(truncated_density), step=self._tf_time_step)
            tf.summary.scalar('density_std', tf.math.reduce_std(truncated_density), step=self._tf_time_step)

        d = tf.stop_gradient(td * truncated_density)

        self._actor_backward_pass(first_obs, first_actions, d)
        self._critic_backward_pass(first_obs, d)

    @tf.function(experimental_relax_shapes=True)
    def _calculate_density(self, policies, old_policies, mask):
        policies_masked = policies * mask + (1 - mask) * tf.ones_like(policies)
        old_policies_masked = old_policies * mask + (1 - mask) * tf.ones_like(old_policies)

        policies_ratio = policies_masked / old_policies_masked
        policies_ratio_prod = tf.reduce_prod(policies_ratio, axis=-1, keepdims=True)

        if self._truncate:
            return tf.tanh(policies_ratio_prod / self._b) * self._b
        else:
            return policies_ratio_prod

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
            priorities = offline_batches['priors']

            yield (
                obs,
                obs_next,
                actions,
                policies[:,:,0],
                rewards,
                obs[:,0],
                actions[:,0],
                dones,
                lengths,
                priorities[:,0],
            )

    @tf.function(experimental_relax_shapes=True)
    def _calculate_priorities(self, obs, actions, old_policies, lengths):
        obs = self._process_observations(obs)
        policies, _ = self._actor.prob(obs, actions)

        mask = tf.sequence_mask(lengths, maxlen=self._n, dtype=tf.float32)

        return tf.reshape(self._calculate_density(policies, old_policies, mask), (-1,))

    def _priorities_update_generator(self):
        """Generates trajectories batches. All tensors are padded with zeros to match self._n number of
        experience tuples in a single trajectory.
        Trajectories are returned in shape [batch, self._n, <obs/actions/etc shape>]
        """
        while True:
            offline_batches, lens = self._memory.get_next_block_to_update(self._n)
            
            lengths = lens
            obs = offline_batches['observations']
            actions = offline_batches['actions']
            policies = offline_batches['policies']

            yield (
                obs,
                actions,
                policies[:,:,0],
                lengths,
            )
