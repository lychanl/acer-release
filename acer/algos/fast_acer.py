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
from prioritized_buffer import MultiPrioritizedReplayBuffer


BUFFERS = {
    'simple': (MultiReplayBuffer, {'buffer_class': VecReplayBuffer}),
    'prioritized': (MultiPrioritizedReplayBuffer, {})
}

DATA_FIELDS = ('lengths', 'obs', 'obs_next', 'actions', 'policies', 'rewards', 'dones', 'priorities')


class FastACER(BaseACERAgent):

    def __init__(self, observations_space: gym.Space, actions_space: gym.Space, actor_layers: Optional[Tuple[int]],
                 critic_layers: Optional[Tuple[int]], b: float = 3, no_truncate: bool = True,
                 update_blocks=1, buffer_type='simple', log_values=(), *args, **kwargs):
        """BaseActor-Critic with Experience Replay

        TODO: finish docstrings
        """

        self._buffer_args = {}
        for key, value in kwargs.items():
            if key.startswith('buffer.'):
                self._buffer_args[key[len('buffer.'):]] = value

        self._actor_args = {}
        for key, value in kwargs.items():
            if key.startswith('actor.'):
                self._actor_args[key[len('actor.'):]] = value

        self._update_blocks = update_blocks
        self._buffer_type=buffer_type

        self._b = b
        self._truncate = not no_truncate

        self.PREPROCESSING = {
            'obs': self._process_observations,
            'obs_next': self._process_observations,
            'rewards': self._process_rewards
        }

        self.LOG_GATHER = {
            'mean': tf.reduce_mean,
            'std': tf.math.reduce_std
        }

        self._log_values = []
        for log_value in log_values:
            vg = log_value.split(':')
            val = vg[0]
            gather = vg[1] if len(log_values) > 1 else lambda x: x

            self._log_values.append((log_value, val, gather))

        super().__init__(observations_space, actions_space, actor_layers, critic_layers, *args, **kwargs)

    def _init_automodel(self):
        self.register_method("mask", self._calculate_mask, {"lengths": "lengths", "n": "memory_params.n"})
        self.register_method('td', self._calculate_td, {
            "values": "critic.value",
            "values_next": "critic.value_next",
            "rewards": "rewards",
            "lengths": "lengths",
            "mask": "base.mask",
            "dones": "dones",
            "n": "memory_params.n"
        })
        self.register_method('weighted_td', self._calculate_weighted_td, {
            "td": "base.td",
            "weights": "actor.sample_weights",
        })
        self.register_method('time_step', lambda: self._tf_time_step, {})

        self.register_component('actor', self._actor)
        self.register_component('critic', self._critic)
        self.register_component('memory_params', self._memory.parameters)
        self.register_component('base', self)

        self._call_list, self._call_list_data = self.prepare_default_call_list(DATA_FIELDS)

        if self._memory.priority:
            self.register_method('memory_priority', *self._memory.priority)
            self._memory_call_list, self._memory_call_list_data = self.prepare_call_list(['base.memory_priority'], DATA_FIELDS)
            replay_gen, dtypes = self._get_experience_replay_generator(seq=True, fields=self._memory_call_list_data)
            self._buffer_update_loader = tf.data.Dataset.from_generator(
                replay_gen, dtypes).prefetch(1)

    def _init_actor(self) -> BaseActor:
        if self._is_discrete:
            return CategoricalActor(
                self._observations_space, self._actions_space, self._actor_layers,
                self._actor_beta_penalty, self._tf_time_step, truncate=self._truncate, b=self._b,
                **self._actor_args
            )
        else:
            return GaussianActor(
                self._observations_space, self._actions_space, self._actor_layers,
                self._actor_beta_penalty, self._actions_bound, self._std, self._tf_time_step, truncate=self._truncate, b=self._b,
                **self._actor_args
            )

    def _init_data_loader(self, _) -> None:
        gen, dtypes = self._get_experience_replay_generator(fields=self._call_list_data)
        self._data_loader = tf.data.Dataset.from_generator(gen, dtypes)

    def _init_replay_buffer(self, memory_size: int, policy_spec: BufferFieldSpec = None):
        if type(self._actions_space) == gym.spaces.Discrete:
            self._actions_shape = (1, )
        else:
            self._actions_shape = self._actions_space.shape

        buffer_cls, buffer_base_args = BUFFERS[self._buffer_type]

        self._memory = buffer_cls(
            action_spec=BufferFieldSpec(shape=self._actions_shape, dtype=self._actor.action_dtype_np),
            obs_spec=BufferFieldSpec(shape=self._observations_space.shape, dtype=self._observations_space.dtype),
            max_size=memory_size,
            policy_spec=policy_spec,
            num_buffers=self._num_parallel_envs,
            **buffer_base_args,
            **self._buffer_args
        )

    def _calculate_mask(self, lengths, n):
        return tf.sequence_mask(lengths, maxlen=n, dtype=tf.float32)

    def _init_critic(self) -> Critic:
        # if self._is_obs_discrete:
        #     return TabularCritic(self._observations_space, None, self._tf_time_step)
        # else:
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
            if self._memory.should_update_block():
                for batch in self._buffer_update_loader.take(self._update_blocks):
                    data = {f: d for f, d in zip(self._memory_call_list_data, batch)}
                    priorities = self._calculate_memory_update(data)
                    self._memory.update_block(priorities.numpy())
                    
                    with tf.name_scope('priorities'):
                        tf.summary.scalar('priorities_mean', tf.reduce_mean(priorities), step=self._tf_time_step)
                        tf.summary.scalar('priorities_std', tf.math.reduce_std(priorities), step=self._tf_time_step)

            experience_replay_iterations = min([round(self._c0 * self._time_step), self._c])
            
            for batch in self._data_loader.take(experience_replay_iterations):
                data = {f: d for f, d in zip(self._call_list_data, batch)}
                self._learn_from_experience_batch(data)
    
    @tf.function(experimental_relax_shapes=True)
    def _calculate_memory_update(self, data):
        data = self.call_list(self._memory_call_list, data, self.PREPROCESSING)

        with tf.name_scope('memory_update_log'):
            for name, value, gather in self._log_values:
                tf.summary.scalar(name, gather(data[value]), self._tf_time_step)

        return data['base.memory_priority']

    @tf.function(experimental_relax_shapes=True)
    def _learn_from_experience_batch(self, data):
        data = self.call_list(self._call_list, data, self.PREPROCESSING)

        with tf.name_scope('log'):
            for name, value, gather in self._log_values:
                tf.summary.scalar(name, gather(data[value]), self._tf_time_step)


    @tf.function(experimental_relax_shapes=True)
    def _calculate_td(self, values, values_next, rewards, lengths, dones, mask, n):
        dones_mask = 1 - tf.cast(
            (tf.expand_dims(tf.range(1, n + 1), 0) == tf.expand_dims(lengths, 1)) & tf.expand_dims(dones, 1),
            tf.float32
        )

        values = values[:,:,0]

        # concat works despite possibly different trajectory lengths because of memory_buffer implementation
        # TODO make it independant from memory_buffer implementation
        values_with_next = tf.concat([values[:,1:], values_next], axis=1)

        next_mask = tf.cast(tf.expand_dims(tf.range(1, n + 1), 0) == tf.expand_dims(lengths, 1), tf.float32)
        values_next = ((1 - next_mask) * values_with_next + next_mask * values_next) * dones_mask

        gamma_coeffs_masked = tf.expand_dims(tf.pow(self._gamma, tf.range(1., n + 1)), axis=0) * mask

        td_parts = rewards + self._gamma * values_next - values
        td_rewards = tf.reduce_sum(td_parts * gamma_coeffs_masked, axis=1, keepdims=True)

        return td_rewards

    @tf.function(experimental_relax_shapes=True)
    def _calculate_weighted_td(self, td, weights):
        return tf.stop_gradient(td * weights)

    def _fetch_offline_batch(self) -> List[Dict[str, Union[np.array, list]]]:
        return self._memory.get_vec(self._batches_per_env, self._memory.n)

    def _get_experience_replay_generator(
            self, seq=False, fields=DATA_FIELDS):
        batch_size_ids = np.arange(self._batch_size)
        specs = {
            'lengths': ('lengths', lambda x, lens: x),
            'obs': ('observations', lambda x, lens: x),
            'obs_next': ('next_observations', lambda x, lens: x[batch_size_ids, lens - 1]),
            'actions': ('actions', lambda x, lens: x),
            'policies': ('policies', lambda x, lens: x[:, :, 0]),
            'rewards': ('rewards', lambda x, lens: x),
            'dones': ('dones', lambda x, lens: x[batch_size_ids, lens - 1]),
            'priorities': ('priors', lambda x, lens: x[:, 0]),
            'time': ('time', lambda x, lens: x),
        }

        dtypes = {
            'lengths': tf.int32,
            'obs': tf.int32 if self._is_obs_discrete else tf.float32,
            'obs_next': tf.int32 if self._is_obs_discrete else tf.float32,
            'actions': self._actor.action_dtype,
            'policies': tf.float32,
            'rewards': tf.float32,
            'dones': tf.bool,
            'priorities': tf.float32,
            'time': tf.int32,
        }

        field_specs = [specs[f] for f in fields]
        field_dtypes = tuple(dtypes[f] for f in fields)

        def experience_replay_generator():
            while True:
                batch, lens = self._memory.get_next_block_to_update() if seq else self._fetch_offline_batch()
                batch["lengths"] = lens
                data = tuple(preproc(batch[field], lens) for field, preproc in field_specs)
                yield data

        return experience_replay_generator, field_dtypes
