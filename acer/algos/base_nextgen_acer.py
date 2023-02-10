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

from algos.base import BaseACERAgent, BaseActor, Critic
from replay_buffer import BufferFieldSpec


def print_batch(batch):
    if batch is None:
        return

    def print_el(el):
        if len(np.shape(el)) == 0:
            if isinstance(el, float) or isinstance(el, np.ndarray) and el.dtype == np.float32:
                return f'{el:.2e}'
            return str(el)
        else:
            return f"[{','.join(map(print_el, el)) if np.shape(el)[0] < 10 else ','.join(map(print_el, el[:10])) + '...'}]"

    keys = []
    vals = []
    for k, v in batch.items():
        keys.append(k)
        vals.append(v.numpy())
    
    print(*keys, sep='\t')

    for row in zip(*vals):
        print(*map(print_el, row), sep='\t')


class BaseNextGenACERAgent(BaseACERAgent):
    DATA_FIELDS = ('lengths', 'obs', 'obs_next', 'actions', 'policies', 'rewards', 'dones', 'priorities')
    ACT_DATA_FIELDS = ('obs', 'actions')

    def __init__(self, observations_space: gym.Space, actions_space: gym.Space, actor_layers: Optional[Tuple[int]],
                 critic_layers: Optional[Tuple[int]], buffers={}, actors={}, critics={},
                 update_blocks=1, buffer_type='simple', log_values=(), log_memory_values=(), log_act_values=(),
                 log_to_file_values=(), log_to_file_act_values=(),
                 actor_type='simple', critic_type='simple', nan_guard=False, 
                 *args, **kwargs):
        """BaseActor-Critic with Experience Replay

        TODO: finish docstrings
        """
        self._actor_type = actor_type
        self._critic_type = critic_type

        self.BUFFERS = buffers
        self.ACTORS = actors
        self.CRITICS = critics

        self._buffer_args = {}
        for key, value in kwargs.items():
            if key.startswith('buffer.'):
                self._buffer_args[key[len('buffer.'):]] = value

        self._actor_args = {}
        for key, value in kwargs.items():
            if key.startswith('actor.'):
                self._actor_args[key[len('actor.'):]] = value

        self._critic_args = {}
        for key, value in kwargs.items():
            if key.startswith('critic.'):
                self._critic_args[key[len('critic.'):]] = value

        self._update_blocks = update_blocks
        self._buffer_type=buffer_type

        self._force_log = (
            [v[1:].split(':')[0] for v in log_values if v[0] == '!']
            + [v[1:].split(':')[0] for v in log_to_file_values if v[0] == '!']
        )
        self._force_log_memory = [v[1:].split(':')[0] for v in log_memory_values if v[0] == '!']

        super().__init__(observations_space, actions_space, actor_layers, critic_layers, *args, **kwargs)

        self.PREPROCESSING = {
            'obs': self._process_observations,
            'obs_next': self._process_observations,
            'rewards': self._process_rewards
        }

        self.LOG_GATHER = {
            'mean': tf.reduce_mean,
            'std': tf.math.reduce_std,
            'min': tf.reduce_min,
            'max': tf.reduce_max,
        }

        def prepare_log_values(spec):
            target_list = []
            for log_value in spec:
                vg = log_value.lstrip('!').split(':')
                val = vg[0]
                gather = self.LOG_GATHER[vg[1]] if len(vg) > 1 else lambda x: x

                target_list.append((log_value, val, gather))
            return target_list

        def check_log_values(log_values, call_list_data, call_list, where=""):
            for _, value, _ in log_values:
                assert value in call_list_data or value in [n for n, _, _ in call_list],\
                    f'Error: {value} not calculated {where}'


        self._log_values = prepare_log_values(log_values)
        self._log_memory_values = prepare_log_values(log_memory_values)
        self._log_to_file_values = prepare_log_values(log_to_file_values)
        self._log_act_values = prepare_log_values(log_act_values)
        self._log_to_file_act_values = prepare_log_values(log_to_file_act_values)
        self._nan_guard = nan_guard
        self._nan_log_prev_mem_batch = None
        self._nan_log_prev_batch = None

        self._init_log_act_automodel()
        check_log_values(self._log_values, self._call_list_data, self._call_list)
        check_log_values(self._log_to_file_values, self._call_list_data, self._call_list)
        check_log_values(self._log_memory_values, self._memory_call_list_data, self._memory_call_list, "in memory updates")

    def _init_automodel(self, skip=()):
        self.register_method("mask", self._calculate_mask, {"lengths": "lengths", "n": "memory_params.n"})
        self.register_method('time_step', lambda: self._tf_time_step, {})
        self.register_method('gamma', lambda: self._gamma, {})

        self.register_component('actor', self._actor)
        self.register_component('critic', self._critic)
        self.register_component('memory_params', self._memory.parameters)
        self.register_component('memory', self._memory)
        self.register_component('base', self)

        self._init_automodel_overrides()

        self._call_list, self._call_list_data = self.prepare_default_call_list(self.DATA_FIELDS, additional=self._force_log)

        if self._memory.priority:
            self.register_method('memory_priority', *self._memory.priority)
            self._memory_call_list, self._memory_call_list_data = self.prepare_call_list(
                ['base.memory_priority'] + self._force_log_memory, self.DATA_FIELDS)

            replay_gen, dtypes = self._get_experience_replay_generator(seq=True, fields=self._memory_call_list_data)
            self._buffer_update_loader = tf.data.Dataset.from_generator(
                replay_gen, dtypes)
        else:
            self._memory_call_list = self._memory_call_list_data = None

    def _init_log_act_automodel(self):
        if self._log_act_values or self._log_to_file_act_values:
            to_log = list({v for _, v, _ in self._log_act_values} | {v for _, v, _ in self._log_to_file_act_values})
            self._log_act_call_list, self._log_act_call_list_data = self.prepare_call_list(to_log, self.ACT_DATA_FIELDS)
        else:
            self._log_act_call_list = self._log_act_call_list_data = None

    def _init_automodel_overrides(self) -> None:
        pass

    def _init_actor(self) -> BaseActor:
        return self.ACTORS[self._actor_type][self._is_discrete](
            self._observations_space, self._actions_space, self._actor_layers,
            self._actor_beta_penalty, tf_time_step=self._tf_time_step,
            batch_size=self._batch_size, num_parallel_envs=self._num_parallel_envs, **self._actor_args
        )

    def _init_data_loader(self, _) -> None:
        gen, dtypes = self._get_experience_replay_generator(fields=self._call_list_data)
        self._data_loader = tf.data.Dataset.from_generator(gen, dtypes)

    def _init_replay_buffer(self, memory_size: int, policy_spec: BufferFieldSpec = None):
        if type(self._actions_space) == gym.spaces.Discrete:
            self._actions_shape = (1, )
        else:
            self._actions_shape = self._actions_space.shape

        buffer_cls, buffer_base_args = self.BUFFERS[self._buffer_type]

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
        return self.CRITICS[self._critic_type](self._observations_space, self._critic_layers, self._tf_time_step, **self._critic_args)

    t = None
    __data = []
    __learn = []
    __run = []

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
                    if self._nan_guard:
                        if not np.isfinite(priorities.numpy()).all():
                            print('NaN on memory update')
                            print('Last mem batch:')
                            print_batch(self._nan_log_prev_mem_batch)
                            print('Last batch:')
                            print_batch(self._nan_log_prev_batch)
                        self._nan_log_prev_mem_batch = data

            experience_replay_iterations = min([round(self._c0 * self._time_step), self._c])
            
            # if self.t is not None:
            #     self.__run.append((datetime.now() - self.t).total_seconds())

            # self.t = datetime.now()

            outs = []
            for batch in self._data_loader.take(experience_replay_iterations):
                data = {f: d for f, d in zip(self._call_list_data, batch)}
                
                # self.__data.append((datetime.now() - self.t).total_seconds())
                # self.t = datetime.now()

                out = self._learn_from_experience_batch(data)

                # self.__learn.append((datetime.now() - self.t).total_seconds())
                # self.t = datetime.now()

                outs.append(out)
                if self._nan_guard:
                    if not np.isfinite(self._actor.act_deterministic(data['obs']).numpy()).all():
                        print('NaN on learn step')
                        print('Last mem batch:')
                        print_batch(self._nan_log_prev_mem_batch)
                        print('Last batch:')
                        print_batch(self._nan_log_prev_batch)
                    self._nan_log_prev_batch = data

            # if len(self.__run) == 1000:
            #     print('RUN', np.mean(self.__run))
            #     self.__run = []

            # if len(self.__data) == 1000:
            #     print('DATA', np.mean(self.__data))
            #     self.__data = []

            # if len(self.__learn) == 1000:
            #     print('LEARN', np.mean(self.__learn))
            #     self.__learn = []
            # if self._time_step % 100 == 0:
            #     import gc, sys
            #     ob = gc.get_objects()
            #     obt = {}
            #     mbt = {}
            #     for o in ob:
            #         obt[type(o)] = obt.get(type(o), 0) + 1
            #         mbt[type(o)] = mbt.get(type(o), 0) + sys.getsizeof(o)
            #     print(
            #         obt[tf.python.eager.def_function.Function], '/',  len(ob),
            #         mbt[max(mbt, key=mbt.get)], ':', max(mbt, key=mbt.get),
            #         'tf:', sum(v for k, v in mbt.items() if 'tensorflow' in str(k)),
            #         'np:', sum(v for k, v in mbt.items() if 'numpy' in str(k))
            #     )
            return np.mean(outs, 0)
    
    @tf.function(experimental_relax_shapes=True)
    def _calculate_memory_update(self, data):
        data = self.call_list(self._memory_call_list, data, self.PREPROCESSING)

        with tf.name_scope('memory_update_log'):
            for name, value, gather in self._log_memory_values:
                tf.summary.scalar(name + " (mem)", gather(data[value]), self._tf_time_step)

        return data['base.memory_priority']

    @tf.function(experimental_relax_shapes=True)
    def _learn_from_experience_batch(self, data):
        data = self.call_list(self._call_list, data, self.PREPROCESSING)

        with tf.name_scope('log'):
            for name, value, gather in self._log_values:
                tf.summary.scalar(name, gather(data[value]), self._tf_time_step)
        return [gather(data[value]) for _, value, gather in self._log_to_file_values]

    def predict_action_log(self, observations, action):
        if self._log_act_call_list is not None:
            data = {}
            if 'action' in self._log_act_call_list_data:
                data['action'] = action
            if 'obs' in self._log_act_call_list_data:
                data['obs'] = observations
            data = self.call_list(self._log_act_call_list, data, {})
            with tf.name_scope('act_log'):
                for name, value, gather in self._log_act_values:
                    tf.summary.scalar(name, gather(data[value]), self._tf_time_step)
            return [gather(data[value]) for _, value, gather in self._log_to_file_act_values]


    def predict_action(self, observations: np.array, is_deterministic: bool = False):
        action, policy, _ = super().predict_action(observations, is_deterministic)
        if not is_deterministic:
            log = self.predict_action_log(observations, action)
        else:
            log = None
        return action, policy, log

    def _fetch_offline_batch(self) -> List[Dict[str, Union[np.array, list]]]:
        return self._memory.get_vec(self._batches_per_env, self._memory.n)

    def _prepare_generator_fields(self, size):
        batch_size_ids = np.arange(size)

        specs = {  # replacing identity functions with none and not calling gives a slight speed increase
            'lengths': ('lengths', None),  # lambda x, lens: x),
            'obs': ('observations', None),  # lambda x, lens: x),
            'obs_next': ('next_observations', lambda x, lens: x[batch_size_ids, lens - 1]),
            'actions': ('actions', None),  # lambda x, lens: x),
            'policies': ('policies', None),  # lambda x, lens: x),
            'rewards': ('rewards', None),  # lambda x, lens: x),
            'dones': ('dones', lambda x, lens: x[batch_size_ids, lens - 1]),
            'priorities': ('priors', lambda x, lens: x[:, 0]),
            'time': ('time', None),  # lambda x, lens: x),
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

        return specs, dtypes

    def _get_experience_replay_generator(
            self, seq=False, fields=None):
        if fields is None:
            fields = self.DATA_FIELDS

        specs, dtypes = self._prepare_generator_fields(self._memory.block if seq else self._batch_size)

        field_specs = [specs[f] for f in fields]
        field_dtypes = tuple(dtypes[f] for f in fields)

        def experience_replay_generator():
            while True:
                batch, lens = self._memory.get_next_block_to_update() if seq else self._fetch_offline_batch()
                batch["lengths"] = lens
                data = tuple(preproc(batch[field], lens) if preproc else batch[field] for field, preproc in field_specs)
                yield data

        return experience_replay_generator, field_dtypes
