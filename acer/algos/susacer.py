from operator import xor
from algos.common.parameters import Parameters, get_adapts_from_kwargs
from typing import Tuple
from algos.base import GaussianActor
from algos.fast_acer import ACTORS, FastACER
from replay_buffer import BufferFieldSpec
import numpy as np
import tensorflow as tf


def _calc_sustain(esteps):
    return 1 - 1 / esteps


def _calc_esteps(sustain):
    return 1 / (1 - sustain)


class SusActor:
    def __init__(
            self, obs_space, action_space, *args,
            sustain=None, esteps=None, limit_sustain_length=None, num_parallel_envs=1, modify_std=False, single_step_mask=False,
            **kwargs):
        assert xor(sustain is None, esteps is None)

        self.parameters = Parameters(
            "actor_params", calculatables={
                'sustain': (_calc_sustain, {'esteps': 'self.esteps'}),
                'esteps': (_calc_esteps, {'sustain': 'self.sustain'})
            },
            sustain=sustain, esteps=esteps, **get_adapts_from_kwargs(kwargs, ['sustain', 'esteps']))

        self.limit_sustain_length = limit_sustain_length or np.inf
        self.ends = np.zeros(num_parallel_envs)

        self.modify_std = modify_std
        self.single_step_mask = single_step_mask

        self.previous_actions = tf.Variable(tf.zeros((num_parallel_envs, *action_space.shape)), dtype=tf.float32, trainable=False)
        self.action_lengths = tf.Variable(tf.zeros((num_parallel_envs,)), dtype=tf.float32, trainable=False)

        if self.single_step_mask:
            self.register_method(
                'sample_weights', self._calculate_truncated_single_step_weights, {
                    'density': 'actor.density',
                    'priorities': 'priorities',
                    'actions': 'actions',
                }
            )

    @property
    def sustain(self):
        return self.parameters['sustain']

    @property
    def esteps(self):
        return self.parameters['esteps']

    def mean_and_std(self, observation):
        std_mult = 1
        if self.modify_std:
            esteps = self.parameters.get_value('esteps')
            if esteps is None:
                esteps = _calc_esteps(self.parameters.get_value('sustain'))
            esteps = tf.minimum(esteps, self.limit_sustain_length)
            std_mult = 1 / tf.stop_gradient(tf.sqrt(esteps))

        return self._forward(observation), tf.exp(self.log_std) * std_mult

    def act(self, observations, new_actions, policies):
        sustain = self.parameters.get_value('sustain')
        if sustain is None:
            sustain = _calc_sustain(self.parameters.get_value('esteps'))

        first_mask = 1 - tf.cast(self.ends, tf.float32)
        limit_mask = tf.cast(
            self.action_lengths < self.limit_sustain_length,
            tf.float32
        )
        mask = first_mask * limit_mask * tf.cast(
            tf.random.uniform(shape=(observations.shape[0],)) < sustain,
            tf.float32
        )

        self.action_lengths.assign(self.action_lengths * mask + 1)

        actions = self.previous_actions * mask + new_actions * (1 - mask)
        sustain_policies = (
            sustain * mask 
            + policies * (1 - sustain) * (1 - (mask - first_mask * limit_mask)) 
            + policies * (1 - first_mask * limit_mask)
        )
        self.previous_actions.assign(actions)
        return actions, tf.stack([1 - mask, sustain_policies, policies], axis=1)

    def update_ends(self, ends):
        self.ends = ends[:,0]

    tf.function(experimental_relax_shapes=True)
    def calculate_probs(self, dist, actions, sustain, n):
        amask = tf.reduce_prod(tf.cast(actions[:,1:] == actions[:, :-1], tf.float32), axis=-1)
        susmask = tf.concat([tf.zeros_like(amask[:,:1]), amask], axis=-1)

        cumprods_mask = tf.cast(
            tf.expand_dims(tf.cumsum(tf.ones_like(susmask), -1), -1) >= tf.expand_dims(tf.cumsum(tf.ones_like(susmask), -1), -2),
            # tf.expand_dims(tf.range(tf.shape(susmask)[1]), -1) <= tf.expand_dims(tf.range(tf.shape(susmask)[1]), -2),
            tf.float32
        )
        cumprods = tf.math.cumprod(
            tf.expand_dims(susmask, -1) * tf.ones(n) * cumprods_mask + (1 - cumprods_mask),
            axis=-2
        )
        
        suslens = tf.reduce_sum(cumprods * cumprods_mask, axis=-1)

        limit_mask = tf.cast(suslens + 1 >= self.limit_sustain_length, tf.float32)
        base_prob_mask = tf.concat([tf.ones_like(limit_mask[:,:1]), limit_mask[:,:-1]], axis=-1)

        base_prob = dist.prob(actions)
        base_prob_log = dist.log_prob(actions)

        return (
            base_prob_mask * base_prob 
            + (1 - susmask - base_prob_mask) * (1 - sustain) * base_prob 
            + susmask * sustain,
            
            base_prob_mask * base_prob_log
            + (1 - susmask - base_prob_mask) * (base_prob_log + tf.math.log(1 - sustain))
            + susmask * tf.math.log(sustain)
        )
    
    @tf.function
    def _calculate_truncated_single_step_weights(self, actions, *args, **kwargs):
        weights = self._calculate_truncated_weights(*args, **kwargs)
        mask = tf.math.cumprod(tf.cast(tf.reduce_all(actions[:,1:] == actions[:,:-1], axis=-1), tf.float32), axis=1)
        mask = tf.concat([tf.ones_like(mask[:,:1]), mask], axis=1)

        return weights * mask


class GaussianSusActor(GaussianActor, SusActor):
    def __init__(self, *args, **kwargs):
        GaussianActor.__init__(self, *args, **kwargs)
        SusActor.__init__(self, *args, **kwargs)

        self.register_method(
            'policies', self.policy, 
            {'observations': 'obs', 'actions': 'actions', 'sustain': 'actor_params.sustain', 'n': 'memory_params.n'}
        )

    def update_ends(self, ends):
        return SusActor.update_ends(self, ends)

    @tf.function
    def act(self, observations: tf.Tensor, **kwargs):
        actions, policy = GaussianActor.act(self, observations, **kwargs)
        return SusActor.act(self, observations, actions, policy)

    def policy(self, observations: tf.Tensor, actions: tf.Tensor, sustain: tf.Tensor, n) -> Tuple[tf.Tensor, tf.Tensor]:
        return self.prob(observations, actions, sustain, n)[0]

    def prob(self, observations: tf.Tensor, actions: tf.Tensor, sustain: tf.Tensor, n) -> Tuple[tf.Tensor, tf.Tensor]:
        dist = GaussianActor._dist(self, observations)
        return SusActor.calculate_probs(self, dist, actions, sustain, n)


ACTORS['sustain'] = {True: None, False: GaussianSusActor}


class SusACER(FastACER):
    def __init__(self, *args, **kwargs) -> None:
        kwargs['actor_type'] = 'sustain'
        kwargs['buffer_type'] = 'prioritized'
        kwargs['buffer.updatable'] = False
        kwargs['buffer.probability_as_actor_out'] = True

        policy_spec = BufferFieldSpec(shape=(2,), dtype=np.float32)

        super().__init__(*args, policy_spec = policy_spec, **kwargs)

    def _init_automodel_overrides(self) -> None:
        self.register_component('actor_params', self._actor.parameters)

    def _prepare_generator_fields(self, size):
        specs, dtypes = FastACER._prepare_generator_fields(self, size)
        specs['policies'] = ('policies', lambda x, lens: np.concatenate([x[:,:1,1], x[:,1:,0]], axis=-1))
        return specs, dtypes 
