from algos.common.parameters import Parameters, get_adapts_from_kwargs
from typing import Tuple
from algos.base import GaussianActor
from algos.fast_acer import ACTORS, FastACER
import numpy as np
import tensorflow as tf


class SusActor:
    def __init__(self, *args, sustain, **kwargs):
        self.parameters = Parameters("actor_params", sustain=sustain, **get_adapts_from_kwargs(kwargs, ['sustain']))

        self.previous_actions = None
        self.ends = 1

    @property
    def sustain(self):
        return self.parameters['sustain']

    def act(self, observations, new_actions, policies):
        if self.previous_actions is None:
            self.previous_actions = new_actions

        mask = (
            1 - tf.cast(self.ends, tf.float32)
        ) * tf.cast(
            tf.random.uniform(shape=(observations.shape[0],)) < self.parameters.get_value('sustain'),
            tf.float32
        )

        actions = self.previous_actions * mask + new_actions * (1 - mask)
        self.previous_actions = actions
        return actions, tf.stack([1 - mask, policies], axis=1)

    def update_ends(self, ends):
        self.ends = ends[:,0]

    def calculate_probs(self, dist, actions, sustain):
        amask = tf.reduce_prod(tf.cast(actions[:,1:] == actions[:, :-1], tf.float32), axis=-1)
        susmask = tf.concat([tf.zeros_like(amask[:,:1]), amask], axis=-1)
        return (
            dist.prob(actions) * (1 - sustain) * (1 - susmask) + sustain * susmask,
            (dist.log_prob(actions) + tf.math.log(1 - sustain)) * (1 - susmask) + tf.math.log(sustain) * susmask,
        )


class GaussianSusActor(GaussianActor, SusActor):
    def __init__(self, *args, **kwargs):
        GaussianActor.__init__(self, *args, **kwargs)
        SusActor.__init__(self, *args, **kwargs)

        self.register_method(
            'policies', self.policy, 
            {'observations': 'obs', 'actions': 'actions', 'sustain': 'actor_params.sustain'}
        )

    def update_ends(self, ends):
        return SusActor.update_ends(self, ends)

    @tf.function
    def act(self, observations: tf.Tensor, **kwargs):
        actions, policy = GaussianActor.act(self, observations, **kwargs)
        return SusActor.act(self, observations, actions, policy)

    def policy(self, observations: tf.Tensor, actions: tf.Tensor, sustain: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        return self.prob(observations, actions, sustain)[0]

    def prob(self, observations: tf.Tensor, actions: tf.Tensor, sustain: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        dist = GaussianActor._dist(self, observations)
        return SusActor.calculate_probs(self, dist, actions, sustain)


ACTORS['sustain'] = {True: None, False: GaussianSusActor}


class SusACER(FastACER):
    def __init__(self, *args, **kwargs) -> None:
        kwargs['actor_type'] = 'sustain'
        kwargs['buffer_type'] = 'prioritized'
        kwargs['buffer.updatable'] = False
        kwargs['buffer.probability_as_actor_out'] = True

        super().__init__(*args, **kwargs)

    def _init_automodel_overrides(self) -> None:
        self.register_component('actor_params', self._actor.parameters)
