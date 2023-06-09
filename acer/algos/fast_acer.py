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
import tensorflow as tf

from algos.base import CategoricalActor, GaussianActor, Critic
from algos.base_nextgen_acer import BaseNextGenACERAgent
from replay_buffer import VecReplayBuffer, MultiReplayBuffer
from prioritized_buffer import MultiPrioritizedReplayBuffer


ACTORS = {
    'simple': {True: CategoricalActor, False: GaussianActor},
}


CRITICS = {
    'simple': Critic,
}

BUFFERS = {
    'simple': (MultiReplayBuffer, {'buffer_class': VecReplayBuffer}),
    'prioritized': (MultiPrioritizedReplayBuffer, {}),
}


class FastACER(BaseNextGenACERAgent):
    BUFFERS = BUFFERS
    ACTORS = ACTORS
    CRITICS = CRITICS

    def __init__(self, *args, **kwargs):
        """BaseActor-Critic with Experience Replay

        TODO: finish docstrings
        """

        super().__init__(*args, **kwargs)

    def _init_automodel(self, skip=()):
        self.register_method('td', self._calculate_td, {
            "values": "critic.value",
            "values_next": "critic.value_next",
            "rewards": "rewards",
            "lengths": "lengths",
            "mask": "base.mask",
            "dones": "dones",
            "n": "memory_params.n"
        })
        self.register_method('density_weighted_td', self._calculate_weighted_td, {
            "td": "base.td",
            "weights": "actor.truncated_density",
        })
        self.register_method('weighted_td', self._calculate_weighted_td, {
            "td": "base.td",
            "weights": "actor.sample_weights",
        })

        super()._init_automodel(skip=skip)

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

        # move next value to the right position 
        next_mask = tf.cast(tf.expand_dims(tf.range(1, n + 1), 0) == tf.expand_dims(lengths, 1), tf.float32)
        values_next = ((1 - next_mask) * values_with_next + next_mask * values_next) * dones_mask

        gamma_coeffs_masked = tf.expand_dims(tf.pow(self._gamma, tf.range(1., n + 1)), axis=0) * mask

        td_parts = rewards + self._gamma * values_next - values
        td_rewards = tf.math.cumsum(td_parts * gamma_coeffs_masked, axis=1)

        return td_rewards

    @tf.function(experimental_relax_shapes=True)
    def _calculate_weighted_td(self, td, weights):
        return tf.stop_gradient(td * weights)
