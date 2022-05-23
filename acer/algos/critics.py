from algos.base import Critic
from algos.common.automodel import AutoModelComponent

import numpy as np
import tensorflow as tf


class VarianceCritic(AutoModelComponent):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._value_critic = Critic(*args, **kwargs)
        self._value2_critic = Critic(*args, **kwargs)

        self.value = self._value_critic.value
        self.value2 = self._value2_critic.value

        self.register_method('value', self.value, {'observations': 'obs'})
        self.register_method('value_next', self.value, {'observations': 'obs_next'})
        self.register_method('value2', self.value2, {'observations': 'obs'})
        self.register_method('value2_next', self.value2, {'observations': 'obs_next'})

        self.register_method('std', self.std, {'value': 'self.value', 'value2': 'self.value2'})

        self.register_method('td2', self._calculate_td2, {
            "td": "base.td",
            "values2": "critic.value2",
        })
        self.register_method('weighted_td2', self._calculate_weighted_td2, {
            "td2": "self.td2",
            "weights": "actor.sample_weights",
        })


        self.register_method('optimize', self.optimize, {
            'obs': 'base.first_obs',
            'd': 'base.weighted_td',
            'd2': 'self.weighted_td2'
        })

        self.targets = ['optimize']

    @tf.function
    def std(self, value2):
        variance = value2
        return tf.sqrt(tf.maximum(variance, 0))

    @tf.function
    def _calculate_td2(self, td, values2):
        td2 = td ** 2 - values2[:,:1]

        return td2

    @tf.function(experimental_relax_shapes=True)
    def _calculate_weighted_td2(self, td2, weights):
        return tf.stop_gradient(td2 * weights)

    @tf.function
    def loss(self, critic, obs, d) -> tf.Tensor:
        value = critic.value(obs)
        return tf.reduce_mean(-tf.math.multiply(value, d))

    @tf.function
    def optimize(self, obs, d, d2):
        for critic, cd, in [
            (self._value_critic, d),
            (self._value2_critic, d2)
        ]:
            with tf.GradientTape() as tape:
                loss_v = self.loss(critic, obs, cd)
            grads = tape.gradient(loss_v, critic.trainable_variables)
            gradients = zip(grads, critic.trainable_variables)

            critic.optimizer.apply_gradients(gradients)

    def init_optimizer(self, *args, **kwargs):
        self._value_critic.init_optimizer(*args, **kwargs)
        self._value2_critic.init_optimizer(*args, **kwargs)
