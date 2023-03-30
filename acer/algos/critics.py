import functools
from typing import Optional, Tuple

import gym
from algos.base import BaseModel, Critic
from algos.common.automodel import AutoModelComponent

import numpy as np
import tensorflow as tf


VARIANCE_FUNS = {
    'identity': tf.identity,
    'exp': tf.exp,
    'square': tf.square,
    'abs': tf.abs
}


"""
TODO Check if properly handles dones while calculating d2
"""
class VarianceCritic(AutoModelComponent):
    @staticmethod
    def get_args():
        args = Critic.get_args()
        args['variance_fun'] = (str, 'identity')
        return args

    def __init__(self, *args, variance_fun='identity', **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._value_critic = Critic(*args, **kwargs)
        self._value2_critic = Critic(*args, **kwargs)

        self.value = self._value_critic.value
        self.value2 = self._value2_critic.value

        self.variance_fun = VARIANCE_FUNS[variance_fun]

        self.register_method('value', self.value, {'observations': 'obs'})
        self.register_method('value_next', self.value, {'observations': 'obs_next'})
        self.register_method('value2', self.value2, {'observations': 'obs'})
        self.register_method('value2_next', self.value2, {'observations': 'obs_next'})

        self.register_method('std', self.std, {'value2': 'self.value2'})

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
        variance = self.variance_fun(value2)
        return tf.sqrt(tf.maximum(variance, 0))

    @tf.function
    def _calculate_td2(self, td, values2):
        td2 = td ** 2 - self.variance_fun(values2[:,:1])

        return td2

    @tf.function(experimental_relax_shapes=True)
    def _calculate_weighted_td2(self, td2, weights):
        return tf.stop_gradient(td2 * weights)

    @tf.function
    def loss(self, critic, fun, obs, d) -> tf.Tensor:
        value = fun(critic.value(obs))
        return tf.reduce_mean(-tf.math.multiply(value, d))

    @tf.function
    def optimize(self, obs, d, d2):
        for critic, fun, cd, in [
            (self._value_critic, tf.identity, d),
            (self._value2_critic, self.variance_fun, d2)
        ]:
            with tf.GradientTape() as tape:
                loss_v = self.loss(critic, fun, obs, cd)
            grads = tape.gradient(loss_v, critic.trainable_variables)
            gradients = zip(grads, critic.trainable_variables)

            critic.optimizer.apply_gradients(gradients)

    def init_optimizer(self, *args, **kwargs):
        self._value_critic.init_optimizer(*args, **kwargs)
        self._value2_critic.init_optimizer(*args, **kwargs)


class QuantileCritic(Critic):
    @staticmethod
    def get_args():
        args = Critic.get_args()
        args['n_quantiles'] = (int, 50)
        args['kappa'] = (float, 0)
        args['outliers_q'] = (int, None)
        return args

    def __init__(self, *args, n_quantiles=50, kappa=0, outliers_q=None, **kwargs) -> None:
        super().__init__(*args, nouts=n_quantiles, **kwargs)

        if outliers_q is None:
            outliers_q = n_quantiles // 20

        self.outliers_q = outliers_q

        self.nouts = n_quantiles
        quantiles_loc = (np.arange(0, n_quantiles) + 0.5) / n_quantiles
        self.quantiles = quantiles_loc.reshape((1, 1, n_quantiles))
        self.kappa = kappa

        self.register_method('quantiles', self.calc_quantiles, {'observations': 'obs'})
        self.register_method('quantiles_next', self.calc_quantiles, {'observations': 'obs_next'})

        self.register_method('quantile_td', self.calculate_quantile_tds, {
            'quantiles': 'self.quantiles', 'values': 'self.value', 'td': 'base.td'
        })
        self.register_method('quantile_d', self.calculate_quantile_d, {'qtd': 'self.quantile_td', 'weights': 'actor.sample_weights'})

        self.register_method('value', self.value, {'quantiles': 'self.quantiles'})
        self.register_method('value_next', self.value, {'quantiles': 'self.quantiles_next'})

        self.register_method('optimize', self.optimize, {
            'observations': 'base.first_obs',
            'd': 'self.quantile_d'
        })

        self.register_method('outliers_mask', self.outliers_mask, {'quantile_td': 'self.quantile_td'})

        if n_quantiles % 2 == 0:
            self.register_method('median', functools.partial(self.mq, n=n_quantiles // 2 - 1), {'quantiles': 'self.quantiles'})
        else:
            self.register_method('median', functools.partial(self.q, n=n_quantiles // 2), {'quantiles': 'self.quantiles'})

        for n in range(n_quantiles):
            self.register_method(f'q{n}', functools.partial(self.q, n=n), {'quantiles': 'self.quantiles'})
            self.register_method(f'q-{n + 1}', functools.partial(self.q, n=-(n+1)), {'quantiles': 'self.quantiles'})
            self.register_method(
                f'value_q{n}_diff', functools.partial(self.value_q_diff, n=n),
                {'value': 'self.value', 'quantiles': 'self.quantiles'}
            )

        self.register_parameterized_method('quantiles', self.calc_quantiles, {}, ['observations'])

    @tf.function
    def outliers_mask(self, quantile_td):
        return tf.cast((quantile_td[...,self.outliers_q] < 0) | (quantile_td[...,-(self.outliers_q+1)] > 0), tf.float32)

    @tf.function(experimental_relax_shapes=True)
    def calc_quantiles(self, observations):
        return self._forward(observations)

    @tf.function(experimental_relax_shapes=True)
    def calculate_quantile_tds(self, quantiles, values, td):
        qtd = tf.expand_dims(td + values[:,0], -1) - quantiles[:,:1]
        return qtd

    @tf.function
    def calculate_quantile_d(self, qtd, weights):
        quantile_weights = (self.quantiles - tf.cast(qtd < 0, tf.float32))
        weights = tf.expand_dims(weights, axis=-1)

        if self.kappa > 0:
            # Quantile Huber loss gradient
            huber = tf.where(
                tf.abs(qtd) < self.kappa, 
                qtd / self.kappa,
                tf.sign(qtd)
            )
            return tf.stop_gradient(huber * quantile_weights * weights)
        else:
            # Quantile loss gradient
            return tf.stop_gradient(quantile_weights * weights)

    @tf.function
    def value(self, quantiles):
        return tf.reduce_mean(quantiles, -1, keepdims=True)

    @tf.function
    def q(self, quantiles, n):
        return quantiles[:,:,n]

    @tf.function
    def mq(self, quantiles, n):
        return (quantiles[:, :, n] + quantiles[:, :, n + 1]) / 2

    @tf.function
    def dqnqm(self, quantiles, n, m):
        return quantiles[:,:,n] - quantiles[:,:,m]

    @tf.function
    def value_q_diff(self, value, quantiles, n):
        return quantiles[:,0,n] - value[:,0,0]

    @tf.function
    def loss(self, observations: np.array, d: np.array) -> tf.Tensor:
        """Computes Critic's loss.

        Args:
            observations: batch [batch_size, observations_dim] of observations vectors
            d: batch [batch_size, 1] of gradient update coefficient (summation term in the Equation (9)) from
                the paper (1))
        """

        qs = self.calc_quantiles(observations)
        loss = tf.reduce_mean(-tf.expand_dims(qs, axis=1) * d)

        return loss
