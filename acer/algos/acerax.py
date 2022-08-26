from typing import Tuple
from algos.fast_acer import FastACER, ACTORS
from algos.varsigmaactors import VarSigmaActor

from replay_buffer import BufferFieldSpec

import numpy as np
import tensorflow as tf


class AceraxActor(VarSigmaActor):
    def __init__(self, *args, alpha, **kwargs):
        self.alpha = alpha

        VarSigmaActor.__init__(self, *args, **kwargs)

        self.register_method('optimize', self.optimize_mean if self.separate_nn_std else self.optimize, {
            'observations': 'base.first_obs',
            'actions': 'base.first_actions',
            'd': 'base.weighted_td',
            'm': 'means'
        })
        self.register_method('optimize_std', self.optimize_std, {
            'observations': 'base.first_obs',
            'actions': 'base.first_actions',
            'm': 'means'
        })

    @tf.function
    def act(self, observations: tf.Tensor, **kwargs) -> Tuple[tf.Tensor, tf.Tensor]:
        dist = self._dist(observations)
        means = dist.loc
        actions, policies = self._act(dist)
        return actions, tf.concat([tf.expand_dims(policies, -1), means], axis=-1)

    def std_loss(self, observations: np.array, actions: np.array, m: np.array, **kwargs) -> tf.Tensor:
        # from ACERAX paper code
        mean, std = self.mean_and_std(observations)
        mean = tf.stop_gradient(mean)
        alpha = self.alpha
  
        no_alpha =  tf.reduce_sum(
            tf.scalar_mul(0.5, tf.square(
                tf.math.multiply(m - mean,
                tf.math.reciprocal(std))
            )))
        with_alpha = tf.reduce_sum(
            tf.scalar_mul(0.5, tf.square(
                tf.math.multiply(actions - mean,
                tf.math.reciprocal(std))
            )))
        total_loss = no_alpha + tf.scalar_mul(alpha, with_alpha) + tf.scalar_mul(
            (1 + alpha),
            tf.reduce_sum(tf.math.log(std))
        )

        return total_loss


ACTORS['acerax'] = {False: AceraxActor}


class ACERAX(FastACER):
    def __init__(self, *args, actions_space, actor_type=None, **kwargs):
        self.DATA_FIELDS = self.DATA_FIELDS + ('means',)
        policy_spec = BufferFieldSpec(shape=(actions_space.shape[0] + 1,), dtype=np.float32)
        FastACER.__init__(self, *args, actions_space=actions_space, actor_type='acerax', policy_spec=policy_spec, **kwargs)


    def _prepare_generator_fields(self, size):
        specs, dtypes = FastACER._prepare_generator_fields(self, size)
        specs['policies'] = ('policies', lambda x, lens: x[:,:,0])
        specs['means'] = ('policies', lambda x, lens: x[:,0,1:])
        dtypes['means'] = tf.float32

        return specs, dtypes 
