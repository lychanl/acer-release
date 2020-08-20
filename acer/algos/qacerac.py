from collections import deque
from typing import Optional, List, Union, Dict, Tuple
# import os
# os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import gym
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

import tf_utils
from algos.base import BaseActor, Critic, BaseACERAgent, GaussianActor
from algos.acerac import NoiseGaussianActor, ACERAC
from algos.quantile_acer import QuantileCritic, QACER
from replay_buffer import BufferFieldSpec, PrevReplayBuffer, MultiReplayBuffer

class QACERAC(ACERAC, QACER):
    def __init__(self, observations_space: gym.Space, actions_space: gym.Space, actor_layers: Optional[Tuple[int]],
                 critic_layers: Optional[Tuple[int]], b: float = 3, tau: int = 2, alpha: int = None, atoms: int = 50,
                 kappa: float = 0., td_clip: float = None, use_normalized_density_ratio: bool = False, *args, **kwargs):
        """Actor-Critic with Experience Replay and autocorrelated actions.

        Args:
            observations_space: observations' vectors Space
            actions_space: actions' vectors Space
            actor_layers: number of units in Actor's hidden layers
            critic_layers: number of units in Critic's hidden layers
            b: density ratio truncating coefficient
            tau: update window size
            alpha: autocorrelation coefficient. If None, 1 - (1 / tau) is set
        """
        self._tau = tau
        if alpha is None:
            self._alpha = 1 - (1 / tau)
        else:
            self._alpha = alpha

        QACER.__init__(self, observations_space, actions_space, actor_layers, critic_layers, **kwargs)

        ACERAC.__init__(self, observations_space, actions_space, actor_layers, critic_layers, b, tau, alpha,
            td_clip, use_normalized_density_ratio, *args, **kwargs)

    def _init_actor(self) -> BaseActor:
        return ACERAC._init_actor(self)

    def _init_critic(self) -> Critic:
        return QACER._init_critic(self)
    
    @tf.function(experimental_relax_shapes=True)
    def _learn_from_experience_batch(self, obs: tf.Tensor, obs_next: tf.Tensor, actions: tf.Tensor,
                                     old_means: tf.Tensor, rewards: tf.Tensor, dones: tf.Tensor,
                                     lengths: tf.Tensor, is_prev_noise: tf.Tensor,
                                     prev_obs: tf.Tensor, prev_actions: tf.Tensor, prev_means: tf.Tensor):
        obs = self._process_observations(obs)
        obs_next = self._process_observations(obs_next)
        prev_obs = self._process_observations(prev_obs)
        rewards = self._process_rewards(rewards)

        # TODO whole (tiled) matrices in init
        is_prev_noise_mask = tf.cast(tf.expand_dims(is_prev_noise, 1), tf.float32)

        c_invs = self._get_c_invs(actions, is_prev_noise_mask)
        eta_repeated, mu_repeated = self._get_prev_noise(actions, is_prev_noise_mask, prev_actions, prev_means, prev_obs)

        with tf.GradientTape(persistent=True) as tape:
            means = self._actor.act_deterministic(obs)
            values, values_next = tf.split(self._critic.value(tf.concat([obs, obs_next], axis=0)), 2)
            values_next = values_next * tf.expand_dims((1 - tf.cast(dones, tf.float32)), axis=2)
            values_first = tf.slice(values, [0, 0, 0], [actions.shape[0], 1, -1])

            actions_flatten = tf.reshape(actions, (actions.shape[0], -1))
            means_flatten = tf.reshape(means, (actions.shape[0], -1))
            old_means_flatten = tf.reshape(old_means, (actions.shape[0], -1))

            actions_repeated = tf.repeat(tf.expand_dims(actions_flatten, axis=1), self._tau, axis=1)
            means_repeated = tf.repeat(tf.expand_dims(means_flatten, axis=1), self._tau, axis=1)
            old_means_repeated = tf.repeat(tf.expand_dims(old_means_flatten, axis=1), self._tau, axis=1)

            # 1, 2, ..., n trajectories mask over repeated action tensors
            actions_mask = tf.expand_dims(
                tf.sequence_mask(
                    tf.range(actions.shape[2], actions_repeated.shape[2] + actions.shape[2], actions.shape[2]),
                    actions_repeated.shape[2],
                    dtype=tf.float32
                ),
                axis=0
            )

            # trajectories shorter than tau mask
            zeros_mask = tf.expand_dims(tf.sequence_mask(lengths, maxlen=self._tau, dtype=tf.float32), 2)

            actions_mu_diff_current = tf.expand_dims(
                (actions_repeated - means_repeated - eta_repeated) * zeros_mask * actions_mask,
                axis=2
            )
            actions_mu_diff_old = tf.expand_dims(
                (actions_repeated - old_means_repeated - mu_repeated) * zeros_mask * actions_mask,
                axis=2
            )

            if self._use_normalized_density_ratio:
                density_ratio = self._compute_normalized_density_ratio(
                    actions_mu_diff_current, actions_mu_diff_old, c_invs
                )
            else:
                density_ratio = self._compute_soft_truncated_density_ratio(
                    actions_mu_diff_current, actions_mu_diff_old, c_invs
                )

            with tf.name_scope('acerac'):
                tf.summary.scalar('mean_density_ratio', tf.reduce_mean(density_ratio), step=self._tf_time_step)
                tf.summary.scalar('max_density_ratio', tf.reduce_max(density_ratio), step=self._tf_time_step)

            gamma_coeffs = tf.math.cumprod(tf.ones_like(rewards) * self._gamma, exclusive=True, axis=1)
            td_rewards = tf.math.cumsum(rewards * gamma_coeffs, axis=1)

            values_first_repeated = tf.repeat(values_first, self._tau, 1)
            pows = tf.tile(tf.expand_dims(tf.range(1, self._tau + 1), axis=0), [actions.shape[0], 1])
            td = (-tf.expand_dims(values_first_repeated, axis=2)
                  + tf.expand_dims(tf.expand_dims(td_rewards, axis=2), axis=3)
                  + tf.expand_dims(tf.expand_dims(tf.pow(self._gamma, tf.cast(pows, tf.float32)), axis=2), axis=3)
                      * tf.expand_dims(values_next, axis=3))

            if self._td_clip is not None:
                td = tf.clip_by_value(td, -self._td_clip, self._td_clip)

            td_actor = tf.reduce_mean(td, axis=[2, 3])
            td_critic = self._quantile_loss(td, reduce_dim=2)

            d_mask = density_ratio * tf.squeeze(zeros_mask)
            d_actor = td_actor * d_mask
            d_critic = td_critic *  tf.expand_dims(d_mask, axis=2)

            c_mu = tf.matmul(c_invs, tf.transpose(actions_mu_diff_current, [0, 1, 3, 2]))
            c_mu_d = c_mu * tf.expand_dims(tf.expand_dims(d_actor, axis=2), 3)

            c_mu_mean = (tf.reduce_sum(tf.squeeze(c_mu_d), axis=1) / tf.expand_dims(tf.cast(lengths, tf.float32), 1))

            bounds_penalty = tf.scalar_mul(
                    self._actor.beta_penalty,
                    tf.square(tf.maximum(0.0, tf.abs(means) - self._actions_bound))
            )
            bounds_penalty = tf.squeeze(zeros_mask) * tf.reduce_sum(
                bounds_penalty,
                axis=2
            )

            bounds_penalty = tf.reduce_sum(bounds_penalty, axis=1) / tf.cast(lengths, tf.float32)
            actor_loss = tf.matmul(tf.expand_dims(means_flatten, axis=1), tf.expand_dims(tf.stop_gradient(c_mu_mean), axis=2))
            actor_loss = -tf.reduce_mean(tf.squeeze(actor_loss)) + tf.reduce_mean(bounds_penalty)

            d_mean = tf.reduce_sum(d_critic, axis=1) / tf.expand_dims(tf.cast(lengths, tf.float32), axis=1)
            critic_loss = -tf.reduce_mean(tf.reduce_sum(tf.squeeze(values_first) * tf.stop_gradient(d_mean), axis=-1))

        grads_actor = tape.gradient(actor_loss, self._actor.trainable_variables)
        
        with tf.name_scope('actor'):
            tf.summary.scalar(f'gradient_norm', tf.linalg.global_norm(grads_actor), step=self._tf_time_step)
        grads_var_actor = zip(grads_actor, self._actor.trainable_variables)
        self._actor_optimizer.apply_gradients(grads_var_actor)

        with tf.name_scope('actor'):
            tf.summary.scalar(f'batch_actor_loss', actor_loss, step=self._tf_time_step)
            tf.summary.scalar(f'batch_bounds_penalty', tf.reduce_mean(bounds_penalty), step=self._tf_time_step)

        grads_critic = tape.gradient(critic_loss, self._critic.trainable_variables)

        with tf.name_scope('critic'):
            tf.summary.scalar(f'gradient_norm', tf.linalg.global_norm(grads_critic), step=self._tf_time_step)
        grads_var_critic = zip(grads_critic, self._critic.trainable_variables)
        self._critic_optimizer.apply_gradients(grads_var_critic)

        with tf.name_scope('critic'):
            tf.summary.scalar(f'batch_critic_loss', critic_loss, step=self._tf_time_step)
            tf.summary.scalar(f'batch_value_mean', tf.reduce_mean(values), step=self._tf_time_step)
            
            if self._critic._outputs > 1:
                value_lower = tf.slice(tf.squeeze(values_first), [0, 0], [-1, self._critic._outputs - 1])
                value_higher = tf.slice(tf.squeeze(values_first), [0, 1], [-1, self._critic._outputs - 1])

                correct_order = value_lower < value_higher
                tf.summary.scalar('correct_quantile_order', tf.reduce_mean(tf.cast(correct_order, tf.float32)), self._tf_time_step)

    
    def _fetch_offline_batch(self):
        return ACERAC._fetch_offline_batch(self)
