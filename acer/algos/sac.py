import gym
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from typing import Optional, Tuple

from algos.common.automodel import AutoModelComponent
from algos.base import BaseModel
from algos.varsigmaactors import VarSigmaActor
from replay_buffer import MultiReplayBuffer, VecReplayBuffer
from algos.base_nextgen_acer import BaseNextGenACERAgent


class TwinQDelayedCritic(AutoModelComponent, tf.keras.Model):
    @staticmethod
    def get_args():
        args = BaseModel.get_args()
        args['update_delay'] = (int, 1)
        args['tau'] = (float, 0.005)
        return args

    def __init__(
            self, observations_space: gym.Space, action_space: gym.Space, layers,
            *args, update_delay=1, tau=0.005, **kwargs) -> None:

        super().__init__(*args, **kwargs)

        self._update_delay = update_delay
        self._tau = tau

        if isinstance(observations_space, gym.spaces.Discrete):
            assert isinstance(action_space, gym.spaces.Discrete)
            q_input_space = observations_space
        else:
            q_input_space_size = observations_space.shape[0]
            low = observations_space.low
            high = observations_space.high
            if not isinstance(action_space, gym.spaces.Discrete):
                q_input_space_size += action_space.shape[0]
                low = np.concatenate([low, action_space.low])
                high = np.concatenate([high, action_space.high])
            q_input_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)

        if isinstance(action_space, gym.spaces.Discrete):
            q_outs = action_space.n
            self.discrete_actions = True
        else:
            q_outs = 1
            self.discrete_actions = False

        self._q1 = BaseModel(q_input_space, layers, q_outs, *args, **kwargs)
        self._q2 = BaseModel(q_input_space, layers, q_outs, *args, **kwargs)
        self._target_q1 = BaseModel(q_input_space, layers, q_outs, *args, **kwargs)
        self._target_q2 = BaseModel(q_input_space, layers, q_outs, *args, **kwargs)

        if not isinstance(observations_space, gym.spaces.Discrete):
            input_shape = (None, *q_input_space.shape)
            self._q1.build(input_shape)
            self._q2.build(input_shape)
            self._target_q1.build(input_shape)
            self._target_q2.build(input_shape)

        self._target_q1.set_weights(self._q1.get_weights())
        self._target_q2.set_weights(self._q2.get_weights())

        for prefix in '', 'target_':
            for ob in '', '_next':
                self.register_method(f'{prefix}qs{ob}', self.target_qs if prefix else self.qs, {
                    'obs': f'obs{ob}',
                    'actions': 'actor.next_actions' if ob else 'actions'
                })
                self.register_method(f'{prefix}q{ob}', self.min_q, {
                    'qs': f'self.{prefix}qs{ob}',
                })
        self.register_method('td_q_est', self.td_q_est, {
            'next_min_target_qs': 'self.target_q_next',
            'dones': 'dones',
            'rewards': 'rewards',
            'discount': 'base.gamma',
            'entropy_coef': 'actor.entropy_coef',
            'next_log_prob': 'actor.next_log_prob'
        })
        self.register_method('optimize', self.optimize, {
            'obs': 'base.first_obs',
            'actions': 'base.first_actions',
            'td_q_est': 'self.td_q_est',
            'timestep': 'base.time_step'
        })
        self.targets = ['optimize']

    @tf.function
    def qs(self, obs, actions):
        if not self.discrete_actions:
            input = tf.concat([obs, actions], axis=-1)
        else:
            input = obs

        out1 = self._q1._forward(input)
        out2 = self._q2._forward(input)

        if self.discrete_actions:
            out1 = tf.gather_nd(out1, tf.expand_dims(actions, -1))
            out2 = tf.gather_nd(out2, tf.expand_dims(actions, -1))

        return tf.concat([out1, out2], axis=-1)

    @tf.function
    def target_qs(self, obs, actions):
        if not self.discrete_actions:
            input = tf.concat([obs, actions], axis=-1)
        else:
            input = obs

        out1 = self._target_q1._forward(input)
        out2 = self._target_q2._forward(input)

        if self.discrete_actions:
            out1 = tf.gather_nd(out1, tf.expand_dims(actions, -1))
            out2 = tf.gather_nd(out2, tf.expand_dims(actions, -1))

        return tf.concat([out1, out2], axis=-1)

    @tf.function
    def min_q(self, qs):
        return tf.reduce_min(qs, axis=-1)

    @tf.function
    def td_q_est(self, next_min_target_qs, dones, rewards, discount, entropy_coef, next_log_prob):
        target = rewards + tf.expand_dims(
            discount * (next_min_target_qs - entropy_coef * next_log_prob) * (1 - tf.cast(dones, tf.float32)),
            -1
        )
        return target

    @tf.function
    def optimize(self, obs, actions, td_q_est, timestep):
        td_q_est = tf.stop_gradient(td_q_est)
        with tf.GradientTape(persistent=True) as tape:
            qs = self.qs(obs, actions)
            loss_qs = tf.reduce_mean((qs - td_q_est) ** 2)

        for q, target_q in (
            (self._q1, self._target_q1),
            (self._q2, self._target_q2),
        ):
            grads = tape.gradient(loss_qs, q.trainable_variables)
            gradients = zip(grads, q.trainable_variables)
            q.optimizer.apply_gradients(gradients)

            assign_mask = tf.cast(timestep % self._update_delay == 0, tf.float32)
            for w1, w2 in zip(q.trainable_variables, target_q.trainable_variables):
                w2.assign((w2 * (1 - self._tau) + w1 * self._tau) * assign_mask + w2 * (1 - assign_mask))

        return loss_qs

    def init_optimizer(self, *args, **kwargs):
        self._q1.init_optimizer(*args, **kwargs)
        self._q2.init_optimizer(*args, **kwargs)
        self._target_q1.init_optimizer(*args, **kwargs)
        self._target_q2.init_optimizer(*args, **kwargs)


class GaussianSoftActor(VarSigmaActor):
    @staticmethod
    def get_args():
        args = VarSigmaActor.get_args()
        args['nn_std'] = (bool, True)
        args['target_entropy'] = (float, None)
        args['clip_mean'] = (float, None)
        return args

    def __init__(self, obs_space, action_space, *args, target_entropy=None, nn_std=True, clip_mean=None, **kwargs):
        super().__init__(obs_space, action_space, *args, custom_optimization=True, nn_std=nn_std, clip_mean=clip_mean, **kwargs)
        if target_entropy is None:
            target_entropy = -np.prod(action_space.shape)
        self._target_entropy = target_entropy
        self._log_entropy_coef = tf.Variable(0., dtype=tf.float32)

        self.register_method('entropy_coef', self.entropy_coef, {})

        self.register_method('optimize', self.optimize, {
            'obs': 'base.first_obs',
        })

        self.register_method('next_actions_with_log_prob', self.sample_inplace_with_log_prob, {
            'obs': 'obs_next'
        })

        self.register_method('next_actions', self.sample_inplace, {
            'act_with_log_prob': 'self.next_actions_with_log_prob'
        })

        self.register_method('next_log_prob', self.expected_action_log_prob, {
            'act_with_log_prob': 'self.next_actions_with_log_prob'
        })

        self.register_method(
            'sample_weights', self._calculate_truncated_weights, {
                'priorities': 'priorities'
            }
        )

    @tf.function
    def _calculate_truncated_weights(self, priorities):
        return tf.reshape(priorities, (-1, 1))

    @tf.function
    def sample_inplace_with_log_prob(self, obs):
        dist = self._dist(obs)
        return dist.sample_with_log_prob()

    @tf.function
    def sample_inplace(self, act_with_log_prob):
        act, _ = act_with_log_prob
        return act
    
    @tf.function
    def expected_action_log_prob(self, act_with_log_prob):
        _, lp = act_with_log_prob
        return lp

    @tf.function
    def entropy_coef(self):
        return tf.exp(self._log_entropy_coef)

    # @tf.function
    # def mean_and_std(self, observations):
    #     mean, std = super().mean_and_std(observations)
    #     return mean, tf.ones_like(std) * 0.25

    @tf.function
    def loss(self, obs):
        mean, std = self.mean_and_std(obs)
        dist = self.distribution(
            loc=mean,
            scale_diag=std
        )

        actions, log_probs = dist.sample_with_log_prob()
        entropy_coef = tf.exp(self._log_entropy_coef)

        qs = self.call_now('critic.qs', {'obs': obs, 'actions': actions})
        min_q = tf.reduce_min(qs, axis=-1)

        entropy_loss = -tf.reduce_mean(self._log_entropy_coef * tf.stop_gradient(self._target_entropy + log_probs))
        actor_loss = tf.reduce_mean(tf.stop_gradient(entropy_coef) * log_probs - min_q)

        return entropy_loss + actor_loss


class SAC(BaseNextGenACERAgent):
    ACTORS = {'simple': {False: GaussianSoftActor}}
    CRITICS = {'simple': TwinQDelayedCritic}
    BUFFERS = {
        'simple': (MultiReplayBuffer, {'buffer_class': VecReplayBuffer}),
    }

    def __init__(self, *args, **kwargs):
        if 'buffer.n' in kwargs:
            assert kwargs['buffer.n'] == 1
        else:
            kwargs['buffer.n'] = 1

        super().__init__(*args, **kwargs)

        self.register_method('weighted_td', self._calculate_weighted_td, {
            "td_q_est": "critic.td_q_est",
            "qs": "critic.qs",
            "weights": "actor.sample_weights",
        })


    def _init_automodel(self, skip=()):
        super()._init_automodel(skip=skip)

    def _init_critic(self):
        # if self._is_obs_discrete:
        #     return TabularCritic(self._observations_space, None, self._tf_time_step)
        # else:
        return self.CRITICS[self._critic_type](
            self._observations_space, self._actions_space,
            tf_time_step=self._tf_time_step, **self._critic_args
        )

    def _calculate_weighted_td(self, td_q_est, qs, weights):
        return (tf.reduce_mean(qs, axis=-1, keepdims=True) - td_q_est) * weights
