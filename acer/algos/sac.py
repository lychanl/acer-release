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



class TwinQDelayedCritic(AutoModelComponent):
    def __init__(
            self, observations_space: gym.Space, action_space: gym.Space, layers, _,
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
        self.register_method('qd', self.qd, {
            'qs': 'self.qs',
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
            'qd': 'self.qd',
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
    def qd(self, qs, next_min_target_qs, dones, rewards, discount, entropy_coef, next_log_prob):
        target = rewards + tf.expand_dims(
            discount * (next_min_target_qs * (1 - tf.cast(dones, tf.float32)) - entropy_coef * next_log_prob),
            -1
        )
        return qs - tf.expand_dims(target, -1)

    @tf.function
    def optimize(self, obs, actions, qd, timestep):
        qd = tf.stop_gradient(qd)
        with tf.GradientTape(persistent=True) as tape:
            loss_qs = self.qs(obs, actions) * qd

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

    def init_optimizer(self, *args, **kwargs):
        self._q1.init_optimizer(*args, **kwargs)
        self._q2.init_optimizer(*args, **kwargs)
        self._target_q1.init_optimizer(*args, **kwargs)
        self._target_q2.init_optimizer(*args, **kwargs)


class GaussianSoftActor(VarSigmaActor):
    def __init__(self, action_space, *args, target_entropy=None, nn_std=True, clip_mean=2, **kwargs):
        super().__init__(action_space, *args, custom_optimization=True, nn_std=nn_std, clip_mean=clip_mean, **kwargs)
        if target_entropy is None:
            target_entropy = -np.prod(action_space.shape)
        self._target_entropy = target_entropy
        self._log_entropy_coef = tf.Variable(1., dtype=tf.float32)

        self.register_method('entropy_coef', self.entropy_coef, {})

        self.register_method('optimize', self.optimize, {
            'obs': 'base.first_obs',
        })

        self.register_method('next_actions', self.sample_inplace, {
            'obs': 'obs_next'
        })

        self.register_method('next_log_prob', self.expected_action_log_prob, {
            'obs': 'obs_next',
            'act': 'self.next_actions'
        })

    @tf.function
    def sample_inplace(self, obs):
        dist = self._dist(obs)
        return dist.sample()
    
    @tf.function
    def expected_action_log_prob(self, obs, act):
        dist = self._dist(obs)
        return dist.log_prob(act)

    @tf.function
    def entropy_coef(self):
        return tf.exp(self._log_entropy_coef)

    @tf.function
    def loss(self, obs):
        mean, std = self.mean_and_std(obs)
        dist = tfp.distributions.MultivariateNormalDiag(
            loc=mean,
            scale_diag=std
        )

        actions = dist.sample()

        log_probs = dist.log_prob(actions)
        entropy_coef = tf.exp(self._log_entropy_coef)

        qs = self.call_now('critic.qs', {'obs': obs, 'actions': actions})
        min_q = tf.reduce_min(qs, axis=-1)

        entropy_loss = -tf.reduce_mean(self._log_entropy_coef * tf.stop_gradient(self._target_entropy + log_probs))
        actor_loss = tf.reduce_mean(tf.stop_gradient(entropy_coef) * log_probs - min_q)

        return entropy_loss + actor_loss


class SAC(BaseNextGenACERAgent):
    def __init__(self, *args, **kwargs):
        if 'buffer.n' in kwargs:
            assert kwargs['buffer.n'] == 1
        else:
            kwargs['buffer.n'] = 1

        actors = {'simple': {False: GaussianSoftActor}}
        critics = {'simple': TwinQDelayedCritic}
        buffers = {'simple': (MultiReplayBuffer, {'buffer_class': VecReplayBuffer})}

        super().__init__(*args, actors=actors, critics=critics, buffers=buffers, **kwargs)

    def _init_automodel(self, skip=()):

        super()._init_automodel(skip=skip)

    def _init_critic(self):
        # if self._is_obs_discrete:
        #     return TabularCritic(self._observations_space, None, self._tf_time_step)
        # else:
        return self.CRITICS[self._critic_type](
            self._observations_space, self._actions_space,
            self._critic_layers, self._tf_time_step, **self._critic_args
        )

