from functools import wraps
from time import time
from typing import Tuple, List, Union, Dict

import gym
import numpy as np
import tensorflow as tf

from environment import BaseMultiEnv


def normc_initializer():
    """Normalized column initializer from the OpenAI baselines"""
    def _initializer(shape, dtype=None, partition_info=None):
        out = np.random.randn(*shape)
        out *= 1 / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out, dtype=dtype)
    return _initializer


def flatten_experience(buffers_batches: List[Dict[str, Union[np.array, list]]])\
        -> Tuple[np.array, np.array, np.array, np.array, np.array, np.array]:
    """Parses experience from the buffers (from dictionaries) into matrices that can be feed into
    neural network in a single pass.

    Args:
        buffers_batches: trajectories fetched from replay buffers

    Returns:
        Tuple with matrices:
            * batch [batch_size, observations_dim] of observations
            * batch [batch_size, observations_dim] of 'next' observations
            * batch [batch_size, actions_dim] of actions
    """
    observations = np.concatenate([batch['observations'] for batch in buffers_batches], axis=0)
    next_observations = np.concatenate([batch['next_observations'] for batch in buffers_batches], axis=0)
    actions = np.concatenate([batch['actions'] for batch in buffers_batches], axis=0)
    policies = np.concatenate([batch['policies'] for batch in buffers_batches], axis=0)
    rewards = np.concatenate([batch['rewards'] for batch in buffers_batches], axis=0)
    dones = np.concatenate([batch['dones'] for batch in buffers_batches], axis=0)

    return observations, next_observations, actions, policies, rewards, dones


def get_env_variables(env):
    """Returns OpenAI Gym environment specific variables like action space dimension"""
    if type(env.observation_space) == gym.spaces.discrete.Discrete:
        observations_dim = env.observation_space.n
    else:
        observations_dim = env.observation_space.shape[0]
    if type(env.action_space) == gym.spaces.discrete.Discrete:
        continuous = False
        actions_dim = env.action_space.n
        action_scale = 1
    else:
        continuous = True
        actions_dim = env.action_space.shape[0]
        action_scale = np.maximum(env.action_space.high, np.abs(env.action_space.low))
    max_steps_in_episode = env.spec.max_episode_steps
    return action_scale, actions_dim, observations_dim, continuous, max_steps_in_episode


def reset_env_and_agent(agent, env: BaseMultiEnv) -> List[np.array]:
    """Resets environment and the agent

    Args:
        agent: agent to be reset
        env: environment to be reset

    Returns:
        initial states of the environments
    """
    agent.reset()
    current_states = env.reset_all()
    return current_states


class RunningMeanVariance:
    def __init__(self, epsilon: float = 1e-4, shape: Tuple = ()):
        """Computes running mean and variance with Welford's online algorithm (Parallel algorithm)

        Reference:
            https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance

        Args:
            epsilon: small value for numerical stability
            shape: shape of the normalized vector
        """
        self.mean = np.zeros(shape=shape, dtype=np.float32)
        self.var = np.ones(shape=shape, dtype=np.float32)
        self.count = epsilon

    def update(self, x: np.array):
        """Updates statistics with given batch [batch_size, vector_size] of samples

        Args:
            x: batch of samples
        """
        batch_mean = np.mean(x, axis=0, dtype=np.float32)
        batch_var = np.var(x, axis=0, dtype=np.float32)
        batch_count = x.shape[0]

        if self.count < 1:
            self.count, self.mean, self.var = batch_count, batch_mean, batch_var
        else:
            new_count = self.count + batch_count
            delta = batch_mean - self.mean
            new_mean = self.mean + delta * batch_count / new_count

            m_a = self.var * (self.count - 1)
            m_b = batch_var * (batch_count - 1)
            m_2 = m_a + m_b + np.square(delta) * self.count * batch_count / new_count
            new_var = m_2 / (new_count - 1)
            self.count, self.mean, self.var = new_count, new_mean, new_var


class RunningMeanVarianceTf:
    def __init__(self, epsilon: float = 1e-4, shape: Tuple = ()):
        """TensorFlow version of RunningMeanVariance

        Args:
            epsilon: small value for numerical stability
            shape: shape of the normalized vector
        """
        self.mean = tf.zeros(shape=shape)
        self.var = tf.ones(shape=shape)
        self.count = tf.Variable(initial_value=epsilon)

    def update(self, x: tf.Tensor):
        """Updates statistics with given batch [batch_size, vector_size] of samples

        Args:
            x: batch of samples
        """
        batch_mean = tf.reduce_mean(x, axis=0)
        batch_var = tf.math.reduce_variance(x, axis=0)
        batch_count = x.shape[0]

        if tf.math.less(self.count, 1):
            self._assign_new_values(batch_count, batch_mean, batch_var)
        else:
            new_count = self.count + batch_count
            delta = batch_mean - self.mean
            new_mean = self.mean + delta * batch_count / new_count

            m_a = self.var * (self.count - tf.constant(1))
            m_b = batch_var * (batch_count - tf.constant(1))
            m_2 = m_a + m_b + np.square(delta) * self.count * batch_count / new_count
            new_var = m_2 / (new_count - tf.constant(1))
            self._assign_new_values(new_count, new_mean, new_var)

    def _assign_new_values(self, count: tf.Tensor, mean: tf.Tensor, var: tf.Tensor):
        self.count.assign(count)
        self.mean.assign(mean)
        self.var.assign(var)


def timing(f):
    """Function decorator that measures time elapsed while executing a function."""
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        print('func:%r took: %2.4f sec' % (f.__name__, te-ts))
        return result
    return wrap