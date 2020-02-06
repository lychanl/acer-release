from typing import Tuple, List, Union, Dict

import gym
import numpy as np
import tensorflow as tf

from environment import BaseMultiEnv


def normc_initializer():
    """Normalized column initializer from the OpenAI baselines"""
    def _initializer(shape, dtype=None, partition_info=None):
        out = np.random.randn(*shape).astype(np.float32)
        out *= 1 / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)
    return _initializer


def tf_function_with_spec_from_object(dimension_attrs: List[str]):
    """Creates tf.function abstraction with dynamic TypeSpec and autograph off"""
    def tf_function_wrapper(f):
        def wrapper(self, *args, **kwargs):
            spec = [tf.TensorSpec(shape=(None, getattr(self, attr))) for attr in dimension_attrs]
            return tf.function(input_signature=spec)(f)(self, *args, **kwargs)
        return wrapper
    return tf_function_wrapper


def flatten_experience(buffers_batches: List[Dict[str, Union[np.array, list]]]) -> Tuple[np.array, np.array, np.array]:
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

    return observations, next_observations, actions


def unflatten_batches(
        values: np.array, values_next: np.array,
        policies: np.array, buffers_batches: List[Dict[str, Union[np.array, list]]]
) -> Tuple[List[np.array], List[np.array], List[np.array]]:
    """Parses flat experience into separate batches per buffer - restores original division.

    Args:
        values: batch [batch_size, 1] of value approximations
        values_next: batch [batch_size, 1] of value approximations of the 'next' states
        policies: batch [batch_size, 1] of probabilities (probability densities)
        buffers_batches: original trajectories sampled sampled from the buffers

    Returns:
        Tuple with matrices:
            * batch [batch_size, 1] of policy function values (probabilities or probability densities)
            * batch [batch_size, 1] of value function approximation for "current" observations
            * batch [batch_size, 1] of value function approximation for "next" observations
    """

    actions_idx, observations_idx = _get_flatten_experience_indices(buffers_batches)

    policies_batches = []
    values_next_batches = []
    values_batches = []

    for i in range(len(buffers_batches)):
        policies_batches.append(policies[actions_idx[i]:actions_idx[i + 1]])
        values_batches.append(values[observations_idx[i]:observations_idx[i + 1]])
        values_next_batches.append(values_next[observations_idx[i]:observations_idx[i + 1]])

    return policies_batches, values_batches, values_next_batches


def _get_flatten_experience_indices(buffers_batches) -> Tuple[List[int], List[int]]:
    """Restores indices of the original division from the experience sampled from the buffers.

    Args:
        buffers_batches: original trajectories sampled sampled from the buffers

    Returns:
        restored indices
    """
    actions_idx = [0]
    states_idx = [0]

    for i in range(1, len(buffers_batches)):
        actions_idx.append(actions_idx[i - 1] + len(buffers_batches[i - 1]['actions']))
        states_idx.append(states_idx[i - 1] + len(buffers_batches[i - 1]['observations']))

    actions_idx.append(actions_idx[-1] + len(buffers_batches[-1]['actions']))
    states_idx.append(actions_idx[-1] + len(buffers_batches[-1]['observations']))

    return actions_idx, states_idx


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
