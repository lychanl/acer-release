from typing import Tuple, List

import gym
import numpy as np


def flatten_experience(buffers_batches):
    observations = np.concatenate([batch['observations'] for batch in buffers_batches], axis=0)
    next_observations = np.concatenate([batch['next_observations'] for batch in buffers_batches], axis=0)
    actions = np.concatenate([batch['actions'] for batch in buffers_batches], axis=0)

    return observations, next_observations, actions


def unflatten_batches(values, values_next, policies, buffers_batches):

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


def reset_env_and_agent(agent, env):
    agent.reset()
    current_states = env.reset_all()
    return current_states
