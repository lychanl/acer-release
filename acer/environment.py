from abc import ABC, abstractmethod
from typing import List, Tuple, Optional

import gym
import numpy as np


class BaseMultiEnv(ABC):
    """Abstract multi environment class. This is wrapper over set of gym environments to run
    them in parallel. Provides same interface as gym's Env class.
    note: running step automatically resets environment if 'done' flag is received as True

    TODO: write tests
    """

    def __init__(self, env_id: str, n_envs: int, dtype: np.dtype = np.float32):
        """Creates 'n_envs' gym environments.

        Args:
            env_id: Name of environment to instantiate
            n_envs: Number of environments to instantiate
            dtype: type of returned observations
        """
        self._env_id = env_id
        self._nenvs = n_envs
        self._envs = [gym.make(self._env_id) for _ in range(self._nenvs)]
        self.dtype = dtype

    def step(self, actions: list) -> List[Tuple[np.array, float, bool, dict]]:
        """Performs one step in all of the created environments.
        note;

        Args:
            actions: 2D Numpy array of actions to perform, one action vector per environment

        Returns:
            List of tuples from each environment '.step' call. Each tuple contains containing:
                * next state vector
                * reward
                * 'done' indicator
                * additional info dictionary

        """
        assert len(actions) == self._nenvs, f"'actions' should contain one vector per environment " \
                                            f"(expected: {self._nenvs}, passed: {len(actions)})"

        results = self._multi_step(actions)
        return results

    @abstractmethod
    def _multi_step(self, actions: list) -> List[Tuple[np.array, float, bool, dict]]:
        ...

    def reset(self, env_idx: int) -> np.array:
        """Reset one environment
        Sets seeds to the flushed environments if seed is passed

        Returns:
            vectors of initial states
        """

        return self._envs[env_idx].reset().astype(self.dtype)

    def reset_all(self, seeds: Optional[List[int]] = None) -> np.array:
        """Resets all of the environments.
        Sets seeds to the flushed environments if 'seeds' list is passed.

        Args:
            seeds: list of seeds to be set up in fresh environments
        Returns:
            vectors of initial states for each environment
        """
        assert not seeds or len(seeds) == self._nenvs, f"'seeds' list should contain {self._nenvs} values " \
                                                       f"(found: {len(seeds)})"

        states = []

        if seeds:
            for env, seed in zip(self._envs, seeds):
                states.append(env.reset())
                env.seed(seed)
        else:
            for env in self._envs:
                states.append(env.reset())

        return np.array(states).astype(self.dtype)

    @property
    def observation_space(self) -> gym.spaces.Space:
        """
        Returns:
             environment's observation Space object
        """
        return self._envs[0].observation_space

    @property
    def action_space(self) -> gym.spaces.Space:
        """
         Returns:
            environment's actions Space object
        """
        return self._envs[0].action_space

    @property
    def spec(self) -> gym.envs.registration.EnvSpec:
        """
        Returns:
            environment's EnvSpec object

        """
        return self._envs[0].spec

    def __repr__(self):
        return f"{self.__class__.__name__}(env_id='{self._env_id}', nenvs={self._nenvs})"


class SequentialEnv(BaseMultiEnv):
    """Simulates parallel environments execution with sequential calls to the environments in a loop"""

    def _multi_step(self, actions: list) -> List[Tuple[np.array, float, bool, dict]]:
        results = []
        for env, action in zip(self._envs, actions):
            step_results = env.step(action)
            results.append((
                step_results[0].astype(self.dtype),
                self.dtype(step_results[1]),
                step_results[2],
                step_results[3]
            ))

        return results
