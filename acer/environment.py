from abc import ABC, abstractmethod
from typing import List, Tuple, Optional

import cv2
import gym
import numpy as np

import utils


class WarpFrame(gym.ObservationWrapper):
    """ based on atari_wrappers.WarpFrame, but does not convert to grayscale
    """

    def __init__(self, env, dim):
        """Warp frames to the specified size (dim x dim)."""
        gym.ObservationWrapper.__init__(self, env)
        self.width = dim
        self.height = dim
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(self.height, self.width, 3),
            dtype=np.uint8)

    def observation(self, frame):
        frame = cv2.resize(
            frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
        return frame


class MaxEnv(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        self.prev = None

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        max_obs = np.max([self.prev, obs], 0)
        self.prev = obs
        return max_obs, rew, done, info

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        self.prev = obs
        return obs


class LuminanceWrapper(gym.Wrapper):

    def __init__(self, env):
        super(LuminanceWrapper, self).__init__(env)
        self.observation_space = gym.spaces.Box(
            low=np.min(self.observation_space.low),
            high=np.max(self.observation_space.high),
            shape=self.observation_space.shape[:-1] + (self.observation_space.shape[-1] + 1,),
            dtype=self.observation_space.dtype
        )

        # coefficients are based on https://en.wikipedia.org/wiki/Grayscale
        self.coeffs = np.array([0.2126, 0.7152, 0.0722], dtype=self.observation_space.dtype)

    def get_with_luminance(self, obs):
        return np.concatenate((obs, np.sum(obs * self.coeffs, axis=-1, keepdims=True)), axis=-1)

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        return self.get_with_luminance(obs)

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        return self.get_with_luminance(obs), rew, done, info


class NoopResetEnv(gym.Wrapper):
    def __init__(self, env, noop_max=30):
        """Sample initial states by taking random number of no-ops on reset.
        No-op is assumed to be action 0.
        """
        gym.Wrapper.__init__(self, env)
        self.noop_max = noop_max
        self.override_num_noops = None
        self.noop_action = 0
        assert env.unwrapped.get_action_meanings()[0] == 'NOOP'

    def reset(self, **kwargs):
        """ Do no-op action for a number of steps in [1, noop_max]."""
        self.env.reset(**kwargs)
        if self.override_num_noops is not None:
            noops = self.override_num_noops
        else:
            noops = self.unwrapped.np_random.randint(1, self.noop_max + 1) #pylint: disable=E1101
        assert noops > 0
        obs = None
        for _ in range(noops):
            obs, _, done, _ = self.env.step(self.noop_action)
            if done:
                obs = self.env.reset(**kwargs)
        return obs

    def step(self, ac):
        return self.env.step(ac)


def wrap_atari(env, dim=84):
    env = WarpFrame(env, dim)
    env = NoopResetEnv(env, noop_max=30)
    env = MaxEnv(env)
    env = LuminanceWrapper(env)
    return env


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
        if utils.is_atari(env_id):
            self._envs = [wrap_atari(gym.make(self._env_id)) for _ in range(self._nenvs)]
        else:
            self._envs = [gym.make(self._env_id) for _ in range(self._nenvs)]
        self.dtype = dtype

    def step(self, actions: np.array) -> List[Tuple[np.array, float, bool, dict]]:
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
