from collections import deque
from itertools import islice
from typing import Optional, Union, Tuple, List, Dict

import numpy as np


class ReplayBuffer:
    """Stores last trajectories in the memory
    TODO: write tests
    """

    def __init__(self, max_size: int):
        """Initializes queues for trajectories. Each of the sample stored in the buffer
        has the same probability of being drawn.

        Args:
            max_size: maximum number of entries to be stored in the buffer - only newest entries are stored.
        """
        self._max_size = max_size
        self._current_size = 0

        self._actions = deque(maxlen=max_size)
        self._observations = deque(maxlen=max_size)
        self._next_observations = deque(maxlen=max_size)
        self._policies = deque(maxlen=max_size)
        self._rewards = deque(maxlen=max_size)
        self._done = deque(maxlen=max_size)
        self._end = deque(maxlen=max_size)

    def put(self, action: Union[int, float, list], observation: np.array,
            reward: float, next_observation: np.ndarray,
            policy: float, is_done: bool,
            end: bool):
        """Stores one experience sample in the memory.

        Args:
            action: performed action
            observation: state in which action was performed
            reward: earned reward
            next_observation: next state after performing action
            policy: probability/density of stored action
            is_done: True if episode ended and was not terminated by reaching
                the maximum number of steps per episode
            end: True if episode ended

        """
        self._actions.append(action)
        self._observations.append(observation)
        self._rewards.append(reward)
        self._next_observations.append(next_observation)
        self._policies.append(policy)
        self._done.append(is_done)
        self._end.append(end)

        self._current_size = np.min([self._current_size + 1, self._max_size])

    def get(self, trajectory_len: Optional[int] = None) -> Dict[str, Union[np.array, list]]:
        """Samples random trajectory. Trajectory length is truncated to the 'trajectory_len' value.
        If trajectory length is not provided, whole episode is fetched.

        Returns:
            Dictionary with single, truncated experience trajectory
        """
        if self._current_size == 0:
            # empty buffer
            return {
                "actions": np.array([]),
                "observations": [],
                "rewards": [],
                "next_observations": np.array([]),
                "policies": np.array([]),
                "dones": [],
                "ends": []
            }

        start_index = self._sample_random_index()

        if not trajectory_len or start_index + trajectory_len > self._current_size:
            # trajectory end index out of bounds
            end_index = self._current_size
        else:
            end_index = start_index + trajectory_len

        # truncate trajectory to one episode length
        for i, end in enumerate(islice(self._end, start_index, end_index)):
            if end:
                end_index = start_index + i + 1

        actions = list(islice(self._actions, start_index, end_index))
        observations = np.array(list(islice(self._observations, start_index, end_index)))
        rewards = list(islice(self._rewards, start_index, end_index))
        next_observations = np.array(list(islice(self._next_observations, start_index, end_index)))
        policies = np.array(list(islice(self._policies, start_index, end_index)))
        done = list(islice(self._done, start_index, end_index))
        end = list(islice(self._end, start_index, end_index))

        return {
            "actions": actions,
            "observations": observations,
            "rewards": rewards,
            "next_observations": next_observations,
            "policies": policies,
            "dones": done,
            "ends": end
        }

    def get_newest(self, trajectory_len: Optional[int] = None) -> Dict[str, Union[np.array, list]]:
        """Samples random trajectory from the last episode.
        Trajectory length is truncated to the 'trajectory_len' value. If trajectory length
        is not provided, whole episode is fetched.

        TODO: correct implementation, '_sample_random_index_newest' is hardcoded

        Returns:
            Dictionary with single, truncated experience trajectory
        """
        if self._current_size == 0:
            # empty buffer
            return {
                "actions": np.array([]),
                "observations": [],
                "rewards": [],
                "next_observations": np.array([]),
                "policies": np.array([]),
                "dones": [],
                "ends": []
            }

        start_index = self._sample_random_index_newest()

        if not trajectory_len or start_index + trajectory_len > self._current_size:
            # trajectory end index out of bounds
            end_index = self._current_size
        else:
            end_index = start_index + trajectory_len

        # truncate trajectory to one episode length
        for i, end in enumerate(islice(self._end, start_index, end_index)):
            if end:
                end_index = start_index + i + 1

        actions = list(islice(self._actions, start_index, end_index))
        observations = np.array(list(islice(self._observations, start_index, end_index)))
        rewards = list(islice(self._rewards, start_index, end_index))
        next_observations = np.array(list(islice(self._next_observations, start_index, end_index)))
        policies = np.array(list(islice(self._policies, start_index, end_index)))
        done = list(islice(self._done, start_index, end_index))
        end = list(islice(self._end, start_index, end_index))

        return {
            "actions": actions,
            "observations": observations,
            "rewards": rewards,
            "next_observations": next_observations,
            "policies": policies,
            "dones": done,
            "ends": end
        }

    def _sample_random_index(self) -> int:
        """Uniformly samples one index from the buffer

        Returns:
            sampled index
        """
        return np.random.randint(low=0, high=self._current_size)

    def _sample_random_index_newest(self) -> int:
        """Uniformly samples one index from the buffer. Sampled action comes from the last episode.

        TODO: correct implementation, hardcoded '-200' right now

        Returns:
            sampled index
        """
        return np.random.randint(low=np.max([self._current_size - 200, 0]), high=self._current_size)

    @property
    def size(self) -> int:
        """Returns number of steps stored in the buffer

        Returns:
            size of buffer
        """
        return self._current_size

    def __repr__(self):
        return f"{self.__class__.__name__}(max_size={self._max_size})"


class MultiReplayBuffer:
    """Encapsulates ReplayBuffers from multiple environments."""

    def __init__(self, max_size: int, num_buffers: int):
        """Creates ReplayBuffers with proper size.

        Args:
            max_size: maximum number of entries in single buffer
            num_buffers: number of buffers to be created
        """
        self._n_buffers = num_buffers
        self._max_size = max_size
        self._buffers = [ReplayBuffer(max_size) for _ in range(num_buffers)]

    def put(self, steps: List[Tuple[Union[int, float, list], np.array, float, np.array, np.array, bool, bool]]):
        """Stores gathered experiences in the buffers. Accepts list of steps.

        Args:
            steps: List of steps, each of step Tuple consists of: action, observation, reward,
             next_observation, policy, is_done flag, end flag. See ReplayBuffer's put() for format description

        """
        assert len(steps) == self._n_buffers, f"'steps' list should provide one step (experience) for every buffer" \
                                              f"(found: {len(steps)} steps, expected: {self._n_buffers}"

        for buffer, step in zip(self._buffers, steps):
            buffer.put(*step)

    def get(self, trajectories_lens: Optional[List[int]] = None) -> List[Dict[str, Union[np.array, list]]]:
        """Samples trajectory from every buffer.

        Args:
            trajectories_lens: List of maximum lengths of trajectory to be fetched from corresponding buffer
        Returns:
            list of sampled trajectory from every buffer, see ReplayBuffers s .get() for format description
        """
        assert trajectories_lens is None or len(trajectories_lens) == self._n_buffers,\
            f"'steps' list should provide one step (experience) for every buffer" \
            f" (found: {len(trajectories_lens)} steps, expected: {self._n_buffers}"

        if trajectories_lens:
            samples = [buffer.get(trajectory_len) for buffer, trajectory_len in zip(self._buffers, trajectories_lens)]
        else:
            samples = [buffer.get() for buffer in self._buffers]

        return samples

    def get_newest(self, trajectories_lens: Optional[List[int]] = None) -> List[Dict[str, Union[np.array, list]]]:
        """Samples trajectories from the last episode buffers.

        TODO: ReplayBuffer.get_newest to be implemented, right now it is a hardcoded stub

        Args:
            trajectories_lens: List of maximum lengths of trajectory to be fetched from corresponding buffer

        Returns:
            list of sampled trajectory from every buffer, see ReplayBuffers s .get() for format description

        """
        assert trajectories_lens is None or len(trajectories_lens) == self._n_buffers, \
            f"'steps' list should provide one step (experience) for every buffer" \
            f" (found: {len(trajectories_lens)} steps, expected: {self._n_buffers}"

        if trajectories_lens:
            samples = [buffer.get_newest(trajectory_len) for buffer, trajectory_len in
                       zip(self._buffers, trajectories_lens)]
        else:
            samples = [buffer.get_newest() for buffer in self._buffers]

        return samples

    @property
    def size(self) -> List[int]:
        """
        Returns:
            list of buffers' sizes
        """
        return [buffer.size for buffer in self._buffers]

    @property
    def n_buffers(self) -> int:
        """
        Returns:
            number of instantiated buffers
        """
        return self._n_buffers

    def __repr__(self):
        return f"{self.__class__.__name__}(max_size={self._max_size})"

