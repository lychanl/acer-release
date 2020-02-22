
from dataclasses import dataclass
from typing import Optional, Union, Tuple, List, Dict, Type

import numpy as np


@dataclass
class BufferFieldSpec:
    """Specification of the replay buffer's data"""
    shape: tuple
    dtype: Union[Type, np.dtype] = np.float32


class ReplayBuffer:
    def __init__(self, max_size: int, action_spec: BufferFieldSpec, obs_spec: BufferFieldSpec):
        """Stores trajectories.

        Each of the sample stored in the buffer has the same probability of being drawn.

        Args:
            max_size: maximum number of entries to be stored in the buffer - only newest entries are stored.
        """
        self._max_size = max_size
        self._current_size = 0
        self._pointer = 0

        self._actions = self._init_field(action_spec)
        self._obs = self._init_field(obs_spec)
        self._rewards = self._init_field(BufferFieldSpec((), np.float32))
        self._policies = self._init_field(BufferFieldSpec((), np.float32))
        self._dones = self._init_field(BufferFieldSpec((), bool))
        self._ends = self._init_field(BufferFieldSpec((), bool))

    def _init_field(self, field_spec: BufferFieldSpec):
        shape = (self._max_size, *field_spec.shape)
        return np.ndarray(shape=shape, dtype=field_spec.dtype)

    def put(self, action: Union[int, float, list], observation: np.array,
            reward: float, policy: float, is_done: bool,
            end: bool):
        """Stores one experience tuple in the buffer.

        Args:
            action: performed action
            observation: state in which action was performed
            reward: earned reward
            policy: probability/probability density of executing stored action
            is_done: True if episode ended and was not terminated by reaching
                the maximum number of steps per episode
            end: True if episode ended

        """
        self._actions[self._pointer] = action
        self._obs[self._pointer] = observation
        self._rewards[self._pointer] = reward
        self._policies[self._pointer] = policy
        self._dones[self._pointer] = is_done
        self._ends[self._pointer] = end

        self._move_pointer()
        self._update_size()

    def _move_pointer(self):
        if self._pointer + 1 == self._max_size:
            self._pointer = 0
        else:
            self._pointer += 1

    def _update_size(self):
        self._current_size = np.min([self._current_size + 1, self._max_size])

    def get(self, trajectory_len: Optional[int] = None) -> Dict[str, np.array]:
        """Samples random trajectory. Trajectory length is truncated to the 'trajectory_len' value.
        If trajectory length is not provided, whole episode is fetched. If trajectory is shorter than
        'trajectory_len', trajectory is truncated to one episode only.

        Returns:
            Dictionary with single, truncated experience trajectory
        """
        if self._current_size == 0:
            # empty buffer
            return {
                "actions": np.array([]),
                "observations": np.array([]),
                "rewards": np.array([]),
                "next_observations": np.array([]),
                "policies": np.array([]),
                "dones": np.array([]),
                "ends": np.array([])
            }
        start_index = self._sample_random_index()
        end_index = start_index + 1

        current_length = 1
        while True:
            if end_index == self._max_size:
                if self._pointer == 0:
                    break
                else:
                    end_index = 0
                    continue
            if end_index == self._pointer \
                    or self._ends[end_index - 1] \
                    or (trajectory_len and current_length == trajectory_len):
                break
            end_index += 1
            current_length += 1

        start_index_next = start_index + 1
        end_index_next = end_index + 1

        if start_index_next == self._max_size:
            start_index_next = 0

        if end_index_next == self._max_size + 1:
            end_index_next = 1

        if start_index_next > end_index_next:
            buffer_slice_next = np.r_[0: end_index_next, start_index_next: self._max_size]
        else:
            buffer_slice_next = np.r_[start_index_next: end_index_next]

        if start_index > end_index:
            buffer_slice = np.r_[0: end_index, start_index: self._max_size]
        else:
            buffer_slice = np.r_[start_index: end_index]

        actions = self._actions[buffer_slice]
        observations = self._obs[buffer_slice]
        rewards = self._rewards[buffer_slice]
        next_observations = self._obs[buffer_slice_next]
        policies = self._policies[buffer_slice]
        done = self._dones[buffer_slice]
        end = self._ends[buffer_slice]

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
        """Uniformly samples one index from the buffer"""
        return np.random.randint(low=0, high=self._current_size)

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

    def __init__(self, max_size: int, num_buffers: int, action_spec: BufferFieldSpec, obs_spec: BufferFieldSpec):
        """Encapsulates ReplayBuffers from multiple environments.

        Args:
            max_size: maximum number of entries in a single buffer
            num_buffers: number of buffers to be created
        """
        self._n_buffers = num_buffers
        self._max_size = max_size
        self._buffers = [ReplayBuffer(int(max_size / num_buffers), action_spec, obs_spec) for _ in range(num_buffers)]

    def put(self, steps: List[Tuple[Union[int, float, list], np.array, float, np.array, bool, bool]]):
        """Stores gathered experiences in the buffers. Accepts list of steps.

        Args:
            steps: List of steps, each of step Tuple consists of: action, observation, reward,
             next_observation, policy, is_done flag, end flag. See ReplayBuffer.put() for format description

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
            list of sampled trajectory from every buffer, see ReplayBuffer.get() for format description
        """
        assert trajectories_lens is None or len(trajectories_lens) == self._n_buffers,\
            f"'steps' list should provide one step (experience) for every buffer" \
            f" (found: {len(trajectories_lens)} steps, expected: {self._n_buffers}"

        if trajectories_lens:
            samples = [buffer.get(trajectory_len) for buffer, trajectory_len in zip(self._buffers, trajectories_lens)]
        else:
            samples = [buffer.get() for buffer in self._buffers]

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
