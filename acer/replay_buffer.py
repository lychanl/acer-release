from __future__ import annotations
import pickle
from dataclasses import dataclass
import functools
from typing import Optional, Union, Tuple, List, Dict, Type, Callable

import numpy as np
from numpy.core.defchararray import array


@dataclass
class BufferFieldSpec:
    """Specification of the replay buffer's data"""
    shape: tuple
    dtype: Union[Type, np.dtype] = np.float32


class ReplayBuffer:
    def __init__(
            self, max_size: int, action_spec: BufferFieldSpec,
            obs_spec: BufferFieldSpec, policies_spec: BufferFieldSpec = None):
        """Stores trajectories.

        Each of the samples stored in the buffer has the same probability of being drawn.

        Args:
            max_size: maximum number of entries to be stored in the buffer - only newest entries are stored.
            action_spec: BufferFieldSpec with actions space specification
            obs_spec: BufferFieldSpec with observations space specification
        """
        self._max_size = max_size
        self._current_size = 0
        self._pointer = 0

        self._actions = self._init_field(action_spec)
        self._obs = self._init_field(obs_spec)
        self._obs_next = self._init_field(obs_spec)
        self._rewards = self._init_field(BufferFieldSpec((), np.float32))
        self._policies = self._init_field(policies_spec or BufferFieldSpec((), np.float32))
        self._dones = self._init_field(BufferFieldSpec((), bool))
        self._ends = self._init_field(BufferFieldSpec((), bool))

    def _init_field(self, field_spec: BufferFieldSpec):
        shape = (self._max_size, *field_spec.shape)
        return np.zeros(shape=shape, dtype=field_spec.dtype)

    def put(self, action: Union[int, float, list], observation: np.array, next_observation: np.array,
            reward: float, policy: float, is_done: bool,
            end: bool):
        """Stores one experience tuple in the buffer.

        Args:
            action: performed action
            observation: state in which action was performed
            next_observation: next state
            reward: earned reward
            policy: probability/probability density of executing stored action
            is_done: True if episode ended and was not terminated by reaching
                the maximum number of steps per episode
            end: True if episode ended

        """
        self._actions[self._pointer] = action
        self._obs[self._pointer] = observation
        self._obs_next[self._pointer] = next_observation
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

    def get(self, trajectory_len: Optional[int] = None) -> Tuple[Dict[str, np.array], int]:
        """Samples random trajectory. Trajectory length is truncated to the 'trajectory_len' value.
        If trajectory length is not provided, whole episode is fetched. If trajectory is shorter than
        'trajectory_len', trajectory is truncated to one episode only.

        Returns:
            Tuple with dictionary with truncated experience trajectory and first (middle) index
        """
        if self._current_size == 0:
            # empty buffer
            return ({
                "actions": np.array([]),
                "observations": np.array([]),
                "rewards": np.array([]),
                "next_observations": np.array([]),
                "policies": np.array([]),
                "dones": np.array([]),
                "ends": np.array([])
            }, -1)
        sample_index = self._sample_random_index()
        start_index, end_index = self._get_indices(sample_index, trajectory_len)

        batch = self._fetch_batch(end_index, sample_index, start_index)
        return batch
    
    def get_vec(self, length: int, trajectory_len: int) -> Tuple[Dict[str, np.array], int]:
        raise NotImplementedError()

    def _fetch_batch(self, end_index: int, sample_index: int, start_index: int) -> Tuple[Dict[str, np.array], int]:

        middle_index = sample_index - start_index \
            if sample_index >= start_index else self._max_size - start_index + sample_index
        if start_index >= end_index:
            buffer_slice = np.r_[start_index: self._max_size, 0: end_index]
        else:
            buffer_slice = np.r_[start_index: end_index]\
        
        return self._fetch_slice(buffer_slice), middle_index

    def _fetch_slice(self, buffer_slice) -> Dict[str, np.array]:
        actions = self._actions[buffer_slice]
        observations = self._obs[buffer_slice]
        rewards = self._rewards[buffer_slice]
        next_observations = self._obs_next[buffer_slice]
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

    def _get_indices(self, sample_index: int, trajectory_len: int) -> Tuple[int, int]:
        start_index = sample_index
        end_index = sample_index + 1
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
        return start_index, end_index

    def _sample_random_index(self, size=None) -> int:
        """Uniformly samples one index from the buffer"""
        return np.random.randint(low=0, high=self._current_size, size=size)

    @property
    def size(self) -> int:
        """Returns number of steps stored in the buffer

        Returns:
            size of buffer
        """
        return self._current_size

    def __repr__(self):
        return f"{self.__class__.__name__}(max_size={self._max_size})"


class WindowReplayBuffer(ReplayBuffer):

    def __init__(self, max_size: int, action_spec: BufferFieldSpec, obs_spec: BufferFieldSpec):
        """Extends ReplayBuffer to fetch 'window' of experiences around selected index.
        Also policies buffer is replaced to store data in same specification as actions
        (used in algorithms with autocorrelated actions).

        Each of the samples stored in the buffer has the same probability of being drawn.

        Args:
            max_size: maximum number of entries to be stored in the buffer - only newest entries are stored.
            action_spec: BufferFieldSpec with actions space specification
            obs_spec: BufferFieldSpec with observations space specification
        """
        super().__init__(max_size, action_spec, obs_spec, self._init_field(action_spec))

    def _get_indices(self, sample_index: int, trajectory_len: int) -> Tuple[int, int]:
        current_length = 0
        start_index = sample_index
        end_index = sample_index + 1
        is_start_finished = False
        is_end_finished = False
        while True:
            if trajectory_len is not None and current_length == trajectory_len:
                break

            if not is_end_finished:
                if end_index == self._max_size:
                    if self._pointer == 0:
                        is_end_finished = True
                    else:
                        end_index = 0

            if not is_start_finished:
                if start_index == self._pointer or (start_index != 0 and self._ends[start_index - 1]):
                    is_start_finished = True

            if not is_end_finished:
                if end_index == self._pointer or self._ends[end_index]:
                    is_end_finished = True

            if not is_start_finished:
                if start_index == 0:
                    if self._current_size < self._max_size or self._pointer == self._max_size - 1:
                        is_start_finished = True
                    else:
                        start_index = self._max_size - 1
                else:
                    start_index -= 1

            if not is_end_finished:
                end_index += 1
            current_length += 1
        return start_index, end_index


def PrevReplayBuffer(n: int = 1):
    return functools.partial(VecReplayBuffer, n=n)


# class _PrevReplayBuffer(ReplayBuffer):

#     def __init__(self, n: int, max_size: int, action_spec: BufferFieldSpec, obs_spec: BufferFieldSpec):
#         """Extends ReplayBuffer to fetch a number of experience tuples before selected index.
#         If selected index indicates first time step in a episode, no additional tuples are attached.

#         Also policies buffer is replaced to store data in same specification as actions
#         (used in algorithms with autocorrelated actions).

#         Args:
#             n: Number of previous experience tuples to be fetched
#             max_size: maximum number of entries to be stored in the buffer - only newest entries are stored.
#             action_spec: BufferFieldSpec with actions space specification
#             obs_spec: BufferFieldSpec with observations space specification
#         """
#         super().__init__(max_size, action_spec, obs_spec)
#         self._policies = self._init_field(action_spec)
#         self._n = n

#     def get(self, trajectory_len: Optional[int] = None) -> Tuple[Dict[str, np.array], int]:
#         if self._current_size == 0:
#             # empty buffer
#             return ({
#                 "actions": np.array([]),
#                 "observations": np.array([]),
#                 "rewards": np.array([]),
#                 "next_observations": np.array([]),
#                 "policies": np.array([]),
#                 "dones": np.array([]),
#                 "ends": np.array([])
#             }, -1)

#         sample_index = self._sample_random_index()
        
#         n = 0
#         """
#         for _n in range(1, self._n + 1):
#             index = sample_index - _n
#             if index < 0:
#                 if self._current_size < self._max_size:
#                     break
#                 index = self._max_size + index
#             if self._ends[index] or self._pointer == index:
#                 break

#             n = _n
#         """

#         n = self._n

#         if sample_index - n < self._pointer <= sample_index:
#             n = sample_index - self._pointer

#         if sample_index - n < 0:
#             if self._current_size < self._max_size:
#                 n = sample_index
#             elif self._pointer > sample_index - n + self._max_size:
#                 n = self._max_size + sample_index - self._pointer
        
#         if sample_index - n >= 0:
#             r = np.r_[sample_index - n]
#         else:
#             r = np.r_[sample_index:self._max_size, 0:self._max_size]
#         ends = self._ends[r]
#         if ends.any():
#             n = n - np.nonzero(ends)[-1][0] - 1

#         start_index, end_index = self._get_indices(sample_index, trajectory_len)
        
#         start_index = (start_index - n) % self._max_size

#         batch = self._fetch_batch(end_index, sample_index, start_index)
#         return batch


class MultiReplayBuffer:

    def __init__(self, max_size: int, num_buffers: int, action_spec: BufferFieldSpec, obs_spec: BufferFieldSpec,
                buffer_class: Callable[[int, BufferFieldSpec, BufferFieldSpec], ReplayBuffer] = ReplayBuffer,
                policy_spec: BufferFieldSpec = None, *args, **kwargs):
        """Encapsulates ReplayBuffers from multiple environments.

        Args:
            max_size: maximum number of entries to be stored in the buffer - only newest entries are stored.
            action_spec: BufferFieldSpec with actions space specification
            obs_spec: BufferFieldSpec with observations space specification
            buffer_class: class of a buffer to be created
        """
        self._n_buffers = num_buffers
        self._max_size = max_size

        # assert issubclass(buffer_class, ReplayBuffer), "Buffer class should derive from ReplayBuffer"

        self._buffers = [
            buffer_class(int(max_size / num_buffers), action_spec, obs_spec, policy_spec, *args, **kwargs) for _ in range(num_buffers)
        ]

    def put(self, steps: List[Tuple[Union[int, float, list], np.array, float, np.array, bool, bool]]):
        """Stores experience in the buffers. Accepts list of steps.

        Args:
            steps: List of steps, each of step Tuple consists of: action, observation, reward,
             next_observation, policy, is_done flag, end flag. See ReplayBuffer.put() for format description

        """
        assert len(steps) == self._n_buffers, f"'steps' list should provide one step (experience) for every buffer" \
                                              f"(found: {len(steps)} steps, expected: {self._n_buffers}"

        for buffer, step in zip(self._buffers, steps):
            buffer.put(*step)

    def get(self, trajectories_lens: Optional[List[int]] = None) -> List[Tuple[Dict[str, Union[np.array, list]], int]]:
        """Samples trajectory from every buffer.

        Args:
            trajectories_lens: List of maximum lengths of trajectory to be fetched from corresponding buffer

        Returns:
            list of sampled trajectory from every buffer, see ReplayBuffer.get() for the format description
        """
        assert trajectories_lens is None or len(trajectories_lens) == self._n_buffers,\
            f"'steps' list should provide one step (experience) for every buffer" \
            f" (found: {len(trajectories_lens)} steps, expected: {self._n_buffers}"

        if trajectories_lens:
            samples = [buffer.get(trajectory_len) for buffer, trajectory_len in zip(self._buffers, trajectories_lens)]
        else:
            samples = [buffer.get() for buffer in self._buffers]

        return samples
    
    def get_vec(self, length_per_buffer: int, trajectory_len: int) -> Tuple[Dict[str, np.array], np.array]:
        vecs = [buffer.get_vec(length_per_buffer, trajectory_len) for buffer in self._buffers]

        batch = {
            key: np.concatenate([vec[0][key] for vec in vecs])
            for key in vecs[0][0].keys()
        }

        batch["time"] *= self._n_buffers
        lens = np.concatenate([vec[1] for vec in vecs])

        if len(vecs[0]) == 2:
            return batch, lens
        else:
            return batch, lens, np.concatenate([vec[2] for vec in vecs])

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

    def save(self, path: str):
        """Dumps the buffer onto the disk.

        Args:
            path: file path to the new file
        """

        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str) -> MultiReplayBuffer:
        """Loads the buffer from the disk

        Args:
            path: path where the buffer was stored
        """
        with open(path, 'rb') as f:
            buffer = pickle.load(f)
        return buffer


class VecReplayBuffer(ReplayBuffer):

    def __init__(
        self, max_size: int, action_spec: BufferFieldSpec, obs_spec: BufferFieldSpec,
        policy_spec: BufferFieldSpec = None, n: int = None):
        """Extends ReplayBuffer to fetch a number of experience tuples before selected index.
        If selected index indicates first time step in a episode, no additional tuples are attached.

        Also policies buffer is replaced to store data in same specification as actions
        (used in algorithms with autocorrelated actions).

        Args:
            n: Number of previous experience tuples to be fetched
            max_size: maximum number of entries to be stored in the buffer - only newest entries are stored.
            action_spec: BufferFieldSpec with actions space specification
            obs_spec: BufferFieldSpec with observations space specification
        """
        super().__init__(max_size, action_spec, obs_spec, policy_spec or action_spec)
        self._n = n

    def _zero(self):
        return ({
            "actions": np.array([]),
            "observations": np.array([]),
            "rewards": np.array([]),
            "next_observations": np.array([]),
            "policies": np.array([]),
            "dones": np.array([]),
            "ends": np.array([]),
            "time": np.array([]),
        }, -1)

    def get(self, trajectory_len: Optional[int] = None) -> Tuple[Dict[str, np.array], int]:
        if self._current_size == 0:
            # empty buffer
            return self._zero()

        sample_index = self._sample_random_index()
        n = self._n

        if sample_index - n < self._pointer <= sample_index:
            n = sample_index - self._pointer

        if sample_index - n < 0:
            if self._current_size < self._max_size:
                n = sample_index
            elif self._pointer > sample_index - n + self._max_size:
                n = self._max_size + sample_index - self._pointer
        
        if sample_index - n >= 0:
            r = np.r_[sample_index - n:sample_index]
        else:
            r = np.r_[sample_index:self._max_size, 0:self._max_size]
        ends = self._ends[r]
        if ends.any():
            n = n - np.nonzero(ends)[-1][0] - 1

        start_index, end_index = self._get_indices(sample_index, trajectory_len)
        
        start_index = (start_index - n) % self._max_size

        batch = self._fetch_batch(end_index, sample_index, start_index)
        return batch

    def _zero_vec(self, length):
        return ({
            "actions": np.zeros((length, 0)),
            "observations": np.zeros((length, 0)),
            "rewards": np.zeros((length, 0)),
            "next_observations": np.zeros((length, 0)),
            "policies": np.zeros((length, 0)),
            "dones": np.zeros((length, 0)),
            "ends": np.zeros((length, 0)),
            "time": np.zeros(length)
        }, np.repeat(-1, length))

    def get_vec(self, length: int, trajectory_len: int) -> Tuple[Dict[str, np.array], np.array]:
        if self._current_size == 0:
            # empty buffer
            return self._zero_vec(length)

        sample_indices = self._sample_random_index(length)

        return self._get_sampled_vec(length, trajectory_len, sample_indices)

    def _get_sampled_vec(self, length, trajectory_len, sample_indices):
        prev_lens = np.repeat(self._n, length)

        if self._n is not None:
            prev_pointer_ind = np.logical_and(sample_indices - prev_lens < self._pointer, self._pointer <= sample_indices)
            prev_lens[prev_pointer_ind] = sample_indices[prev_pointer_ind] - self._pointer

            if self._current_size < self._max_size:
                prev_current_size_ind = sample_indices - prev_lens < 0
                prev_lens[prev_current_size_ind] = sample_indices[prev_current_size_ind]

            prev_pointer_ovf_ind = np.logical_and(sample_indices - prev_lens < 0, self._pointer > sample_indices - prev_lens + self._max_size)
            prev_lens[prev_pointer_ovf_ind] = (self._max_size - self._pointer) + sample_indices[prev_pointer_ovf_ind]


        lens = np.repeat(trajectory_len, length)

        pointer_ind = np.logical_and(sample_indices < self._pointer, self._pointer < sample_indices + lens)
        lens[pointer_ind] = self._pointer - sample_indices[pointer_ind]

        if self._current_size < self._max_size:
            current_size_ind = np.logical_and(sample_indices + lens > self._current_size, self._current_size < self._max_size)
            lens[current_size_ind] = self._current_size - sample_indices[current_size_ind]

        pointer_ovf_ind = np.logical_and(sample_indices + lens > self._max_size, self._pointer < sample_indices + lens - self._max_size)
        lens[pointer_ovf_ind] = (self._pointer + self._max_size) - sample_indices[pointer_ovf_ind]

        n = self._n or 0
        selection = (np.repeat(np.expand_dims(sample_indices, 1), n + trajectory_len, axis=1) + np.arange(-n, trajectory_len))
        selection = selection % self._max_size

        batch = self._fetch_slice(selection)

        ends_mask = np.cumsum(batch['ends'][:,self._n:-1], axis=1) == 0
        lens = np.minimum(lens, ends_mask.sum(axis=1) + 1)

        if self._n is not None:
            prev_ends_mask = np.flip(np.cumsum(np.flip(batch['ends'][:,:self._n], 1), axis=1), 1) == 0
            prev_lens = np.minimum(prev_lens, prev_ends_mask.sum(axis=1))
            mask = np.concatenate([prev_ends_mask, np.ones((length, 1)), ends_mask], axis=1)
        else:
            mask = np.concatenate([np.ones((length, 1)), ends_mask], axis=1)
        

        for k, v in batch.items():
            m = mask
            while len(m.shape) < len(v.shape):
                m = np.expand_dims(m, -1)
            batch[k] = v * m

        time = ((self._pointer - 1) - sample_indices) % self._max_size
        batch['time'] = np.asarray(time).astype(np.int32)

        return (batch, lens, prev_lens) if self._n is not None else (batch, lens)


class PrioritizedReplayBuffer(VecReplayBuffer):
    def __init__(
            self, max_size: int, action_spec: BufferFieldSpec, obs_spec: BufferFieldSpec, policy_spec: BufferFieldSpec,
            block: int, levels: int = 2):
        super().__init__(max_size, action_spec, obs_spec, policy_spec=policy_spec, n=None)
        self._udpate_pointer = 0

        self._block = block
        self._levels = levels

        self._rands = None

        self._priorities = []
        self._priorities_cumsums = []
        self._block_starts = []
        self._block_ends = []
        self._total_priorities = 0

        self._level_block_size = int((max_size // block) ** (1 / levels))
        if self._level_block_size ** levels * block < max_size:
            self._level_block_size += 1
        level_blocks = max_size
        for level in range(levels + 1):

            self._priorities.append(np.zeros(level_blocks, dtype=np.float32))
            self._priorities_cumsums.append(np.zeros(level_blocks, dtype=np.float32))

            prev_level_block = level_blocks
            block_size = self._level_block_size if level > 0 else block
            level_blocks = level_blocks // block_size + (level_blocks % block_size != 0)

            self._block_starts.append(np.arange(level_blocks) * block_size)
            self._block_ends.append(np.minimum((np.arange(level_blocks) + 1) * block_size, prev_level_block))

    def put(self, action: Union[int, float, list], observation: np.array, next_observation: np.array, reward: float, policy: float, is_done: bool, end: bool):
        self._priorities[0][self._pointer] = 1
        self._update_block_priorities(self._pointer // self._block)
        return super().put(action, observation, next_observation, reward, policy, is_done, end)

    def _update_block_priorities(self, block):
        for level in range(self._levels + 1):
            block_start = self._block_starts[level][block]
            block_end = self._block_ends[level][block]
            
            self._priorities_cumsums[level][block_start:block_end] = np.cumsum(self._priorities[level][block_start:block_end])
            
            if level < self._levels:
                self._priorities[level + 1][block] = np.sum(self._priorities[level][block_start:block_end])
            else:
                self._total_priorities = self._priorities[-1].sum()

            block = block // self._level_block_size

    def _sample_random_index(self, size=None) -> int:

        rands = np.random.random(size=size) * self._total_priorities
        self._rands = rands
        return self._sample_indices_from_rands(rands)

    def _sample_indices_from_rands(self, rands):
        ids = np.zeros_like(rands, dtype=int)

        for i, r in enumerate(rands):
            block = 0
            for level in reversed(range(self._levels + 1)):
                start = self._block_starts[level][block]
                end = self._block_ends[level][block]
                block = min(np.searchsorted(self._priorities_cumsums[level][start:end], r) + start, end - 1)
                if block > start:
                    r = r - self._priorities_cumsums[level][block - 1]
            ids[i] = min(block, self._current_size - 1)

        return ids

    def _fetch_slice(self, buffer_slice) -> Dict[str, np.array]:
        batch = super()._fetch_slice(buffer_slice)
        batch["priors"] = self._priorities[0][buffer_slice]

        return batch

    def _zero(self):
        return ({
            "priors": np.array([]),
            "actions": np.array([]),
            "observations": np.array([]),
            "rewards": np.array([]),
            "next_observations": np.array([]),
            "policies": np.array([]),
            "dones": np.array([]),
            "ends": np.array([]),
            "time": np.array([]),
        }, -1)

    def _zero_vec(self, length):
        return ({
            "priors": np.ones((length, 0)),
            "actions": np.zeros((length, 0)),
            "observations": np.zeros((length, 0)),
            "rewards": np.zeros((length, 0)),
            "next_observations": np.zeros((length, 0)),
            "policies": np.zeros((length, 0)),
            "dones": np.zeros((length, 0)),
            "ends": np.zeros((length, 0)),
            "time": np.zeros(length)
        }, np.repeat(-1, length))

    def should_update_block(self):
        return self._current_size >= self._block

    def get_next_block_to_update(self, trajectory_len):
        return self._get_sampled_vec(
            self._block, trajectory_len,
            np.arange(self._block_starts[0][self._udpate_pointer], self._block_ends[0][self._udpate_pointer])
        )

    def update_block(self, priorities):
        self._priorities[0][(self._udpate_pointer * self._block):((self._udpate_pointer + 1) * self._block)] = priorities
        self._update_block_priorities(self._udpate_pointer)
        self._udpate_pointer = (self._udpate_pointer + 1) % (self._current_size // self._block)


class MultiPrioritizedReplayBuffer(MultiReplayBuffer):
    def __init__(self, max_size: int, num_buffers: int, action_spec: BufferFieldSpec, obs_spec: BufferFieldSpec, policy_spec: BufferFieldSpec, *args, **kwargs):
        super().__init__(max_size, num_buffers, action_spec, obs_spec, buffer_class=PrioritizedReplayBuffer, policy_spec=policy_spec, *args, **kwargs)

    def should_update_block(self):
        return all(buf.should_update_block() for buf in self._buffers)

    def get_next_block_to_update(self, trajectory_len):
        ret = dict()
        lens = []
        for buf in self._buffers:
            batch, ls = buf.get_next_block_to_update(trajectory_len)
            lens.append(ls)
            for k, v in batch.items():
                ret[k] = ret.get(k, []) + [v]

        for k, v in ret.items():
            ret[k] = np.concatenate(ret[k])

        return ret, np.concatenate(lens)

    def update_block(self, priorities):
        for p, buf in zip(priorities.reshape((self._n_buffers, -1)), self._buffers):
            buf.update_block(p)
