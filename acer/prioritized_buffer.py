from replay_buffer import BufferFieldSpec, VecReplayBuffer, MultiReplayBuffer

import numpy as np
import tensorflow as tf
from typing import Union, Dict


class PrioritizedReplayBuffer(VecReplayBuffer):
    def __init__(
            self, max_size: int, action_spec: BufferFieldSpec, obs_spec: BufferFieldSpec, policy_spec: BufferFieldSpec,
            block: int, levels: int = 2, *args, **kwargs):
        super().__init__(max_size, action_spec, obs_spec, policy_spec=policy_spec, *args, **kwargs)
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
        batch["priors"] = self._priorities[0][buffer_slice] * self._current_size / self._total_priorities

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
            "priors": np.zeros((length, 0)),
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

    def get_next_block_to_update(self, n):
        return self._get_sampled_vec(
            self._block, n,
            np.arange(self._block_starts[0][self._udpate_pointer], self._block_ends[0][self._udpate_pointer])
        )

    def update_block(self, priorities):
        self._priorities[0][(self._udpate_pointer * self._block):((self._udpate_pointer + 1) * self._block)] = priorities
        self._update_block_priorities(self._udpate_pointer)
        self._udpate_pointer = (self._udpate_pointer + 1) % (self._current_size // self._block)


class MultiPrioritizedReplayBuffer(MultiReplayBuffer):
    def __init__(
            self, max_size: int, num_buffers: int,
            action_spec: BufferFieldSpec, obs_spec: BufferFieldSpec, policy_spec: BufferFieldSpec,
            priority: str, block: int, clip: float = -1, alpha: float = 1, beta: float = 1, *args, **kwargs):
        self._clip = clip
        self._alpha = alpha
        self._beta = beta
        self.block = block * num_buffers

        PRIORITIES = {
            "IS": (self._calculate_IS_priorities, {'density': 'actor.density'}),
            "IS1": (self._calculate_IS1_priorities, {'density': 'actor.density'}),
            "prob": (self._calculate_prob_priorities, {'policies': 'actor.policies', 'mask': 'mask'}),
            "TD": (self._calculate_TD_priorities, {'td': 'base.td',}),
            "oTD": (self._calculate_oTD_priorities, {'td': 'base.td',}),
            "weightedTD": (self._calculate_weighted_TD_priorities, {'td': 'base.td', 'weights': 'actor.density'}),
        }
        
        assert priority in PRIORITIES, f"priority {priority} not in {', '.join(PRIORITIES.keys())}"

        priority = PRIORITIES[priority]

        super().__init__(
            max_size, num_buffers, action_spec, obs_spec,
            buffer_class=PrioritizedReplayBuffer, policy_spec=policy_spec,
            priority_spec=priority, block=block, *args, **kwargs)

    def should_update_block(self):
        return all(buf.should_update_block() for buf in self._buffers)

    def get_next_block_to_update(self):
        ret = dict()
        lens = []
        for buf in self._buffers:
            batch, ls = buf.get_next_block_to_update(self.n)
            lens.append(ls)
            for k, v in batch.items():
                ret[k] = ret.get(k, []) + [v]

        for k, v in ret.items():
            ret[k] = np.concatenate(ret[k])

        return ret, np.concatenate(lens)

    def update_block(self, priorities):
        for p, buf in zip(priorities.reshape((self._n_buffers, -1)), self._buffers):
            buf.update_block(p)

    @tf.function
    def _priorities_postprocess(self, priorities, lower_clip, upper_clip):

        if self._alpha:
            smoothed = priorities ** self._alpha
        else:
            smoothed = priorities

        if lower_clip < 0:
            lower_clip = 0
        if upper_clip < 0:
            upper_clip = tf.float32.max

        return tf.clip_by_value(smoothed, lower_clip, upper_clip)


    @tf.function(experimental_relax_shapes=True)
    def _calculate_IS_priorities(self, density):
        return self._priorities_postprocess(tf.reduce_mean(density, -1), 1 / self._clip, self._clip)

    @tf.function(experimental_relax_shapes=True)
    def _calculate_prob_priorities(self, policies, mask):
        policies_masked = policies * mask + (1 - mask) * tf.ones_like(policies)

        priorities = tf.reshape(tf.reduce_prod(policies_masked, axis=1), (-1,))
        
        return self._priorities_postprocess(priorities, 1 / self._clip, self._clip)

    @tf.function(experimental_relax_shapes=True)
    def _calculate_IS1_priorities(self, density):
        return self._priorities_postprocess(density, 1 / self._clip, 1)

    @tf.function(experimental_relax_shapes=True)
    def _calculate_density(self, policies, old_policies, mask):
        policies_masked = policies * mask + (1 - mask) * tf.ones_like(policies)
        old_policies_masked = old_policies * mask + (1 - mask) * tf.ones_like(old_policies)

        policies_ratio = policies_masked / old_policies_masked
        policies_ratio_prod = tf.math.cumprod(policies_ratio, axis=-1, keepdims=True)

        return policies_ratio_prod

    @tf.function(experimental_relax_shapes=True)
    def _calculate_TD_priorities(self, td):
        return self._priorities_postprocess(tf.abs(tf.reduce_mean(td, -1)), 1 / self._clip, self._clip)

    @tf.function(experimental_relax_shapes=True)
    def _calculate_weighted_TD_priorities(self, td, weights):
        return self._priorities_postprocess(tf.abs(tf.reduce_mean(td * weights, -1)), 1 / self._clip, self._clip)

    @tf.function(experimental_relax_shapes=True)
    def _calculate_oTD_priorities(self, td):
        return self._priorities_postprocess(tf.maximum(tf.reduce_mean(td, -1) + self._beta, 0), self._beta, self._clip)
