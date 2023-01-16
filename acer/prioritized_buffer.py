import functools
from replay_buffer import BufferFieldSpec, VecReplayBuffer, MultiReplayBuffer

import numpy as np
import tensorflow as tf
from typing import Union, Dict


class PrioritizedReplayBuffer(VecReplayBuffer):
    def __init__(
            self, max_size: int, action_spec: BufferFieldSpec, obs_spec: BufferFieldSpec, policy_spec: BufferFieldSpec,
            block: int = 1024, levels: int = 1, probability_as_actor_out: bool = False, *args, **kwargs):
        super().__init__(max_size, action_spec, obs_spec, policy_spec=policy_spec, *args, **kwargs)
        self._udpate_pointer = 0

        self._block = block
        self._levels = levels
        self._prob_as_actor_out = probability_as_actor_out

        self._rands = None

        self._priorities = []
        self._priorities_cumsums = []
        self._block_starts = []
        self._block_ends = []
        self._total_priorities = 0
        self._total_abs_reward = 0
        self._total_reward = 0
        self._total_2_reward = 0

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
        if self._prob_as_actor_out:
            self._priorities[0][self._pointer] = policy[0]    
            policy = policy[1:]
        else:
            self._priorities[0][self._pointer] = 1

        self._update_block_priorities(self._pointer // self._block)

        if self._current_size == self._max_size:
            self._total_abs_reward -= abs(self._rewards[self._pointer])
            self._total_reward -= self._rewards[self._pointer]
            self._total_2_reward -= self._rewards[self._pointer] ** 2
        self._total_abs_reward += abs(reward)
        self._total_reward += reward
        self._total_2_reward += reward ** 2

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
            priority: str = 'IS', block: int = 256, clip: float = -1, alpha: float = 1, beta: float = 0, eps=1e-4, *args, updatable=True, **kwargs):
        self._clip = clip
        self._alpha = alpha
        self._beta = beta
        self._eps = eps
        self.block = block * num_buffers
        self.updatable = updatable

        self._mean_abs_reward = tf.Variable(0, dtype=tf.float32)
        self._std_reward = tf.Variable(0, dtype=tf.float32)

        PRIORITIES = {
            "IS": (self._calculate_IS_priorities, {'density': 'actor.density'}),
            "IS1": (self._calculate_IS1_priorities, {'density': 'actor.density'}),
            "prob": (self._calculate_prob_priorities, {'policies': 'actor.policies', 'mask': 'mask'}),
            "TD": (self._calculate_TD_priorities, {'td': 'base.td',}),
            "oTD": (self._calculate_oTD_priorities, {'td': 'base.td',}),
            # "piTD": (self._calculate_piTD_priorities, {'td': 'base.td', 'policies': 'actor.policies', 'e_probs': 'actor.expected_probs'}),
            # "soft_piTD": (self._calculate_soft_piTD_priorities, {'td': 'base.td', 'policies': 'actor.policies', 'e_probs': 'actor.expected_probs'}),
            # "softsign_piTD": (self._calculate_softsign_piTD_priorities, {'td': 'base.td', 'policies': 'actor.policies', 'e_probs': 'actor.expected_probs'}),
            # "nd_piTD": (self._calculate_nd_piTD_priorities, {'td': 'base.td', 'policies': 'actor.policies', 'e_probs': 'actor.expected_probs'}),
            # "soft_nd_piTD": (self._calculate_soft_nd_piTD_priorities, {'td': 'base.td', 'policies': 'actor.policies', 'e_probs': 'actor.expected_probs'}),
            "weightedTD": (self._calculate_weighted_TD_priorities, {'td': 'base.td', 'weights': 'actor.density'}),
        }

        TDS = {
            "": 'base.td',
            "trIS": 'base.density_weighted_td',
        }
    
        for pre in ("", "nd", "first", "last"):
            for post in ("", "soft", "softsign", "sign"):
                for scale in ("", "abs", "std"):
                    for td in ("", "trIS"):
                        name = "_".join(filter(lambda x: x, [post, pre, td, "piTD", scale]))

    
                        PRIORITIES[name] = (
                            functools.partial(self._calculate_piTD_priorities, pre=pre, post=post, scale=scale),
                            {
                                'td': TDS[td],
                                'policies': 'actor.policies', 'e_probs': 'actor.expected_probs',
                                'mean_abs_rewards': 'memory.mean_abs_rewards', 'std_rewards': 'memory.std_rewards'
                            }
                        )
        
        assert priority in PRIORITIES, f"priority {priority} not in {', '.join(PRIORITIES.keys())}"

        priority = PRIORITIES[priority]

        super().__init__(
            max_size, num_buffers, action_spec, obs_spec,
            buffer_class=PrioritizedReplayBuffer, policy_spec=policy_spec,
            priority_spec=priority, block=block, *args, **kwargs)

        self.register_method('mean_abs_rewards', self.mean_abs_reward, {})
        self.register_method('std_rewards', self.mean_abs_reward, {})

    def put(self, *args, **kwargs):
        super().put(*args, **kwargs)
        self._mean_abs_reward.assign(sum(b._total_abs_reward / b._current_size for b in self._buffers) / len(self._buffers))
        self._std_reward.assign(
            (sum(b._total_2_reward for b in self._buffers) - sum(b._total_reward for b in self._buffers) ** 2)
            / sum(b._current_size for b in self._buffers)
        )

    @tf.function
    def mean_abs_reward(self):
        return self._mean_abs_reward

    @tf.function
    def std_reward(self):
        return self._std_reward

    def should_update_block(self):
        return self.updatable and all(buf.should_update_block() for buf in self._buffers)

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

    @tf.function(experimental_relax_shapes=True)
    def _calculate_piTD_priorities(self, td, policies, e_probs, mean_abs_rewards, std_rewards, pre="", post="", scale=""):

        policies_norm = (tf.math.cumprod(policies, -1) - e_probs)
        if pre == "nd":
            policies_norm = policies_norm / e_probs

        elif pre == "first":
            policies_norm = policies_norm[:,0]
            td = td[:,:1]

        elif pre == "last":
            policies_norm = policies_norm[:,-1]
            td = td[:,-1:]

        if scale == "abs":
            td = td / tf.maximum(mean_abs_rewards, self._eps)

        elif scale == "std":
            td = td / tf.maximum(std_rewards, self._eps)

        base = -policies_norm * td

        if post == "":
            processed = tf.maximum(base, 0) + 1

        elif post == "soft":
            processed = tf.nn.elu(base) + 1

        elif post == "softsign":
            processed = tf.nn.softsign(base) + 1

        elif post == "sign":
            processed = tf.sign(base) + 1

        return self._priorities_postprocess(tf.reduce_mean(processed, -1) * (1 - self._beta) + self._beta, 0, self._clip)

    @tf.function(experimental_relax_shapes=True)
    def _calculate_soft_piTD_priorities(self, td, policies, e_probs):

        policies_norm = (tf.math.cumprod(policies, -1) - e_probs) / e_probs

        base = -policies_norm * td

        return self._priorities_postprocess(self._beta + (1 - self._beta) * (1 + tf.nn.elu(tf.reduce_mean(base, -1))), 0, self._clip)

    @tf.function(experimental_relax_shapes=True)
    def _calculate_softsign_piTD_priorities(self, td, policies, e_probs):

        policies_norm = (tf.math.cumprod(policies, -1) - e_probs) / e_probs

        base = -policies_norm * td

        return self._priorities_postprocess(self._beta + (1 - self._beta) * (1 + tf.nn.softsign(tf.reduce_mean(base, -1))), 0, self._clip)

    @tf.function(experimental_relax_shapes=True)
    def _calculate_nd_piTD_priorities(self, td, policies, e_probs):

        policies_norm = (tf.math.cumprod(policies, -1) - e_probs)

        base = -policies_norm * td

        return self._priorities_postprocess(tf.maximum(tf.reduce_mean(base, -1) + self._beta, 0), self._beta, self._clip)

    @tf.function(experimental_relax_shapes=True)
    def _calculate_soft_nd_piTD_priorities(self, td, policies, e_probs):

        policies_norm = (tf.math.cumprod(policies, -1) - e_probs)

        base = -policies_norm * td

        return self._priorities_postprocess(self._beta + (1 - self._beta) * (1 + tf.nn.elu(tf.reduce_mean(base, -1))), 0, self._clip)
