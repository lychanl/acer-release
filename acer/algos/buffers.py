import numpy as np
from replay_buffer import MultiReplayBuffer, VecReplayBuffer

import tensorflow as tf


class AdaptiveSizeBuffer(VecReplayBuffer):
    def get_vec(self, length: int, trajectory_len: int, size_limit: int):
        if self._current_size == 0:
            return self._zero_vec(length)

        sample_indices = self._sample_random_index(length, size_limit=size_limit)

        return self._get_sampled_vec(length, trajectory_len, sample_indices)

    def _sample_random_index(self, size=None, size_limit=np.inf) -> int:
        """Uniformly samples one index from the buffer"""
        limit = min(size_limit, self._current_size)
        return np.random.randint(low=self._pointer - limit, high=self._pointer, size=size) % self._max_size

class ISAdaptiveSizeBuffer(MultiReplayBuffer):
    def __init__(self, *args, is_decay=0.999, initial_limit=1000000, target_is_variance=1, update_speed=5, **kwargs):
        super().__init__(*args, **kwargs, buffer_class=AdaptiveSizeBuffer)
        self.target_is_variance = target_is_variance
        self.size_limit = tf.Variable(float(initial_limit))

        self.is_mean = tf.Variable(0.)
        self.is_decay = is_decay
        self.update_speed = update_speed

        self.register_method("update_is_mean", self.update_is_mean, {"weights": "actor.density"})
        self.register_method(
            "is_variance", self.is_variance, {
                "weights": "actor.density",
                "is_mean": "self.update_is_mean"
        })
        self.register_method(
            "update_size_limit", self.update_size_limit, {
                "is_variance": "self.is_variance"
        })

    def get(self, *args, **kwargs):
        assert False, "get() shouldn't be used except by legacy algos"

    def get_vec(self, length_per_buffer: int, trajectory_len: int):
        vecs = [buffer.get_vec(length_per_buffer, trajectory_len, int(self.size_limit.numpy())) for buffer in self._buffers]

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

    @tf.function
    def update_is_mean(self, weights):
        return self.is_mean.assign(
            self.is_mean * self.is_decay
            + tf.reduce_mean(tf.math.log(tf.maximum(weights[:, 0], 1e-8))) * (1 - self.is_decay)
        )

    @tf.function
    def is_variance(self, weights, is_mean):
        return (tf.math.log(tf.maximum(weights[:, 0], 1e-8)) - is_mean) ** 2

    @tf.function
    def update_size_limit(self, is_variance):
        return self.size_limit.assign_add(tf.sign(self.target_is_variance - is_variance) * self.update_speeds)
