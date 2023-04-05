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
    @staticmethod
    def get_args():
        args = MultiReplayBuffer.get_args()
        args['is_ref_decay'] = (float, 0.999)
        args['initial_limit'] = (float, 100000)
        args['min_size_limit'] = (int, 1000)
        args['ref_type'] = (str, 'mean')
        args['target_is_dispersion'] = (float, 10)
        args['update_speed'] = (float, 5)
        args['update_type'] = (str, 'lin', {'choices': ('lin', 'exp')})
        return args

    def __init__(
            self, *args, is_ref_decay=0.999, initial_limit=100000,
            min_size_limit=1000, ref_type='mean', target_is_dispersion=10,
            update_speed=5, update_type='lin', **kwargs):
        super().__init__(*args, **kwargs, buffer_class=AdaptiveSizeBuffer)
        self.target_is_dispersion = target_is_dispersion
        self.size_limit = tf.Variable(float(initial_limit))
        self.ref_type = ref_type

        self.is_ref = tf.Variable(0.)
        self.is_ref_decay = is_ref_decay
        self.update_speed = update_speed
        self.update_type = update_type
        self.min_size_limit = min_size_limit

        self.register_method("log_weights", self.log_weights, {"weights": "actor.density"})
        self.register_method("update_is_ref", self.update_is_ref, {"log_weights": "self.log_weights"})
        self.register_method(
            "is_dispersion", self.is_dispersion, {
                "log_weights": "self.log_weights",
                "is_ref": "self.update_is_ref"
        })
        self.register_method(
            "update_size_limit", self.update_size_limit, {
                "is_dispersion": "self.is_dispersion"
        })

        self.targets = ['update_is_ref', 'update_size_limit']

    # overrides
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

    # is-based adaptation
    @tf.function
    def log_weights(self, weights):
        return tf.math.log(tf.maximum(weights[:, 0], 1e-8))

    @tf.function
    def update_is_ref(self, log_weights):
        if self.ref_type == 'mean':
            return self.is_ref.assign(
                self.is_ref * self.is_ref_decay
                + tf.reduce_mean(log_weights) * (1 - self.is_ref_decay)
            )
        else:
            return 0

    @tf.function
    def is_dispersion(self, log_weights, is_ref):
        return (log_weights - is_ref) ** 2

    @tf.function
    def update_size_limit(self, is_dispersion):
        change = tf.reduce_mean(tf.sign(self.target_is_dispersion - is_dispersion)) * self.update_speed
        if self.update_type == 'lin':
            new_limit = self.size_limit + change
        else:
            new_limit = tf.exp(tf.math.log(self.size_limit) + change)
        return self.size_limit.assign(tf.clip_by_value(new_limit, self.min_size_limit, self._max_size))
