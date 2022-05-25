
from algos.fast_acer import FastACER

import tensorflow as tf


class StdClippedACER(FastACER):
    def __init__(self, *args, alpha=1, eps=0., scale_td=False, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._alpha = alpha
        self._eps = eps
        self._scale_td = scale_td

    def _init_automodel_overrides(self):
        self.register_method('weighted_td', self._calculate_weighted_td, {
            "td": "base.td",
            "weights": "actor.sample_weights",
            "std": "critic.std"
        })

    @tf.function
    def _calculate_weighted_td(self, td, weights, std):
        std = tf.maximum(std, self._eps)[:,:,0]
        clipped = tf.clip_by_value(td, -self._alpha * std, self._alpha * std)
        if self._scale_td:
            clipped = clipped / std

        return tf.stop_gradient(weights * clipped)
