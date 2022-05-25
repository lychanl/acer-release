
from algos.fast_acer import FastACER

import tensorflow as tf


class StdClippedACER(FastACER):
    def __init__(self, *args, alpha=1, eps=0, scale_td=False, **kwargs) -> None:
        super(FastACER, self).__init__(*args, **kwargs)
        self._alpha = alpha
        self._eps = eps
        self._scale_td = scale_td

    def _init_automodel(self):
        super()._init_automodel()
        self.register_method('density_weighted_td', self._calculate_weighted_td, {
            "td": "base.td",
            "weights": "actor.truncated_density",
            "std": "critic.std"
        })

    def _calculate_weighted_td(self, td, weights, std):
        td = super()._calculate_weighted_td(td, weights)
        std = tf.maximum(std, self._eps)
        clipped = tf.clip_by_value(td, -self._alpha * std, self._alpha * std)
        if self._scale_td:
            clipped = clipped / std

        return tf.stop_gradient(weights * clipped)
