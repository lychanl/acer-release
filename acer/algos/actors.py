from algos.base import GaussianActor, CategoricalActor

import tensorflow as tf


class StdClippedBaseActor:
    def __init__(self, *args, alpha=1, eps=0, scale_td=False, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self._alpha = alpha
        self._eps = eps
        self._scale_td = scale_td

        self.register_method('optimize', self.optimize, {
            'observations': 'base.first_obs',
            'actions': 'base.first_actions',
            'd': 'self.weighted_clipped_td'
        })

        self.register_method('weighted_clipped_td', self._calculate_weighted_clipped_td, {
            "td": "base.td",
            "weights": "actor.sample_weights",
            "std": "critic.std"
        })

    def _calculate_weighted_clipped_td(self, td, weights, std):
        std = tf.maximum(std, self._eps)
        clipped = tf.clip_by_value(td, -self._alpha * std, self._alpha * std)
        if self._scale_td:
            clipped = clipped / std

        return tf.stop_gradient(weights * clipped)


class StdClippedCategoricalActor(CategoricalActor, StdClippedBaseActor):
    pass


class StdClippedGaussianActor(GaussianActor, StdClippedBaseActor):
    pass

