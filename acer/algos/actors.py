import functools
from algos.base import GaussianActor, CategoricalActor

import tensorflow as tf


class StdClippedBaseActor:
    def __init__(self, *args, alpha=1, eps=0, scale_td=False, clip_weighted=False, **kwargs) -> None:
        self._alpha = alpha
        self._eps = eps
        self._scale_td = scale_td
        self._clip_weighted = clip_weighted

        self.register_method('optimize', self.optimize, {
            'observations': 'base.first_obs',
            'actions': 'base.first_actions',
            'd': 'self.weighted_clipped_td'
        })

        self.register_method('weighted_clipped_td', 
            self._calculate_clipped_weighted_td if clip_weighted else self._calculate_weighted_clipped_td, {
            "td": "base.td",
            "weights": "actor.sample_weights",
            "std": "critic.std"
        })

    @tf.function
    def _calculate_weighted_clipped_td(self, td, weights, std):
        std = tf.maximum(std, self._eps)[:,:,0]
        clipped = tf.clip_by_value(td, -self._alpha * std, self._alpha * std)
        if self._scale_td:
            clipped = clipped / std

        return tf.stop_gradient(weights * clipped)

    @tf.function
    def _calculate_clipped_weighted_td(self, td, weights, std):
        std = tf.maximum(std, self._eps)[:,:,0]
        weighted = weights * td
        clipped = tf.clip_by_value(weighted, -self._alpha * std, self._alpha * std)
        if self._scale_td:
            clipped = clipped / std

        return tf.stop_gradient(weighted)


class StdClippedCategoricalActor(CategoricalActor, StdClippedBaseActor):
    def __init__(self, *args, **kwargs) -> None:
        CategoricalActor.__init__(self, *args, **kwargs)
        StdClippedBaseActor.__init__(self, *args, **kwargs)


class StdClippedGaussianActor(GaussianActor, StdClippedBaseActor):
    def __init__(self, *args, **kwargs) -> None:
        GaussianActor.__init__(self, *args, **kwargs)
        StdClippedBaseActor.__init__(self, *args, **kwargs)


class TD2RegularizedActor:
    def __init__(self, *args, eta=0.1, kappa=1, **kwargs) -> None:
        self._eta = tf.Variable(eta)
        self._kappa = kappa

        self.register_method('optimize', self.optimize, {
            'observations': 'base.first_obs',
            'actions': 'base.first_actions',
            'd': 'self.regularized_d'
        })

        self.register_method('regularized_d', self._calculate_regularized_d, {
            'd': 'base.weighted_td',
            'td': 'base.td',
            'weights': 'self.sample_weights',
            'eta': 'self.eta'
        })

        self.register_method('eta', self.eta, {})

        self.register_method('eta_decay', self._eta_decay, {
            'eta': 'self.eta'
        })

        self.targets.append('eta_decay')

    def _calculate_regularized_d(self, d, td, weights, eta):
        g = td ** 2 * weights
        return d - eta * g

    def _eta_decay(self, eta):
        self._eta.assign(eta * self._kappa)

    def eta(self):
        return self._eta


class TD2RegularizedCategoricalActor(CategoricalActor, TD2RegularizedActor):
    def __init__(self, *args, **kwargs) -> None:
        CategoricalActor.__init__(self, *args, **kwargs)
        TD2RegularizedActor.__init__(self, *args, **kwargs)


class TD2RegularizedGaussianActor(GaussianActor, TD2RegularizedActor):
    def __init__(self, *args, **kwargs) -> None:
        GaussianActor.__init__(self, *args, **kwargs)
        TD2RegularizedActor.__init__(self, *args, **kwargs)
