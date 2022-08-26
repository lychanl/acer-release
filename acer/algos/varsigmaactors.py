import numpy as np
from algos.base import GaussianActor
import tensorflow as tf
import tensorflow_probability as tfp


class VarSigmaActor(GaussianActor):
    def __init__(
            self, obs_space, action_space,
            *args, entropy_bonus=0, dist_std_gradient=True,
            single_std=False, nn_std=False, separate_nn_std=None,
            std_lr=None, initial_log_std=0, **kwargs):
        self.entropy_bonus = entropy_bonus
        self.dist_std_gradient = dist_std_gradient
        self.separate_nn_std = separate_nn_std
        self.single_std = single_std
        self.nn_std = nn_std
        self.std_lr = std_lr
        self.initial_log_std = initial_log_std

        if std_lr:
            assert separate_nn_std is not None

        if separate_nn_std is not None:
            additional_outputs = 0
            if single_std:
                extra_models = ([*separate_nn_std, 1],)
            else:
                extra_models = ([*separate_nn_std, action_space.shape[0]],)
        elif nn_std:
            extra_models = ()
            if single_std:
                additional_outputs = 1
            else:
                additional_outputs = action_space.shape[0]
        else:
            additional_outputs = 0
            extra_models = ()
            if single_std:
                self.log_std = tf.Variable(inital_log_std, dtype=tf.float32)
            else:
                self.log_std = tf.Variable(np.zeros(action_space.shape) + inital_log_std, dtype=tf.float32)

        GaussianActor.__init__(self, obs_space, action_space, *args, **kwargs, additional_outputs=additional_outputs, extra_models=extra_models)

        self.register_method('std', self.std, {'observations': 'obs'})
        if self.std_lr:
            self.register_method('optimize_std', self.optimize_std, {
                'observations': 'base.first_obs',
                'actions': 'base.first_actions',
                'd': 'base.weighted_td'
            })
            self.register_method('optimize', self.optimize_mean, {
                'observations': 'base.first_obs',
                'actions': 'base.first_actions',
                'd': 'base.weighted_td'
            })
            self.targets.append('optimize_std')

    @property
    def mean_trainable_variables(self):
        return self._hidden_layers.trainable_variables

    @property
    def std_trainable_variables(self):
        return self._extra_hidden_layers.trainable_variables

    @tf.function(experimental_relax_shapes=True)
    def mean_and_std(self, observations):
        out = self._forward(observations)
        if self.separate_nn_std:
            mean = out
            log_std, = self._extras_forward(observations)
            log_std +=  + self.initial_log_std
        elif self.nn_std:
            mean = out[..., :self._k]
            log_std = out[..., self._k:] + self.initial_log_std
        else:
            mean = out
            log_std = self.log_std
        std = tf.exp(log_std)
        if self.single_std:
            std = tf.repeat(std, self._k, axis=-1)
        return mean, std if self.dist_std_gradient else tf.stop_gradient(std)

    @tf.function(experimental_relax_shapes=True)
    def std(self, observations):
        return self.mean_and_std(observations)[1]

    def loss(self, **kwargs) -> tf.Tensor:
        loss = self.mean_loss(**kwargs)
        if self.std_lr is None:
            # no separate optimizer for std
            loss += self.std_loss(**kwargs)
        return loss

    def mean_loss(self, observations: np.array, actions: np.array, d: np.array, **kwargs) -> tf.Tensor:
        mean, std = self.mean_and_std(observations)
        std = tf.stop_gradient(std)

        dist = tfp.distributions.MultivariateNormalDiag(
            loc=mean,
            scale_diag=std
        )

        return self._loss(mean, dist, actions, d)

    def std_loss(self, observations: np.array, actions: np.array, d: np.array, **kwargs):
        mean, std = self.mean_and_std(observations)
        mean = tf.stop_gradient(mean)

        dist = tfp.distributions.MultivariateNormalDiag(
            loc=mean,
            scale_diag=std
        )

        entropy_bonus = self.entropy_bonus * tf.reduce_mean(dist.entropy())

        return self._loss(mean, dist, actions, d) - entropy_bonus
        
    def init_optimizer(self, lr, *args, **kwargs):
        self.optimizer = tf.keras.optimizers.Adam(lr=lr, *args, **kwargs)
        if self.std_lr is not None:
            self.std_optimizer = tf.keras.optimizers.Adam(lr=self.std_lr, *args, **kwargs)
        return self.optimizer

    def optimize_mean(self, **loss_kwargs):
        with tf.GradientTape() as tape:
            loss = self.loss(**loss_kwargs)
        grads = tape.gradient(loss, self.mean_trainable_variables)
        gradients = zip(grads, self.mean_trainable_variables)

        self.optimizer.apply_gradients(gradients)

        return loss

    def optimize_std(self, **loss_kwargs):
        with tf.GradientTape() as tape:
            loss = self.std_loss(**loss_kwargs)
        grads = tape.gradient(loss, self.std_trainable_variables)
        gradients = zip(grads, self.std_trainable_variables)

        self.std_optimizer.apply_gradients(gradients)

        return loss