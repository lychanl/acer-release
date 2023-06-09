import numpy as np
from algos.base import GaussianActor
import tensorflow as tf
import tensorflow_probability as tfp


class VarSigmaActor(GaussianActor):
    @staticmethod
    def get_args():
        args = GaussianActor.get_args()
        args['entropy_bonus'] = (float, 0)
        args['single_std'] = (bool, False, {'action': 'store_true'})
        args['nn_std'] = (bool, False, {'action': 'store_true'})
        args['separate_nn_std'] = (int, None, {'nargs': '*'})
        args['std_lr'] = (float, None)
        args['initial_log_std'] = (float, 0)
        args['clip_log_std'] = (float, None, {'nargs': 2})
        return args

    def __init__(
            self, obs_space, action_space,
            *args, entropy_bonus=0, dist_std_gradient=True,
            single_std=False, nn_std=False, separate_nn_std=None,
            std_lr=None, initial_log_std=0, std_loss_args=None, custom_optimization=False,
            clip_log_std=None,
            **kwargs):

        self.entropy_bonus = entropy_bonus
        self.dist_std_gradient = dist_std_gradient
        self.separate_nn_std = separate_nn_std
        self.single_std = single_std
        self.nn_std = nn_std
        self.std_lr = std_lr
        self.initial_log_std = initial_log_std
        self.clip_log_std = clip_log_std

        if std_lr:
            assert separate_nn_std is not None or not nn_std

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
                self.var_log_std = tf.Variable(initial_log_std, dtype=tf.float32, name='log_std')
            else:
                self.var_log_std = tf.Variable(np.zeros(action_space.shape) + initial_log_std, dtype=tf.float32, name='log_std')

        GaussianActor.__init__(self, obs_space, action_space, *args, **kwargs, additional_outputs=additional_outputs, extra_models=extra_models)

        self.register_method('std', self.std, {'observations': 'obs'})

        if not custom_optimization:
            if std_loss_args is None:
                std_loss_args = {
                    'observations': 'base.first_obs',
                    'actions': 'base.first_actions',
                    'd': 'base.weighted_td'
                }

            if self.std_lr:
                self.register_method('optimize_std', self.optimize_std, std_loss_args)
                self.register_method('optimize', self.optimize_mean, {
                    'observations': 'base.first_obs',
                    'actions': 'base.first_actions',
                    'd': 'base.weighted_td'
                })
                self.targets.append('optimize_std')
            else:
                for k, v in std_loss_args.items():
                    self.methods['optimize'][1][k] = v

    @property
    def mean_trainable_variables(self):
        return self._hidden_layers.trainable_variables

    @property
    def std_trainable_variables(self):
        if self.separate_nn_std:
            return self._extra_hidden_layers.trainable_variables
        else:
            return [self.var_log_std]

    @tf.function(experimental_relax_shapes=True)
    def mean_and_log_std(self, observations):
        out = self._forward(observations)
        if self.separate_nn_std:
            mean = out
            log_std, = self._extras_forward(observations)
            log_std += self.initial_log_std
        elif self.nn_std:
            mean = out[..., :self._k]
            log_std = out[..., self._k:] + self.initial_log_std
        else:
            mean = out
            log_std = tf.ones_like(mean[...,:1]) * tf.expand_dims(self.var_log_std, 0)
        if self.single_std:
            log_std = tf.repeat(log_std, self._k, axis=-1)
        
        if self._clip_mean is not None:
            mean = tf.clip_by_value(mean, -self._clip_mean, self._clip_mean)
        if self.clip_log_std is not None:
            log_std = tf.clip_by_value(log_std, self.clip_log_std[0], self.clip_log_std[1])

        return mean, log_std

    @tf.function(experimental_relax_shapes=True)
    def mean_and_std(self, observations):
        mean, log_std = self.mean_and_log_std(observations)
        std = tf.exp(log_std)
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

        dist = self.distribution(
            loc=mean,
            scale_diag=std
        )

        return self._loss(dist, actions, d)

    def std_loss(self, observations: np.array, actions: np.array, d: np.array, **kwargs):
        mean, std = self.mean_and_std(observations)
        mean = tf.stop_gradient(mean)

        dist = self.distribution(
            loc=mean,
            scale_diag=std
        )

        entropy_bonus = self.entropy_bonus * tf.reduce_mean(dist.entropy())

        return self._loss(dist, actions, d) - entropy_bonus
        
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