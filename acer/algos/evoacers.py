import numpy as np
from algos.varsigmaactors import VarSigmaActor

import tensorflow as tf
import tensorflow_probability as tfp


class R15Actor(VarSigmaActor):
    # Actor implementing 1/5th rule
    def __init__(self, *args, ratio=0.2, coeff=0.9, schema='simple', mask_outliers=False, **kwargs):
        assert schema in ('simple', 'stoch')
        self.schema = schema
        self.target_ratio = ratio
        self.coeff = coeff
        self.mask_outliers = mask_outliers

        self.reverse_stoch_coeff = 1 / ratio * (1 - coeff * (1 - ratio))
        std_loss_kwargs = {
            'observations': 'base.first_obs',
            'successes': 'actor.successes',
            'weights': 'actor.sample_weights',
            'mask': 'base.mask'
        }
        if mask_outliers:
            std_loss_kwargs['mask_outliers'] = 'critic.outliers_mask'
        self.mask_outliers = mask_outliers
        VarSigmaActor.__init__(self, *args, **kwargs, std_loss_args=std_loss_kwargs)

        self.register_method('successes', self.successes, {'td': 'base.td'})
        self.register_method('success_ratio', self.success_ratio, {'successes': 'self.successes', 'weights': 'actor.sample_weights', 'mask': 'base.mask'})

    @tf.function
    def successes(self, td):
        return tf.cast(td > 0, tf.float32)

    @tf.function
    def success_ratio(self, successes, weights, mask):
        return tf.reduce_sum(successes * weights * mask) / tf.reduce_sum(weights * mask)

    @tf.function
    def std_loss(self, observations, successes, weights, mask, mask_outliers=None, **kwargs):
        mean, log_std = self.mean_and_log_std(observations)
        if self.schema == 'simple':
            # if success:
            #   std = std / coeff ** (1 - target_ratio)
            # else:
            #   std = std * coeff ** target_ratio
            succ_grad = self.coeff ** (1 - self.target_ratio)
            fail_grad = 1 / self.coeff ** self.target_ratio
        else:
            # if success:
            #   std = std * self.reverse_stoch_coeff
            # else:
            #   std = std * coeff
            succ_grad = self.reverse_stoch_coeff
            fail_grad = self.coeff

        log_grads = tf.math.log(successes * succ_grad + (1 - successes) * fail_grad)

        if self.mask_outliers:
            mask = mask * (1 - mask_outliers)

        return -tf.reduce_mean(
            log_std * tf.stop_gradient(tf.reduce_sum(log_grads * weights * mask, axis=1, keepdims=True) / tf.reduce_sum(mask))
        )


class MedianRuleActor(VarSigmaActor):
    def __init__(self, *args, q=25, **kwargs):
        self.q = q

        VarSigmaActor.__init__(self, *args, **kwargs, std_loss_args={
            'observations': 'base.first_obs',
            'successes': 'actor.successes',
            'weights': 'actor.sample_weights',
            'mask': 'base.mask'
        })

        self.register_method('successes', self.successes, {'quantile_td': 'critic.quantile_td'})
        self.register_method('success_ratio', self.success_ratio, {'successes': 'self.successes', 'weights': 'actor.sample_weights', 'mask': 'base.mask'})

    @tf.function
    def successes(self, quantile_td):
        return tf.cast(quantile_td[...,self.q] > 0, tf.float32)

    @tf.function
    def success_ratio(self, successes, weights, mask):
        return tf.reduce_sum(successes * weights * mask) / tf.reduce_sum(weights * mask)

    @tf.function
    def std_loss(self, observations, successes, weights, mask, **kwargs):
        mean, log_std = self.mean_and_log_std(observations)

        log_grads = 1 - 2 * successes

        return -tf.reduce_mean(log_std * tf.reduce_sum(log_grads * weights * mask, axis=1, keepdims=True) / tf.reduce_sum(mask))


class MedianToValueActor(VarSigmaActor):
    def __init__(self, *args, **kwargs):
        VarSigmaActor.__init__(self, *args, **kwargs, std_loss_args={
            'observations': 'base.first_obs',
            'successes': 'actor.successes',
            'weights': 'actor.sample_weights',
            'mask': 'base.mask'
        })

        self.register_method('successes', self.successes, {'value': 'critic.value', 'median': 'critic.median'})
        self.register_method('success_ratio', self.success_ratio, {'successes': 'self.successes', 'weights': 'actor.sample_weights', 'mask': 'base.mask'})

    @tf.function
    def successes(self, value, median):
        return tf.cast(median < value[..., 0], tf.float32)

    @tf.function
    def success_ratio(self, successes, weights, mask):
        return tf.reduce_sum(successes * weights * mask) / tf.reduce_sum(weights * mask)

    @tf.function
    def std_loss(self, observations, successes, weights, mask, **kwargs):
        mean, log_std = self.mean_and_log_std(observations)
        # median > value -> increase
        # median < value -> decrease
        log_grads = 2 * successes - 1

        return -tf.reduce_mean(log_std * tf.reduce_sum(log_grads * weights * mask, axis=1, keepdims=True) / tf.reduce_sum(mask))


class DelayedMedianRuleActor(VarSigmaActor):
    def __init__(self, obs_space, *args, q=25, delay=100, each_step_delay=False, batch_size, **kwargs):
        assert 'std_lr' in kwargs, "DelayedMedianRuleActor requires separate optimization process for std"
        self.q = q
        self.delay = delay
        self.each_step_delay = each_step_delay

        if each_step_delay:
            self._stored = tf.Variable(0, trainable=False)
            self._stored_obs = tf.Variable(np.zeros((delay, batch_size, *obs_space.shape)), dtype=obs_space.dtype, trainable=False)
            self._stored_quantiles = tf.Variable(np.zeros((delay, batch_size)), dtype=tf.float32, trainable=False)
        else:
            self._stored = tf.Variable(False, trainable=False)
            self._stored_obs = tf.Variable(np.zeros((batch_size, *obs_space.shape)), dtype=obs_space.dtype, trainable=False)
            self._stored_quantiles = tf.Variable(np.zeros(batch_size), dtype=tf.float32, trainable=False)

        VarSigmaActor.__init__(self, obs_space, *args, **kwargs, std_loss_args={
            'time_step': 'base.time_step',
            'stored': 'self.stored',
            'stored_obs': 'self.stored_obs',
            'successes': 'actor.successes',
        })

        self.register_parameterized_method_call('quantiles_old', 'critic.quantiles', {'observations': 'self.stored_obs'})
        self.register_method('stored', self.stored, {})
        self.register_method(
            'stored_obs', self.stored_obs_each_step if each_step_delay else self.stored_obs,
            {'time_step': 'base.time_step'}
        )
        self.register_method(
            'stored_quantiles', self.stored_quantiles_each_step if each_step_delay else self.stored_quantiles,
            {'time_step': 'base.time_step'}
        )
        self.register_method(
            'store_values', self.store_values_each_step if each_step_delay else self.store_values, {
                'stored_obs': 'self.stored_obs', 'obs': 'base.first_obs',
                'stored_quantiles': 'self.stored_quantiles', 'quantiles': 'critic.quantiles',
                'stored': 'self.stored', 'time_step': 'base.time_step'
        })

        self.register_method('successes', self.successes, {
            'quantiles_stored': 'self.stored_quantiles', 'quantiles_old': 'self.quantiles_old'
        })

        self.targets.append('store_values')

    def stored(self):
        return tf.identity(self._stored)

    def stored_obs(self, time_step):
        return tf.identity(self._stored_obs)

    def stored_obs_each_step(self, time_step):
        return tf.identity(self._stored_obs[time_step % self.delay])

    def stored_quantiles(self, time_step):
        return tf.identity(self._stored_quantiles)

    def stored_quantiles_each_step(self, time_step):
        return tf.identity(self._stored_quantiles[time_step % self.delay])

    def store_values_each_step(self, stored_obs, obs, stored_quantiles, quantiles, stored, time_step):
        i = time_step % self.delay
        self._stored_obs[i].assign(stored_obs * 0 + obs)
        self._stored_quantiles[i].assign(stored_quantiles * 0 + quantiles[:,0,self.q])
        self._stored.assign(tf.minimum(self.delay, stored + 1))

    @tf.function
    def store_values(self, stored_obs, obs, stored_quantiles, quantiles, stored, time_step):
        if time_step % self.delay == 0 or not stored:
            self._stored_obs.assign(stored_obs * 0 + obs)
            self._stored_quantiles.assign(stored_quantiles * 0 + quantiles[:,0,self.q])
            self._stored.assign(stored or True)

    @tf.function
    def successes(self, quantiles_stored, quantiles_old):
        return tf.cast(quantiles_old > tf.expand_dims(quantiles_stored, 1), tf.float32)

    @tf.function
    def std_loss(self, stored_obs, successes, **kwargs):
        mean, log_std = self.mean_and_log_std(stored_obs)
        # median > value -> increase
        # median < value -> decrease
        log_grads = 2 * tf.reduce_mean(successes, axis=1, keepdims=True) - 1

        return -tf.reduce_mean(log_std * log_grads)

    @tf.function
    def optimize_std(self, time_step, stored, **loss_kwargs):
        if self.each_step_delay:
            optimize = stored >= self.delay
        else:
            optimize = time_step % self.delay == 0 and stored
        if optimize:
            with tf.GradientTape() as tape:
                loss = self.std_loss(**loss_kwargs)
            grads = tape.gradient(loss, self.std_trainable_variables)
            gradients = zip(grads, self.std_trainable_variables)

            self.std_optimizer.apply_gradients(gradients)

            return loss
        else:
            return 0.


class LeewayDelayedMedianRuleActor(DelayedMedianRuleActor):
    def __init__(self, *args, leeway=None, outliers=None, batch_size, delay=100, each_step_delay=False, **kwargs):
        super().__init__(*args, **kwargs, batch_size=batch_size, each_step_delay=each_step_delay)
        self.leeway = leeway
        self.outliers = outliers
        self._stored_quantiles = tf.Variable(
            np.zeros((delay, batch_size, 5) if each_step_delay else (batch_size, 5,)), dtype=tf.float32, trainable=False
        )
        self.ll = self.q - (leeway or 0)
        self.lh = self.q + (leeway or 0)
        self.ol = outliers or 0
        self.oh = -1 - (outliers or 0)

    @tf.function
    def quantiles_to_store(self, quantiles):
        return tf.stack([
            quantiles[:,0,self.q],
            quantiles[:,0,self.ll],
            quantiles[:,0,self.lh],
            quantiles[:,0,self.ol],
            quantiles[:,0,self.oh],
        ], axis=-1)

    def store_values_each_step(self, stored_obs, obs, stored_quantiles, quantiles, stored, time_step):
        i = time_step % self.delay
        self._stored_obs[i].assign(stored_obs * 0 + obs)
        self._stored_quantiles[i].assign(stored_quantiles * 0 + self.quantiles_to_store(quantiles))
        self._stored.assign(tf.minimum(self.delay, stored + 1))

    @tf.function
    def store_values(self, stored_obs, obs, stored_quantiles, quantiles, stored, time_step):
        if time_step % self.delay == 0 or not stored:
            self._stored_obs.assign(stored_obs * 0 + obs)
            self._stored_quantiles.assign(stored_quantiles * 0 + self.quantiles_to_store(quantiles))
            self._stored.assign(stored or True)

    def successes(self, quantiles_stored, quantiles_old):
        outliers = (
            quantiles_old < tf.expand_dims(quantiles_stored[:,3], -1)
        ) | (
            quantiles_old > tf.expand_dims(quantiles_stored[:,4], -1)
        ) & (self.outliers is not None)
        decreases = (quantiles_old < tf.expand_dims(quantiles_stored[:,1], -1)) & ~outliers
        improvements = (quantiles_old > tf.expand_dims(quantiles_stored[:,2], -1)) & ~outliers
        others = ~decreases & ~improvements
        return tf.cast(improvements, tf.float32) + tf.cast(others, tf.float32) * 0.5


class MedianMinusValueActor(VarSigmaActor):
    def __init__(self, *args, **kwargs):
        VarSigmaActor.__init__(self, *args, **kwargs, std_loss_args={
            'observations': 'base.first_obs',
            'median_value_diff_weighted': 'actor.median_value_diff_weighted',
        })

        self.register_method('median_value_diff', self.median_value_diff, {'value': 'critic.value', 'median': 'critic.median'})
        self.register_method(
            'median_value_diff_weighted', self.median_value_diff_weighted,
            {'median_value_diff': 'self.median_value_diff', 'weights': 'actor.sample_weights', 'mask': 'base.mask'})

    @tf.function
    def median_value_diff(self, value, median):
        return tf.cast(median - value[..., 0], tf.float32)

    @tf.function
    def median_value_diff_weighted(self, median_value_diff, weights, mask):
        return median_value_diff[:,0]

    @tf.function
    def std_loss(self, observations, median_value_diff_weighted, **kwargs):
        mean, log_std = self.mean_and_log_std(observations)
        # median > value -> increase
        # median < value -> decrease

        return -tf.reduce_mean(log_std * tf.expand_dims(median_value_diff_weighted, 1))


class FitBetterActor(VarSigmaActor):
    def __init__(self, *args, threshold='median', **kwargs) -> None:
        THRESHOLDS = {
            'median': 'critic.median',
            'mean': 'critic.value'
        }
        super().__init__(*args, **kwargs, std_loss_args={
            'observations': 'base.first_obs',
            'actions': 'base.first_actions',
            'weights': 'self.sample_weights',
            'td': 'base.td',
            'threshold': THRESHOLDS[threshold],
        })
        assert threshold in 'median', 'mean'

    @tf.function
    def std_loss(self, observations, actions, weights, td, threshold, **kwargs):
        mean, std = self.mean_and_std(observations)
        mask = tf.cast(td >= threshold[..., :1], tf.float32)

        dist = self.distribution(
            loc=mean,
            scale_diag=std
        )

        entropy_bonus = self.entropy_bonus * tf.reduce_mean(dist.entropy())
        
        return -tf.reduce_mean(tf.expand_dims(dist.log_prob(actions), -1) * mask * weights) - entropy_bonus
        
