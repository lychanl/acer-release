from algos.varsigmaactors import VarSigmaActor

import tensorflow as tf


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
