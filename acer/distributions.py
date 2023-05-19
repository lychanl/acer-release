import tensorflow as tf
import tensorflow_probability as tfp


class Distribution:
    def __init__(self, *args, **kwargs):
        pass

    def prob(self, val):
        pass

    def log_prob(self, val):
        pass

    def sample(self):
        pass

    def mode(self):
        pass

    def entropy(self):
        pass

    @tf.function
    def sample_with_log_prob(self):
        sample = self.sample()
        return sample, self.log_prob(sample)


class MultivariateNormalDiag(Distribution):
    def __init__(self, loc, scale_diag) -> None:
        self._distr = tfp.distributions.Normal(loc, scale_diag)
        self._loc = loc

    @tf.function
    def prob(self, val):
        return tf.reduce_prod(self._distr.prob(val), axis=-1)

    @tf.function
    def log_prob(self, val):
        return tf.reduce_sum(self._distr.log_prob(val), axis=-1)

    @tf.function
    def sample(self):
        return self._distr.sample()

    @tf.function
    def mode(self):
        return self._loc

    @tf.function
    def entropy(self):
        return self._distr.entropy()


class SquashedMultivariateNormalDiag(MultivariateNormalDiag):
    epsilon = 1e-6

    @tf.function
    def prob(self, val):
        gaussian = tf.math.atanh(val)
        return super().prob(gaussian) / tf.reduce_prod(1 - val ** 2, axis=-1)

    @tf.function
    def log_prob(self, val):
        gaussian = tf.math.atanh(val)
        return self._log_prob(val, gaussian)

    @tf.function
    def _log_prob(self, val, gaussian):
        return MultivariateNormalDiag.log_prob(self, gaussian) - tf.reduce_sum(tf.math.log(1 - val ** 2), axis=-1)

    @tf.function
    def _sample(self):
        gaussian = self._distr.sample()
        sample = tf.clip_by_value(tf.tanh(gaussian), -1 + self.epsilon, 1 - self.epsilon)
        return gaussian, sample

    @tf.function
    def sample(self):
        return self._sample()[1]

    @tf.function
    def mode(self):
        return tf.tanh(self._loc)

    @tf.function
    def sample_with_log_prob(self):
        gaussian, sample = self._sample()
        log_prob = self._log_prob(sample, gaussian)
        return sample, log_prob


class SquashedMultivariateNormalDiagV2(SquashedMultivariateNormalDiag):
    epsilon = 1e-6

    @tf.function
    def prob(self, val):
        gaussian = tf.math.atanh(val)
        return super().prob(gaussian) / tf.reduce_prod(1 - val ** 2 + self.epsilon, axis=-1)

    @tf.function
    def _log_prob(self, val, gaussian):
        return super().log_prob(gaussian) - tf.reduce_sum(tf.math.log(1 - val ** 2 + self.epsilon), axis=-1)

    @tf.function
    def _sample(self):
        gaussian = self._distr.sample()
        sample = tf.tanh(gaussian)
        return gaussian, sample


DISTRIBUTIONS = {
    'normal': MultivariateNormalDiag,
    'squashed': SquashedMultivariateNormalDiag,
    'squashed-v2': SquashedMultivariateNormalDiagV2
}
