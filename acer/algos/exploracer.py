from typing import Optional, List, Union, Dict, Tuple
import gym
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

from algos.fast_acer import FastACER
from algos.base import BaseACERAgent, BaseActor, CategoricalActor, GaussianActor, Critic
from replay_buffer import BufferFieldSpec, VecReplayBuffer, MultiReplayBuffer


@tf.function
def _kl_diff(old_mean: tf.Tensor, mean: tf.Tensor, std: tf.Tensor) -> tf.Tensor:
    return tf.reduce_sum(0.5 * tf.square(old_mean - mean) / tf.square(std), axis=-1)


@tf.function
def _is_diff(old_mean: tf.Tensor, mean: tf.Tensor, std: tf.Tensor) -> tf.Tensor:
    return tf.exp(tf.reduce_sum(tf.square(old_mean - mean) / tf.square(std), axis=-1)) - 1


DIFF_FUNCTIONS = {
    'KL': _kl_diff,
    'IS': _is_diff
}


class VarSigmaGaussianActor(GaussianActor):

    def __init__(self, observations_space: gym.Space, actions_space: gym.Space, layers: Optional[Tuple[int]],
                 beta_penalty: float, actions_bound: float, *args, **kwargs):
        """BaseActor for continuous actions space. Uses MultiVariate Gaussian Distribution as policy distribution.

        TODO: introduce [a, b] intervals as allowed actions bounds

        Args:
            observations_dim: dimension of observations space
            layers: list of hidden layer sizes
            beta_penalty: penalty for too confident actions coefficient
            actions_bound: upper (lower == '-actions_bound') bound for allowed actions,
             required in case of continuous actions
        """
        super().__init__(
            observations_space, actions_space, layers, beta_penalty, actions_bound,
            *args, **kwargs)

        self._std_layers = self._build_layers(observations_space, layers, self.actions_dim)

    @tf.function
    def _std_forward(self, observations: np.array) -> tf.Tensor:
        x = self._std_layers[0](observations)
        for layer in self._std_layers[1:]:
            x = layer(x)
        return x

    def _dist(self, observations: np.array) -> tf.Tensor:
        mean = self._forward(observations)
        log_std = self._std_forward(observations)
        dist = tfp.distributions.MultivariateNormalDiag(
            loc=mean,
            scale_diag=tf.exp(log_std)
        )

        return dist


class DistVarSigmaGaussianActor(VarSigmaGaussianActor):
    def __init__(self, observations_space: gym.Space, actions_space: gym.Space, layers: Optional[Tuple[int]],
                 beta_penalty: float, actions_bound: float, std_loss_mult: float = 0.1, std_loss_delay: float = 0.,
                 std_diff_fun='KL', *args, **kwargs):
        """BaseActor for continuous actions space. Uses MultiVariate Gaussian Distribution as policy distribution.

        TODO: introduce [a, b] intervals as allowed actions bounds

        Args:
            observations_dim: dimension of observations space
            layers: list of hidden layer sizes
            beta_penalty: penalty for too confident actions coefficient
            actions_bound: upper (lower == '-actions_bound') bound for allowed actions,
             required in case of continuous actions
        """
        super().__init__(
            observations_space, actions_space, layers, beta_penalty, actions_bound, None,
            *args, **kwargs)

        self._diff = DIFF_FUNCTIONS[std_diff_fun]
        self._std_loss_mult = std_loss_mult
        self._std_loss_delay = std_loss_delay

    @tf.function
    def _std_loss(
        self, old_mean: tf.Tensor, mean: tf.Tensor, std: tf.Tensor, expected_entropy: tf.Tensor
    ) -> tf.Tensor:
        std_diff = self._diff(old_mean, mean, std)
        return tf.reduce_mean(tf.square(expected_entropy - std_diff))

    @tf.function
    def loss(
        self, old_mean: tf.Tensor, observations: tf.Tensor, actions: tf.Tensor,
        d: tf.Tensor, expected_entropy: tf.Tensor
    ) -> tf.Tensor:
        mean = self._forward(observations)
        log_std = self._std_forward(observations)
        std = tf.exp(log_std)

        dist = tfp.distributions.MultivariateNormalDiag(
            loc=mean,
            scale_diag=tf.stop_gradient(std)
        )

        std_loss = self._std_loss(old_mean, tf.stop_gradient(mean), std, expected_entropy)
        loss = self._loss(mean, dist, actions, d)

        std_loss = tf.where(tf.cast(self._tf_time_step, tf.float32) < self._std_loss_delay, 0., std_loss)

        with tf.name_scope('actor'):
            tf.summary.scalar('std', tf.reduce_mean(std), self._tf_time_step)
            tf.summary.scalar('std_loss', tf.reduce_mean(std_loss), self._tf_time_step)
            tf.summary.scalar('expected_loss', tf.reduce_mean(expected_entropy), self._tf_time_step)
            tf.summary.scalar('success_ratio', tf.reduce_mean(tf.cast(d > 0, tf.float32)), self._tf_time_step)
            tf.summary.scalar(
                'mean_square_means_diff',
                tf.reduce_mean(tf.square(old_mean - mean)),
                self._tf_time_step
            )
            tf.summary.scalar(
                'expected_var',
                tf.reduce_mean(tf.reduce_mean(0.5 * tf.square(old_mean - mean), -1) / expected_entropy),
                self._tf_time_step
            )

        return loss + self._std_loss_mult * std_loss


class StdVarSigmaGaussianActor(VarSigmaGaussianActor):
    def __init__(self, observations_space: gym.Space, actions_space: gym.Space, layers: Optional[Tuple[int]],
                 beta_penalty: float, actions_bound: float, entropy_coeff: float = 0.0, *args, **kwargs):
        """BaseActor for continuous actions space. Uses MultiVariate Gaussian Distribution as policy distribution.

        TODO: introduce [a, b] intervals as allowed actions bounds

        Args:
            observations_dim: dimension of observations space
            layers: list of hidden layer sizes
            beta_penalty: penalty for too confident actions coefficient
            actions_bound: upper (lower == '-actions_bound') bound for allowed actions,
             required in case of continuous actions
        """
        super().__init__(
            observations_space, actions_space, layers, beta_penalty, actions_bound, None,
            *args, **kwargs)

        self._entropy_coeff = entropy_coeff

    @tf.function
    def loss(
        self, observations: tf.Tensor, actions: tf.Tensor, d: tf.Tensor) -> tf.Tensor:
        mean = self._forward(observations)
        log_std = self._std_forward(observations)
        std = tf.exp(log_std)

        dist = tfp.distributions.MultivariateNormalDiag(
            loc=mean,
            scale_diag=std
        )

        loss = self._loss(mean, dist, actions, d)

        with tf.name_scope('actor'):
            tf.summary.scalar('std', tf.reduce_mean(std), self._tf_time_step)

        return loss + -self._entropy_coeff * log_std


class DistExplorACER(FastACER):
    def __init__(
            self, observations_space: gym.Space, actions_space: gym.Space, *args,
            entropy_coeff: float = 1, std_loss_mult: float = 0.1, std_loss_delay: float = 0,
            std_diff_fun: str = 'KL', **kwargs):
        self._entropy_coeff = entropy_coeff
        self._std_loss_mult = std_loss_mult
        self._std_diff_fun = std_diff_fun
        self._std_loss_delay = std_loss_delay
        super().__init__(
            observations_space, actions_space,
            *args, **kwargs, additional_buffer_types=(tf.dtypes.int32,),
            policy_spec=BufferFieldSpec((2, actions_space.shape[0])))


    def _init_actor(self) -> BaseActor:
        if self._is_discrete:
            raise NotImplementedError
        else:
            return DistVarSigmaGaussianActor(
                self._observations_space, self._actions_space, self._actor_layers,
                self._actor_beta_penalty, self._actions_bound,
                self._std_loss_mult, self._std_loss_delay, self._std_diff_fun,
                self._tf_time_step
            )

    @tf.function(experimental_relax_shapes=True)
    def _learn_from_experience_batch(
        self, obs, obs_next, actions, old_policies, rewards,
        first_obs, first_actions, dones, lengths, time):
        """Backward pass with single batch of experience.

        Every experience replay requires sequence of experiences with random length, thus we have to use
        ragged tensors here.

        See Equation (8) and Equation (9) in the paper (1).
        """

        obs = self._process_observations(obs)
        obs_next = self._process_observations(obs_next)
        rewards = self._process_rewards(rewards)

        old_first_means = old_policies[:, 0, 1, :]
        old_policies = old_policies[:, :, 0 ,0]

        policies, _ = self._actor.prob(obs, actions)
        mask = tf.sequence_mask(lengths, maxlen=self._n, dtype=tf.float32)

        td = self._calculate_td(obs, obs_next, rewards, lengths, dones, mask)
        truncated_density = self._calculate_truncated_density(policies, old_policies, mask)

        # expected entropy
        window_size = tf.minimum(tf.cast(self._tf_time_step, tf.float32), tf.constant(self._memory_size, tf.float32))
        expected_entropy = self._entropy_coeff * tf.cast(time, tf.float32) / tf.cast(window_size, tf.float32)

        # expected_entropy = tf.where(td >= 0, 0.8 * tf.ones_like(td), -0.2 * tf.ones_like(td)) * truncated_density
        # expected_entropy = tf.reduce_sum(expected_entropy, axis=1, keepdims=True)

        d = tf.stop_gradient(td * truncated_density)

        self._actor_backward_pass(old_first_means, first_obs, first_actions, d, expected_entropy)
        self._critic_backward_pass(first_obs, d)

    @tf.function
    def _actor_backward_pass(
        self, old_mean: tf.Tensor, observations: tf.Tensor, 
        actions: tf.Tensor, d: tf.Tensor, expected_std_diff: tf.Tensor
    ):
        with tf.GradientTape() as tape:
            loss = self._actor.loss(old_mean, observations, actions, d, expected_std_diff)
        grads = tape.gradient(loss, self._actor.trainable_variables)
        if self._gradient_norm is not None:
            grads = self._clip_gradient(grads, self._actor_gradient_norm_median, 'actor')
        gradients = zip(grads, self._actor.trainable_variables)

        self._actor_optimizer.apply_gradients(gradients)

    def _experience_replay_generator(self):
        """Generates trajectories batches. All tensors are padded with zeros to match self._n number of
        experience tuples in a single trajectory.
        Trajectories are returned in shape [batch, self._n, <obs/actions/etc shape>]
        """
        while True:
            offline_batches, lens = self._fetch_offline_batch()
            
            lengths = lens
            obs = offline_batches['observations']
            obs_next = offline_batches['next_observations'][np.arange(self._batch_size),lens - 1]
            actions = offline_batches['actions']
            rewards = offline_batches['rewards']
            policies = offline_batches['policies']
            dones = offline_batches['dones'][np.arange(self._batch_size),lens - 1]
            time = offline_batches['time']

            yield (
                obs,
                obs_next,
                actions,
                policies,
                rewards,
                obs[:,0],
                actions[:,0],
                dones,
                lengths,
                time
            )


class StdExplorACER(FastACER):
    def __init__(
            self, observations_space: gym.Space, actions_space: gym.Space, *args,
            entropy_coeff: float = 0, **kwargs):
        self._entropy_coeff = entropy_coeff
        super().__init__(observations_space, actions_space, *args, **kwargs)


    def _init_actor(self) -> BaseActor:
        if self._is_discrete:
            raise NotImplementedError
        else:
            return StdVarSigmaGaussianActor(
                self._observations_space, self._actions_space, self._actor_layers,
                self._actor_beta_penalty, self._actions_bound,
                self._entropy_coeff, self._tf_time_step
            )