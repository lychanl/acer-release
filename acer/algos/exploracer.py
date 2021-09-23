from typing import Optional, List, Union, Dict, Tuple
import gym
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

from algos.fast_acer import FastACER
from algos.base import BaseACERAgent, BaseActor, CategoricalActor, GaussianActor, Critic
from replay_buffer import BufferFieldSpec, VecReplayBuffer, MultiReplayBuffer


@tf.function
def _kl_diff(ratio: tf.Tensor) -> tf.Tensor:
    return 0.5 * ratio


@tf.function
def _is_diff(ratio: tf.Tensor) -> tf.Tensor:
    return tf.exp(tf.minimum(ratio, 10)) - 1


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

    # @tf.function
    def _std_forward(self, observations: np.array) -> tf.Tensor:
        batch_dims = observations.shape[:-len(self.obs_shape)]
        
        x = tf.reshape(observations, (-1,) + self.obs_shape)
        for layer in self._std_layers:
            x = layer(x)
        
        return tf.reshape(x, batch_dims + (self.actions_dim,))

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
                 beta_penalty: float, actions_bound: float, std_loss_mult: float = 0.1,
                 std_diff_fun='KL', b=None, h=None, *args, **kwargs):
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
        self._b = b
        self._h = h
        self._std0 = 1

    # @tf.function
    def _std_loss(
        self, old_mean: tf.Tensor, mean: tf.Tensor, std: tf.Tensor, expected_entropy: tf.Tensor
    ) -> tf.Tensor:
        ratio = tf.reduce_sum(tf.square(old_mean - mean) / tf.square(std), axis=-1)
        std_diff = self._diff(ratio)

        """
        with tf.name_scope('actor_loss'):
            tf.summary.scalar('mean ratio', tf.reduce_mean(ratio), self._tf_time_step)
            tf.summary.scalar('max ratio', tf.reduce_max(ratio), self._tf_time_step)
            tf.summary.scalar('min ratio', tf.reduce_min(ratio), self._tf_time_step)
            tf.summary.scalar('ratio fin', tf.reduce_all(tf.math.is_finite(ratio)), self._tf_time_step)
            tf.summary.scalar('mean diff', tf.reduce_mean(std_diff), self._tf_time_step)
            tf.summary.scalar('max diff', tf.reduce_max(std_diff), self._tf_time_step)
            tf.summary.scalar('min diff', tf.reduce_min(std_diff), self._tf_time_step)
            tf.summary.scalar('diff fin', tf.reduce_all(tf.math.is_finite(std_diff)), self._tf_time_step)
        """

        if self._b:
            psi_std_diff = tf.tanh(std_diff / self._b) * self._b
            std_loss = tf.reduce_mean(tf.square(expected_entropy - psi_std_diff))
        else:
            std_loss = tf.reduce_mean(tf.square(expected_entropy - std_diff))

        if self._h:
            h_part = (1 - psi_std_diff / (std_diff + 1e-6))\
                * tf.reduce_sum(tf.square(tf.math.log(std / self._std0) / (tf.ones_like(std) * self._h)), axis=-1)

            """
            with tf.name_scope('actor_loss'):
                tf.summary.scalar('mean h_part', tf.reduce_mean(h_part), self._tf_time_step)
                tf.summary.scalar('max h_part', tf.reduce_max(h_part), self._tf_time_step)
                tf.summary.scalar('min h_part', tf.reduce_min(h_part), self._tf_time_step)
                tf.summary.scalar('h_part fin', tf.reduce_all(tf.math.is_finite(h_part)), self._tf_time_step)
            """

            std_loss = std_loss + tf.reduce_sum(h_part)

        return std_loss

    # @tf.function
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

        with tf.name_scope('actor'):
            tf.summary.scalar('std', tf.reduce_mean(std), self._tf_time_step)
            tf.summary.scalar('min_std', tf.reduce_min(std), self._tf_time_step)
            tf.summary.scalar('max_std', tf.reduce_max(std), self._tf_time_step)
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
            entropy_coeff: float = 1, std_loss_mult: float = 0.1,
            std_diff_fun: str = 'KL', diff_b=None, diff_h=None,  **kwargs):
        self._entropy_coeff = entropy_coeff
        self._std_loss_mult = std_loss_mult
        self._std_diff_fun = std_diff_fun
        self._diff_b = diff_b
        self._diff_h = diff_h
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
                self._std_loss_mult, self._std_diff_fun,
                self._diff_b, self._diff_h,
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
        # expected_entropy = 1 - tf.cast(time, tf.float32) / tf.cast(window_size, tf.float32)

        # expected_entropy = tf.where(td >= 0, 0.8 * tf.ones_like(td), -0.2 * tf.ones_like(td)) * truncated_density
        # expected_entropy = tf.reduce_sum(expected_entropy, axis=1, keepdims=True)

        d = tf.stop_gradient(td * truncated_density)

        self._actor_backward_pass(old_first_means, first_obs, first_actions, d, expected_entropy)
        self._critic_backward_pass(first_obs, d)

    # @tf.function
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


class SingleSigmaActor(GaussianActor):
    def __init__(self, observations_space: gym.Space, actions_space: gym.Space, layers: Optional[Tuple[int]],
                 beta_penalty: float, actions_bound: float, *args, **kwargs):
        super().__init__(
            observations_space, actions_space, layers, beta_penalty, actions_bound,
            *args, **kwargs)
        self._log_std = tf.Variable(self.log_std - 1)


    def _dist(self, observations: np.array) -> tf.Tensor:
        mean = self._forward(observations)
        dist = tfp.distributions.MultivariateNormalDiag(
            loc=mean,
            scale_diag=tf.exp(self._log_std)
        )

        return dist
        
    @tf.function
    def loss(
        self, observations: tf.Tensor, actions: tf.Tensor, d: tf.Tensor, target_std: tf.Tensor) -> tf.Tensor:
        mean = self._forward(observations)
        std = tf.exp(self._log_std)

        dist = tfp.distributions.MultivariateNormalDiag(
            loc=mean,
            scale_diag=tf.stop_gradient(std) * tf.ones_like(mean)
        )

        std_loss = tf.reduce_sum(tf.square(std - tf.stop_gradient(target_std)))

        loss = self._loss(mean, dist, actions, d) + std_loss

        with tf.name_scope('actor'):
            tf.summary.scalar('std', tf.reduce_mean(std), self._tf_time_step)
            tf.summary.scalar('target_std', tf.reduce_mean(target_std), self._tf_time_step)

        return loss


class SingleSigmaExplorACER(FastACER):
    def _init_actor(self) -> BaseActor:
        if self._is_discrete:
            raise NotImplementedError
        else:
            return SingleSigmaActor(
                self._observations_space, self._actions_space, self._actor_layers,
                self._actor_beta_penalty, self._actions_bound, self._std,
                self._tf_time_step,
            )
 
    @tf.function(experimental_relax_shapes=True)
    def _learn_from_experience_batch(
        self, obs, obs_next, actions, old_policies, rewards,
        first_obs, first_actions, dones, lengths):
        """Backward pass with single batch of experience.

        Every experience replay requires sequence of experiences with random length, thus we have to use
        ragged tensors here.

        See Equation (8) and Equation (9) in the paper (1).
        """

        obs = self._process_observations(obs)
        obs_next = self._process_observations(obs_next)
        rewards = self._process_rewards(rewards)
        policies, _ = self._actor.prob(obs, actions)

        mask = tf.sequence_mask(lengths, maxlen=self._n, dtype=tf.float32)

        td = self._calculate_td(obs, obs_next, rewards, lengths, dones, mask)
        truncated_density = self._calculate_truncated_density(policies, old_policies, mask)

        expected_std = tf.reduce_mean(tf.abs(self._actor._forward(obs) - actions), axis=(0, 1))  * tf.sqrt(2 / np.pi)

        d = tf.stop_gradient(td * truncated_density)

        self._actor_backward_pass(first_obs, first_actions, d, expected_std)
        self._critic_backward_pass(first_obs, d)

    # @tf.function
    def _actor_backward_pass(
        self, observations: tf.Tensor, 
        actions: tf.Tensor, d: tf.Tensor, expected_std: tf.Tensor
    ):
        with tf.GradientTape() as tape:
            loss = self._actor.loss(observations, actions, d, expected_std)
        grads = tape.gradient(loss, self._actor.trainable_variables)
        if self._gradient_norm is not None:
            grads = self._clip_gradient(grads, self._actor_gradient_norm_median, 'actor')
        gradients = zip(grads, self._actor.trainable_variables)

        self._actor_optimizer.apply_gradients(gradients)


class MultiSigmaActor(VarSigmaGaussianActor):
    def __init__(self, observations_space: gym.Space, actions_space: gym.Space, layers: Optional[Tuple[int]],
                 beta_penalty: float, actions_bound: float, alpha, rho, std_loss_mult, *args, **kwargs):
        self._alpha = alpha
        self._rho = rho
        self._std_loss_mult = std_loss_mult
        super().__init__(
            observations_space, actions_space, layers, beta_penalty, actions_bound,
            *args, **kwargs)

    @tf.function
    def loss(
        self, observations: tf.Tensor, actions: tf.Tensor, d: tf.Tensor, target_std: tf.Tensor, target_std2, std_weights: tf.Tensor) -> tf.Tensor:
        mean = self._forward(observations)
        eta = self._std_forward(observations)
        std = tf.exp(eta)

        dist = tfp.distributions.MultivariateNormalDiag(
            loc=mean,
            scale_diag=tf.stop_gradient(std) * tf.ones_like(mean)
        )

        return self._loss(mean, dist, actions, d)

    @tf.function
    def std_loss(
            self, observations: tf.Tensor, actions: tf.Tensor, d: tf.Tensor, target_std: tf.Tensor, target_std2, std_weights: tf.Tensor) -> tf.Tensor:
        eta = self._std_forward(observations)
        std = tf.exp(eta)

        rho_std = tf.square(self._rho * std)

        if self._alpha >= 0:
            std_loss = tf.reduce_sum(0.5 * target_std / rho_std, -1)\
                + self._alpha * tf.reduce_sum(0.5 * target_std2 / rho_std, -1)\
                + (1 + self._alpha) * tf.reduce_sum(eta, -1)
        else:
            std_loss = tf.reduce_sum(0.5 * target_std2 / rho_std, -1) + tf.reduce_sum(eta, -1)

        with tf.name_scope('actor'):
            tf.summary.scalar('std', tf.reduce_mean(std), self._tf_time_step)
            tf.summary.scalar('target_std', tf.reduce_mean(target_std), self._tf_time_step)
            tf.summary.scalar('target_std2', tf.reduce_mean(target_std2), self._tf_time_step)
            tf.summary.scalar(
                'weighted_target_std',
                tf.reduce_sum(tf.reduce_mean(target_std, axis=-1) * std_weights) / tf.reduce_sum(std_weights),
                self._tf_time_step
            )
            tf.summary.scalar(
                'weighted_target_std2',
                tf.reduce_sum(tf.reduce_mean(target_std2, axis=-1) * std_weights) / tf.reduce_sum(std_weights),
                self._tf_time_step
            )

        return std_loss

 
    # @tf.function
    def _std_forward(self, observations: np.array) -> tf.Tensor:
        return super(MultiSigmaActor, self)._std_forward(observations) + tf.expand_dims(self.log_std, 0)


    @tf.function
    def _act(self, dist: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        actions = dist.sample(dtype=self.dtype)
        actions_probs = dist.prob(actions)

        with tf.name_scope('actor'):
            tf.summary.scalar(f'batch_action_mean', tf.reduce_mean(actions), step=self._tf_time_step)
            tf.summary.scalar(f'batch_std_mean', tf.reduce_mean(dist.scale.H.diag), step=self._tf_time_step)

        return actions, tf.concat([[actions_probs], dist.mode()], axis=1)


class MultiSigmaExplorACER(FastACER):
    def __init__(
            self, actions_space, *args, time_coeff='linear', alpha=1, rho=1, std_loss_mult=1,
            actor_lr=0, actor_adam_beta1=0, actor_adam_beta2=0, actor_adam_epsilon=0, **kwargs):
        self._time_coeff = time_coeff
        self._alpha = alpha
        self._rho = rho
        self._std_loss_mult = std_loss_mult

        super().__init__(*args, actions_space=actions_space, **kwargs, additional_buffer_types=(tf.dtypes.int32,), policy_spec=BufferFieldSpec((1 + actions_space.shape[0],)))

        self._explor_optimizer = tf.keras.optimizers.Adam(
            lr=actor_lr * std_loss_mult,
            beta_1=actor_adam_beta1,
            beta_2=actor_adam_beta2,
            epsilon=actor_adam_epsilon
        )

    def _init_actor(self) -> BaseActor:
        if self._is_discrete:
            raise NotImplementedError
        else:
            return MultiSigmaActor(
                self._observations_space, self._actions_space, self._actor_layers,
                self._actor_beta_penalty, self._actions_bound, self._alpha, self._rho, self._std_loss_mult,
                self._std, self._tf_time_step,
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

        old_expected_actions = old_policies[:,:,1:]
        old_policies = old_policies[:,:,0]

        obs = self._process_observations(obs)
        obs_next = self._process_observations(obs_next)
        rewards = self._process_rewards(rewards)
        policies, _ = self._actor.prob(obs, actions)

        mask = tf.sequence_mask(lengths, maxlen=self._n, dtype=tf.float32)

        td = self._calculate_td(obs, obs_next, rewards, lengths, dones, mask)
        truncated_density = self._calculate_truncated_density(policies, old_policies, mask)

        window_size = tf.minimum(tf.cast(self._tf_time_step, tf.float32), tf.constant(self._memory_size, tf.float32))
        time_coeff = tf.ones_like(time, tf.float32) if self._time_coeff == 'none' else (
            2 * (1 - tf.cast(time, tf.float32) / window_size )
        ) if self._time_coeff == 'linear' else (
            tf.exp(-tf.cast(time, tf.float32) / window_size * 2) * window_size * (1 - tf.exp(-2 / window_size))
        ) 

        modes = self._actor._dist(obs).mode()
        expected_std = tf.stop_gradient(tf.square(old_expected_actions[:,0,:] - modes[:,0,:]))
        expected_std2 = tf.stop_gradient(tf.square(actions[:,0,:] - modes[:,0,:]))

        d = tf.stop_gradient(td * truncated_density)

        self._actor_backward_pass(first_obs, first_actions, d, expected_std, expected_std2, time_coeff)
        self._critic_backward_pass(first_obs, d)

    @tf.function
    def _actor_backward_pass(
        self, observations: tf.Tensor, 
        actions: tf.Tensor, d: tf.Tensor, expected_std: tf.Tensor, expected_std2, std_weights: tf.Tensor
    ):
        with tf.GradientTape() as tape:
            loss = self._actor.loss(observations, actions, d, expected_std, expected_std2, std_weights)
        grads = tape.gradient(loss, self._actor.trainable_variables)
        if self._gradient_norm is not None:
            grads = self._clip_gradient(grads, self._actor_gradient_norm_median, 'actor')

        with tf.GradientTape() as std_tape:
            std_loss = self._actor.std_loss(observations, actions, d, expected_std, expected_std2, std_weights)
        std_grads = std_tape.gradient(std_loss, self._actor.trainable_variables)

        gradients = zip(grads, self._actor.trainable_variables)
        self._actor_optimizer.apply_gradients(gradients)

        std_gradients = zip(std_grads, self._actor.trainable_variables)
        self._actor_optimizer.apply_gradients(std_gradients)

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
