from typing import Optional, List, Union, Dict, Tuple
import gym
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

from algos.base import BaseACERAgent, BaseActor, CategoricalActor, GaussianActor, Critic


class ExplorerGaussianActor(GaussianActor):
    def __init__(self, observations_space: gym.Space, actions_space: gym.Space, layers: Optional[Tuple[int]],
                 beta_penalty: float, actions_bound: float, explorer, *args, **kwargs):
        super().__init__(observations_space, actions_space, layers, beta_penalty, actions_bound, *args, **kwargs)
        self._explorer = explorer

    def loss(self, observations: np.array, actions: np.array, d: np.array) -> tf.Tensor:
        mean = self._forward(observations)
        std = tf.exp(self.log_std) * tf.sqrt(tf.exp(self._explorer(observations)))
        dist = tfp.distributions.MultivariateNormalDiag(
            loc=mean,
            scale_diag=std
        )

        action_log_probs = tf.expand_dims(dist.log_prob(actions), axis=1)

        bounds_penalty = tf.reduce_sum(
            tf.scalar_mul(
                self._beta_penalty,
                tf.square(tf.maximum(0.0, tf.abs(mean) - self._actions_bound))
            ),
            axis=1,
            keepdims=True
        )
        entropy = dist.entropy()
        # entropy_penalty = 0.01 * entropy

        total_loss = tf.reduce_mean(-tf.math.multiply(action_log_probs, d) + bounds_penalty)

        with tf.name_scope('actor'):
            tf.summary.scalar('mean_std', tf.reduce_mean(std), step=self._tf_time_step)
            tf.summary.scalar('max_std', tf.reduce_max(std), step=self._tf_time_step)
            tf.summary.scalar('min_std', tf.reduce_min(std), step=self._tf_time_step)
            tf.summary.scalar('batch_loss', total_loss, step=self._tf_time_step)
            tf.summary.scalar('batch_bounds_penalty_mean', tf.reduce_mean(bounds_penalty), step=self._tf_time_step)
            tf.summary.scalar('batch_entropy_mean', tf.reduce_mean(entropy), step=self._tf_time_step)

        return total_loss

    def prob(self, observations: tf.Tensor, actions: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        mean = self._forward(observations)
        dist = tfp.distributions.MultivariateNormalDiag(
            loc=mean,
            scale_diag=tf.exp(self.log_std) * tf.sqrt(tf.exp(self._explorer(observations)))
        )

        return dist.prob(actions), dist.log_prob(actions)

    def act(self, observations: tf.Tensor, **kwargs) -> Tuple[tf.Tensor, tf.Tensor]:
        mean = self._forward(observations)

        dist = tfp.distributions.MultivariateNormalDiag(
            loc=mean,
            scale_diag=tf.exp(self.log_std) * tf.sqrt(tf.exp(self._explorer(observations)))
        )

        actions = dist.sample(dtype=self.dtype)
        actions_probs = dist.prob(actions)

        with tf.name_scope('actor'):
            for i in range(self._actions_dim):
                tf.summary.scalar(f'batch_action_{i}_mean', tf.reduce_mean(actions[:, i]), step=self._tf_time_step)

        return actions, actions_probs


class ACERCE(BaseACERAgent):
    def __init__(self, observations_space: gym.Space, actions_space: gym.Space, actor_layers: Optional[Tuple[int]],
                 critic_layers: Optional[Tuple[int]], lam: float = 0.1, b: float = 3, alpha: float = 0.8,
                 explorer_lr: float = 0.001,
                 *args, **kwargs):

        super().__init__(observations_space, actions_space, actor_layers, critic_layers, *args, **kwargs)
        cov_matrix = tf.linalg.diag(tf.square(tf.exp(self._actor.log_std)))
        self._c_inverse = tf.linalg.inv(cov_matrix)
        self._n = 1 / tf.sqrt(((2 * np.pi) ** self._actions_space.shape[0]) * tf.linalg.det(cov_matrix))
        self._lam = lam
        self._b = b
        self._alpha = alpha

        self._explorer_optimizer = tf.keras.optimizers.Adam(
            lr=explorer_lr
        )

    def _init_actor(self) -> BaseActor:
        self._explorer = Critic(self._observations_space, self._critic_layers, self._tf_time_step)
        if self._is_discrete:
            return CategoricalActor(
                self._observations_space, self._actions_space, self._actor_layers,
                self._actor_beta_penalty, self._tf_time_step
            )
        else:
            return ExplorerGaussianActor(
                self._observations_space, self._actions_space, self._actor_layers,
                self._actor_beta_penalty, self._actions_bound, self._explorer, self._std, self._tf_time_step,
            )

    def _init_critic(self) -> Critic:
        return Critic(self._observations_space, self._critic_layers, self._tf_time_step)

    def learn(self):
        experience_replay_iterations = min([round(self._c0 * self._time_step / self._num_parallel_envs), self._c])

        for batch in self._data_loader.take(experience_replay_iterations):
            self._learn_from_experience_batch(*batch)

    @tf.function(experimental_relax_shapes=True)
    def _learn_from_experience_batch(self, obs, obs_next, actions, old_policies,
                                     rewards, first_obs, first_actions, dones, lengths):
        batches_indices = tf.RaggedTensor.from_row_lengths(values=tf.range(tf.reduce_sum(lengths)), row_lengths=lengths)
        values = tf.squeeze(self._critic.value(obs))
        rho = tf.squeeze(self._explorer.value(obs))
        values_next = tf.squeeze(self._critic.value(obs_next)) * (1.0 - tf.cast(dones, tf.dtypes.float32))
        policies, log_policies = tf.split(self._actor.prob(obs, actions), 2, axis=0)
        policies, log_policies = tf.squeeze(policies), tf.squeeze(log_policies)
        indices = tf.expand_dims(batches_indices, axis=2)

        # flat tensor
        policies_ratio = tf.math.divide(policies, old_policies)
        # ragged tensor divided into batches
        policies_ratio_batches = tf.squeeze(tf.gather(policies_ratio, indices), axis=2)

        # cumprod and cumsum do not work on ragged tensors, we transform them into tensors
        # padded with 0 and then apply boolean mask to retrieve original ragged tensor
        batch_mask = tf.sequence_mask(policies_ratio_batches.row_lengths())
        policies_ratio_product = tf.math.cumprod(policies_ratio_batches.to_tensor(), axis=1)

        truncated_densities = tf.ragged.boolean_mask(
            tf.tanh(policies_ratio_product / self._b) * self._b,
            batch_mask
        )

        gamma_coeffs_batches = tf.ones_like(policies_ratio_batches).to_tensor() * self._gamma
        gamma_coeffs = tf.ragged.boolean_mask(
            tf.math.cumprod(gamma_coeffs_batches, axis=1, exclusive=True),
            batch_mask
        ).flat_values

        # flat tensors
        d_coeffs = gamma_coeffs * (rewards + self._gamma * values_next - values) * truncated_densities.flat_values
        # ragged
        d_coeffs_batches = tf.gather_nd(d_coeffs, tf.expand_dims(indices, axis=2))
        # final summation over original batches
        d = tf.stop_gradient(tf.reduce_sum(d_coeffs_batches, axis=1))

        mu = self._actor.act_deterministic(obs)
        actions_diff = tf.expand_dims(actions - mu, axis=1)
        h_i = tf.matmul(tf.matmul(actions_diff, self._c_inverse), tf.linalg.matrix_transpose(actions_diff))
        exp_rho_h = tf.expand_dims(tf.squeeze(h_i) * tf.exp(-rho), axis=1)

        kappa = self._n / tf.sqrt(2.0) * tf.exp(-0.5 * rho)

        with tf.name_scope('actor'):
            tf.summary.scalar('forced_ratio', tf.reduce_mean(policies / tf.squeeze(kappa)), self._tf_time_step)

        # exploration_gain = \
            # tf.sqrt(2.0) * (self._alpha - tf.sqrt(2.0) * tf.exp(-0.5 * exp_rho_h)) * tf.exp(-0.5 * exp_rho_h) * exp_rho_h

        exploration_gain = (self._alpha - tf.sqrt(2.0) * tf.exp(-0.5 * exp_rho_h))

        self._backward_pass(first_obs, first_actions, d, exploration_gain, obs)

        _, new_log_policies = tf.split(self._actor.prob(obs, actions), 2, axis=0)
        new_log_policies = tf.squeeze(new_log_policies)
        approx_kl = tf.reduce_mean(policies - new_log_policies)
        with tf.name_scope('actor'):
            tf.summary.scalar('sample_approx_kl_divergence', approx_kl, self._tf_time_step)

    def _backward_pass(self, observations: tf.Tensor, actions: tf.Tensor, d: tf.Tensor,
                       exploration_gain: tf.Tensor, observations_exploration):
        with tf.GradientTape() as tape:
            loss = self._actor.loss(observations, actions, d)
        grads = tape.gradient(loss, self._actor.trainable_variables)
        gradients = zip(grads, self._actor.trainable_variables)

        self._actor_optimizer.apply_gradients(gradients)

        with tf.GradientTape() as tape:
            loss = self._critic.loss(observations, d)
        grads = tape.gradient(loss, self._critic.trainable_variables)
        gradients = zip(grads, self._critic.trainable_variables)

        self._critic_optimizer.apply_gradients(gradients)

        with tf.GradientTape() as tape:
            loss = self._explorer.loss(observations_exploration, exploration_gain)
        grads = tape.gradient(loss, self._explorer.trainable_variables)
        gradients = zip(grads, self._explorer.trainable_variables)

        self._explorer_optimizer.apply_gradients(gradients)

    def _fetch_offline_batch(self) -> List[Dict[str, Union[np.array, list]]]:
        trajectory_lens = [np.random.geometric(1 - self._lam) + 1 for _ in range(self._num_parallel_envs)]
        batch = []
        [batch.extend(self._memory.get(trajectory_lens)) for _ in range(self._batches_per_env)]
        return batch
