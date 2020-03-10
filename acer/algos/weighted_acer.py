from pathlib import Path
from typing import Optional, List, Union, Dict, Tuple
# import os
# os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import gym
import tensorflow as tf
import numpy as np

from algos.base import BaseACERAgent, BaseActor, CategoricalActor, GaussianActor, BaseCritic, Critic


class WeightCritic(BaseCritic):

    def __init__(self, observations_space: gym.Space, layers: Optional[Tuple[int]], tf_time_step: tf.Variable, *args,
                 **kwargs):
        super().__init__(observations_space, layers, tf_time_step, *args, **kwargs)
        self._init_zero(observations_space)

    def _init_zero(self, observations_space: gym.Space):
        self.compile(loss=tf.keras.losses.mean_absolute_error)
        data = np.random.uniform(
            1e20, -1e20, (int(1e4), observations_space.low.shape[0])
        )
        self.fit(data, np.zeros(shape=(len(data), )), epochs=2)

    def value(self, observations: tf.Tensor,  **kwargs) -> Tuple[tf.Tensor, tf.Tensor]:
        """Calculates lower bound of the value function and value function approximation

        Args:
            observations: batch [batch_size, observations_dim] of observations vectors

        Returns:
            Tuple of two Tensors: ([batch_size, 1], [batch_size, 1]) with value function lower bound
                and value function approximations

        """
        x = self._hidden_layers[0](observations)

        for layer in self._hidden_layers[1:]:
            x = layer(x)

        v = tf.exp(self._v(x))

        return v

    def loss_v2(self, observations: np.array, policies: tf.Tensor, old_policies: tf.Tensor) -> tf.Tensor:
        """Computes Critic's loss.

        Args:
            old_policies: batch [batch_size, observations_dim] of policies from the buffer
            policies: batch [batch_size, observations_dim] of current policies
            observations: batch [batch_size, observations_dim] of observations vectors
        """
        value = self.value(observations)

        loss = tf.losses.mean_squared_error(policies, tf.multiply(value, old_policies))

        with tf.name_scope('critic'):
            tf.summary.scalar('batch_v2_loss', tf.reduce_mean(loss), step=self._tf_time_step)
            tf.summary.scalar('batch_v2_mean', tf.reduce_mean(value), step=self._tf_time_step)
        return loss


class WeightedACER(BaseACERAgent):
    def __init__(self, observations_space: gym.Space, actions_space: gym.Space, actor_layers: Optional[Tuple[int]],
                 critic_layers: Optional[Tuple[int]], lam: float = 0.1, b: float = 3, *args, **kwargs):
        """Actor Critic with weighted importance sampling

        TODO: finish docstrings
        """

        super().__init__(observations_space, actions_space, actor_layers, critic_layers, *args, **kwargs)
        self._lam = lam
        self._b = b

        self._critic_v2 = WeightCritic(self._observations_space, self._critic_layers, self._tf_time_step)

        self._critic_v2_optimizer = tf.keras.optimizers.Adam(
            lr=0.001,
            beta_1=kwargs['critic_adam_beta1'],
            beta_2=kwargs['critic_adam_beta2'],
            epsilon=kwargs['critic_adam_epsilon'],
        )

    def _init_actor(self) -> BaseActor:
        if self._is_discrete:
            return CategoricalActor(
                self._observations_space, self._actions_space, self._actor_layers,
                self._actor_beta_penalty, self._tf_time_step
            )
        else:
            return GaussianActor(
                self._observations_space, self._actions_space, self._actor_layers,
                self._actor_beta_penalty, self._actions_bound, self._std, self._tf_time_step
            )

    def _init_critic(self) -> BaseCritic:
        return Critic(self._observations_space, self._critic_layers, self._tf_time_step)

    def learn(self):
        """
        Performs experience replay learning. Experience trajectory is sampled from every replay buffer once, thus
        single backwards pass batch consists of 'num_parallel_envs' trajectories.

        Every call executes N of backwards passes, where: N = min(c0 * time_step / num_parallel_envs, c).
        That means at the beginning experience replay intensity increases linearly with number of samples
        collected till c value is reached.
        """
        experience_replay_iterations = min([round(self._c0 * self._time_step / self._num_parallel_envs), self._c])

        for batch in self._data_loader.take(experience_replay_iterations):
            self._learn_from_experience_batch(*batch)

    @tf.function(experimental_relax_shapes=True)
    def _learn_from_experience_batch(self, obs, obs_next, actions, old_policies,
                                     rewards, first_obs, first_actions, dones, lengths):
        batches_indices = tf.RaggedTensor.from_row_lengths(values=tf.range(tf.reduce_sum(lengths)), row_lengths=lengths)
        values = tf.squeeze(self._critic.value(obs))
        values_next = tf.squeeze(self._critic.value(obs_next)) * (1.0 - tf.cast(dones, tf.dtypes.float32))
        policies_weights = tf.squeeze(self._critic_v2(obs))
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

        policies_ratio_product_batches = tf.ragged.boolean_mask(
            policies_ratio_product,
            batch_mask
        )

        gamma_coeffs_batches = tf.ones_like(policies_ratio_batches).to_tensor() * self._gamma
        gamma_coeffs = tf.ragged.boolean_mask(
            tf.math.cumprod(gamma_coeffs_batches, axis=1, exclusive=True),
            batch_mask
        ).flat_values

        # flat tensors
        d_coeffs = gamma_coeffs * (rewards + self._gamma * values_next - values) \
            * tf.tanh(policies_ratio_product_batches.flat_values / policies_weights / self._b) * self._b
        # ragged
        d_coeffs_batches = tf.gather_nd(d_coeffs, tf.expand_dims(indices, axis=2))
        # final summation over original batches
        d1 = tf.stop_gradient(tf.reduce_sum(d_coeffs_batches, axis=1))

        self._backward_pass(first_obs, first_actions, d1, tf.stop_gradient(policies), old_policies, obs)

        _, new_log_policies = tf.split(self._actor.prob(obs, actions), 2, axis=0)
        new_log_policies = tf.squeeze(new_log_policies)
        approx_kl = tf.reduce_mean(policies - new_log_policies)
        with tf.name_scope('actor'):
            tf.summary.scalar('sample_approx_kl_divergence', approx_kl, self._tf_time_step)

    def _backward_pass(self, observations: tf.Tensor, actions: tf.Tensor,
                       d1: tf.Tensor, policies: tf.Tensor, old_policies: tf.Tensor, observations_v2: tf.Tensor):
        """Performs backward pass in BaseActor's and Critic's networks

        Args:
            observations: batch [batch_size, observations_dim] of observations vectors
            actions: batch [batch_size, actions_dim] of actions vectors
            d1: batch [batch_size, observations_dim] of gradient update coefficients (v1)
            policies: batch [batch_size, observations_dim] current policy function values
            old_policies: batch [batch_size, observations_dim] policy function values from the buffer
            observations_v2: batch [batch_size, observations_dim] of observations vectors to for V2 update
        """
        with tf.GradientTape() as tape:
            loss = self._actor.loss(observations, actions, d1)
        grads = tape.gradient(loss, self._actor.trainable_variables)
        gradients = zip(grads, self._actor.trainable_variables)

        self._actor_optimizer.apply_gradients(gradients)

        with tf.GradientTape() as tape:
            loss = self._critic.loss(observations, d1)
        grads = tape.gradient(loss, self._critic.trainable_variables)
        gradients = zip(grads, self._critic.trainable_variables)

        self._critic_optimizer.apply_gradients(gradients)

        with tf.GradientTape() as tape:
            loss = self._critic_v2.loss_v2(observations_v2, policies, old_policies)
        grads = tape.gradient(loss, self._critic_v2.trainable_variables)
        gradients = zip(grads, self._critic_v2.trainable_variables)
        self._critic_v2_optimizer.apply_gradients(gradients)

    def _fetch_offline_batch(self) -> List[Dict[str, Union[np.array, list]]]:
        trajectory_lens = [np.random.geometric(1 - self._lam) + 1 for _ in range(self._num_parallel_envs)]
        batch = []
        [batch.extend(self._memory.get(trajectory_lens)) for _ in range(self._batches_per_env)]
        return batch

    def save(self, path: Path, **kwargs):
        super().save(path, **kwargs)
        critic_v1_path = str(path / 'critic_v1.tf')
        self._critic.save_weights(critic_v1_path, overwrite=True)