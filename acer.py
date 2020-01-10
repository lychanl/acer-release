from abc import ABC, abstractmethod
from typing import Optional, List, Tuple, Union, Dict

import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

from utils import flatten_experience, unflatten_batches
from replay_buffer import MultiReplayBuffer


class Critic(tf.keras.Model):

    def __init__(self, observations_dim: int, layers: Optional[List[int]], *args, **kwargs):
        """Value function approximation MLP network.

        Args:
            observations_dim: dimension of observations space
            layers: list of hidden layers sizes
        """
        super().__init__(*args, **kwargs)
        self.hidden_1 = tf.keras.layers.Dense(observations_dim, activation='tanh')
        self.hidden_body = [tf.keras.layers.Dense(units, activation='tanh') for units in layers]
        self.hidden_value = tf.keras.layers.Dense(1)

    @tf.function(input_signature=[tf.TensorSpec(shape=(None, 3))])
    def call(self, observations: np.array, **kwargs) -> tf.Tensor:
        """Calculates value function given observations batch

        Args:
            observations: batch (batch_size, observations_dim) of observations vectors

        Returns:
            Tensor (batch_size, 1) witch value function values

        """
        x = self.hidden_1(observations)
        for layer in self.hidden_body:
            x = layer(x)

        value = self.hidden_value(x)
        return value

    def loss(self, observations: np.array, z: np.array) -> tf.Tensor:
        """Computes loss function

        Args:
            observations: batch (batch_size, observations_dim) of observations vectors
            z: batch (batch_size, 1) of 'z' (gradient update multiplier) vectors
        """
        return tf.reduce_mean(-tf.math.multiply(self.call(observations), z))


class Actor(ABC, tf.keras.Model):

    def __init__(self, observations_dim: int, actions_dim: int, layers: Optional[List[int]],
                 beta_penalty: float, *args, **kwargs):
        """Policy function MLP network

        Args:
            observations_dim: dimension of observations space
            layers: list of hidden layer sizes
            beta_penalty: penalty for too confident actions coefficient
        """
        super().__init__(*args, **kwargs)
        self.hidden_1 = tf.keras.layers.Dense(observations_dim, activation='tanh')
        self.hidden_body = [tf.keras.layers.Dense(units, activation='tanh') for units in layers]
        self.hidden_logits = tf.keras.layers.Dense(actions_dim)

        self._actions_dim = actions_dim
        self._beta_penalty = beta_penalty

    @abstractmethod
    def loss(self, observations: np.array, actions: np.array, z: np.array) -> tf.Tensor:
        """Computes loss function

        Args:
            observations: batch (batch_size, observations_dim) of observations vectors
            actions: batch (batch_size, actions_dim) of actions vectors
            z: batch (batch_size, 1) of 'z' (gradient update multiplier) vectors
        """

    @abstractmethod
    def prob(self, observations: np.array, actions: np.array) -> tf.Tensor:
        """Computes probabilities (densities) of performing action given observations vector

        Args:
            observations: batch (batch_size, observations_dim)  of observations vectors
            actions: batch (batch_size, actions_dim) of actions vectors

        Returns:
             Tensor (batch_size, actions_dim, 1) with computed probabilities
        """

    @abstractmethod
    def log_prob(self, observations: np.array, actions: np.array) -> tf.Tensor:
        """Computes logarithm of probabilities (densities) of performing action given observations vector

        Args:
            observations: batch (batch_size, observations_dim)  of observations vectors
            actions: batch (batch_size, actions_dim)  of actions vectors

        Returns:
             Tensor (batch_size, actions_dim, 1) with computed log probabilities
        """

    @abstractmethod
    def call(self, observations: np.array, **kwargs) -> Tuple[tf.Tensor, tf.Tensor]:
        """Given observations vectors batch, computes actions probabilities (densities) and
        samples actions.

        Args:
            observations: observations vectors batch

        Returns:
            tuple with two Tensors:
                * actions (batch_size, actions_dim)
                * probabilities/densities (batch_size, 1)
        """


class CategoricalActor(Actor):

    def __init__(self, observations_dim: int, actions_dim: int, layers: Optional[List[int]], *args, **kwargs):
        """Actor for discrete actions space. Uses Categorical Distribution"""
        super().__init__(observations_dim, actions_dim, layers, *args, **kwargs)

    def _call_logits(self, observations: np.array) -> tf.Tensor:
        x = self.hidden_1(observations)
        for layer in self.hidden_body:
            x = layer(x)
        return self.hidden_logits(x)

    def loss(self, observations: np.array, actions: np.array, z: np.array):
        logits = self._call_logits(observations)

        # TODO: make '10' hyperparameter
        logits_div = tf.divide(logits, 10)
        log_probs = tf.nn.log_softmax(logits_div)
        action_log_probs = tf.gather_nd(log_probs, tf.expand_dims(actions, axis=1), batch_dims=1)

        # penalty for making actions out of the allowed bounds
        penalty = tf.reduce_sum(
            tf.scalar_mul(
                self._beta_penalty,
                tf.square(tf.maximum([0.0] * self._actions_dim, tf.abs(logits) - 20))
            ),
            axis=1
        )

        # entropy maximization penalty
        # penalty = self._beta_penalty * (-tf.reduce_sum(tf.math.multiply(probs, log_probs), axis=1))
        return tf.reduce_mean(-tf.math.multiply(action_log_probs, z) + penalty)

    @tf.function(input_signature=[tf.TensorSpec(shape=(None, 3)), tf.TensorSpec(shape=(None, 1))])
    def prob(self, observations: np.array, actions: np.array) -> tf.Tensor:
        logits = tf.divide(self._call_logits(observations), 10)
        probs = tf.nn.softmax(logits)
        action_probs = tf.gather_nd(probs, tf.expand_dims(actions, axis=1), batch_dims=1)
        return action_probs

    @tf.function(input_signature=[tf.TensorSpec(shape=(None, 3)), tf.TensorSpec(shape=(None, 1))])
    def log_prob(self, observations: np.array, actions: np.array) -> tf.Tensor:
        logits = tf.divide(self._call_logits(observations), 10)
        log_probs = tf.nn.log_softmax(logits)
        action_log_probs = tf.gather_nd(log_probs, tf.expand_dims(actions, axis=1), batch_dims=1)
        return action_log_probs

    @tf.function(input_signature=[tf.TensorSpec(shape=(None, 3))])
    def call(self, observations: np.array, **kwargs) -> Tuple[tf.Tensor, tf.Tensor]:
        # TODO: make '10' hyperparameter
        logits = tf.divide(self._call_logits(observations), 10)
        probs = tf.nn.softmax(logits)
        log_probs = tf.nn.log_softmax(logits)

        action = tf.random.categorical(log_probs, num_samples=1, dtype=tf.dtypes.int32)
        action_prob = tf.gather_nd(probs, action, batch_dims=1)

        return tf.squeeze(action, axis=[1]), action_prob


class GaussianActor(Actor):

    def __init__(self, observations_dim: int, actions_dim: int, layers: Optional[List[int]], beta_penalty: float,
                 actions_bound: float, std: List[float], *args, **kwargs):
        """Actor for continuous actions space. Uses MultiVariate Gaussian Distribution as policy distribution.

        TODO: introduce [a, b] intervals to allowed actions bounds
        TODO: std as learned parameter

        Args:
            observations_dim: dimension of observations space
            layers: list of hidden layer sizes
            beta_penalty: penalty for too confident actions coefficient
            actions_bound: upper (lower == '-actions_bound') bound for allowed actions,
             required in case of continuous actions
        """
        super().__init__(observations_dim, actions_dim, layers, beta_penalty, *args, **kwargs)
        self._actions_bound = actions_bound
        self._std = std

    def loss(self, observations: np.array, actions: np.array, z: np.array) -> tf.Tensor:
        mean = self._call_mean(observations)
        dist = tfp.distributions.MultivariateNormalDiag(
            loc=mean,
            scale_diag=self._std)

        action_log_probs = dist.log_prob(actions)

        penalty = tf.reduce_sum(
            tf.scalar_mul(
                self._beta_penalty,
                tf.square(tf.maximum([0.0] * self._actions_dim, tf.abs(mean) - self._actions_bound))
            )
        )

        return tf.reduce_mean(-tf.math.multiply(action_log_probs, z) + penalty)

    def _call_mean(self, observations: np.array) -> tf.Tensor:
        x = self.hidden_1(observations)
        for layer in self.hidden_body:
            x = layer(x)
        return self.hidden_logits(x)

    @tf.function(input_signature=[tf.TensorSpec(shape=(None, 3)), tf.TensorSpec(shape=(None, 1))])
    def prob(self, observations: np.array, actions: np.array) -> tf.Tensor:
        mean = self._call_mean(observations)
        dist = tfp.distributions.MultivariateNormalDiag(
            loc=mean,
            scale_diag=[self._std]
        )

        return dist.prob(actions)

    @tf.function(input_signature=[tf.TensorSpec(shape=(None, 3)), tf.TensorSpec(shape=(None, 1))])
    def log_prob(self, observations: np.array, actions: np.array) -> tf.Tensor:
        mean = self._call_mean(observations)
        dist = tfp.distributions.MultivariateNormalDiag(
            loc=mean,
            scale_diag=[self._std]
        )

        return dist.log_prob(actions)

    @tf.function(input_signature=[tf.TensorSpec(shape=(None, 3))])
    def call(self, observations: np.array, **kwargs) -> Tuple[tf.Tensor, tf.Tensor]:
        mean = self._call_mean(observations)

        dist = tfp.distributions.MultivariateNormalDiag(
            loc=mean,
            scale_diag=[self._std]
        )

        actions = dist.sample(dtype=tf.dtypes.float32)
        actions_probs = dist.prob(actions)

        return actions, actions_probs


class ACER:
    def __init__(self, observations_dim: int, actions_dim: int, actor_layers: List[int],
                 critic_layers: List[int], num_parallel_envs: int, is_discrete: bool, gamma: int,
                 memory_size: int, alpha: float, p: float, b: float, c: int, min_steps_learn: int,
                 std: Optional[List[float]], actor_lr: float, actor_beta_penalty: float, actor_adam_beta1: float,
                 actor_adam_beta2: float, actor_adam_epsilon: float, critic_lr: float, critic_adam_beta1: float,
                 critic_adam_beta2: float, critic_adam_epsilon: float, actions_bound: Optional[float]):
        """Actor-Critic with experience replay algorithm.
        See http://lasa.epfl.ch/publications/uploadedFiles/Waw13.pdf

        TODO: finish docstring (arguments description)
        TODO: refactor converting to tensor
        """

        assert is_discrete or actions_bound, "For continuous actions, 'actions_bound' argument should be specified"
        assert is_discrete or std, "For continuous actions, 'std' argument should be specified"

        self._critic = Critic(observations_dim, critic_layers)

        if is_discrete:
            self._actor = CategoricalActor(observations_dim, actions_dim, actor_layers, actor_beta_penalty)
        else:
            self._actor = GaussianActor(observations_dim, actions_dim, actor_layers,
                                        actor_beta_penalty, actions_bound, std)

        self._memory = MultiReplayBuffer(max_size=memory_size, num_buffers=num_parallel_envs)
        self._p = p
        self._alpha = alpha
        self._b = b
        self._gamma = gamma
        self._timestep = 0
        self._c = c
        self._min_steps_learn = min_steps_learn
        self._num_parallel_envs = num_parallel_envs

        self._actor_optimizer = tf.keras.optimizers.Adam(
            lr=actor_lr,
            beta_1=actor_adam_beta1,
            beta_2=actor_adam_beta2,
            epsilon=actor_adam_epsilon
        )

        self._critic_optimizer = tf.keras.optimizers.Adam(
            lr=critic_lr,
            beta_1=critic_adam_beta1,
            beta_2=critic_adam_beta2,
            epsilon=critic_adam_epsilon
        )

    def save_experience(self, steps: List[
        Tuple[Union[int, float, list], np.array, float, np.array, np.array, bool, bool]
    ]):
        """Stores gathered experiences in the replay buffer. Accepts list of steps.

        Args:
            steps: List of steps, see ReplayBuffer.put() for the detailed format description
        """
        self._timestep += len(steps)
        self._memory.put(steps)

    def reset(self):
        """Resets environments and neural network weights"""
        return NotImplementedError

    def predict_action(self, observations: np.array) -> Tuple[list, np.array]:
        """Predicts actions given observations. Performs forward pass with Actor network.

        Args:
            observations: observations vectors batch

        Returns:
            Tuple of sampled actions and corresponding policy function values
        """
        actions, policies = self._actor(tf.convert_to_tensor(observations, dtype=tf.dtypes.float32))
        return actions.numpy(), policies.numpy()

    def learn(self):
        """Performs experience replay learning. Experience trajectory is sampled from every replay buffer, thus
        resulting in '_num_parallel_envs' of trajectories in total."""
        if self._timestep < self._min_steps_learn:
            return

        for _ in range(self._c):
            offline_batch = self._fetch_offline_batch()
            self._learn_from_experience_batch(offline_batch)

    def _learn_from_experience_batch(self, buffers_batches: List[Dict[str, Union[np.array, list]]]):
        """

        Args:
            buffers_batches:

        Returns:

        """
        observations_flatten, next_observations_flatten, actions_flatten = flatten_experience(buffers_batches)
        # concatenate here to perform one single batch calculation
        values_flatten = self._critic(tf.convert_to_tensor(np.concatenate([observations_flatten, next_observations_flatten], axis=0), dtype=tf.dtypes.float32))
        policies_flatten = self._actor.prob(tf.convert_to_tensor(observations_flatten, dtype=tf.dtypes.float32), tf.convert_to_tensor(actions_flatten, dtype=tf.dtypes.float32)).numpy()
        values_next_flatten = values_flatten[len(observations_flatten):].numpy() * self._gamma
        values_flatten = values_flatten[:len(observations_flatten)].numpy()
        policies_batches, values_batches, values_next_batches = unflatten_batches(
            values_flatten, values_next_flatten, policies_flatten, buffers_batches
        )

        z = []
        for buffer_batch, policies, values, values_next in zip(buffers_batches, policies_batches,
                                                               values_batches, values_next_batches):

            old_policies = buffer_batch['policies']
            densities = self._truncate_densities(policies, old_policies)

            z_t = 0

            for i in range(len(old_policies)):
                reward = buffer_batch['rewards'][i]
                next_state_future_reward = values_next[i]
                future_reward = values[i]
                if buffer_batch['dones'][i]:
                    z_t += (((self._alpha * (1 - self._p)) ** i) * (reward - future_reward) * densities[i])
                else:
                    z_t += (self._alpha * (1 - self._p)) ** i \
                             * (reward + next_state_future_reward - future_reward) \
                             * densities[i]
            z.append(z_t)
        observations = tf.convert_to_tensor(np.array([batch['observations'][0] for batch in buffers_batches]), dtype=tf.dtypes.float32)
        actions = tf.convert_to_tensor(np.array([batch['actions'][0] for batch in buffers_batches]), dtype=tf.dtypes.float32)
        self._apply_gradients(observations, actions, tf.convert_to_tensor(z))

    @tf.function(input_signature=[tf.TensorSpec(shape=(None, 3)), tf.TensorSpec(shape=(None, 1)), tf.TensorSpec(shape=(None, 1))])
    def _apply_gradients(self, observations, actions, z):
        with tf.GradientTape() as tape:
            loss = self._actor.loss(observations, actions, z)
        grads = tape.gradient(loss, self._actor.trainable_variables)
        gradients = zip(grads, self._actor.trainable_variables)

        self._actor_optimizer.apply_gradients(gradients)

        with tf.GradientTape() as tape:
            loss = self._critic.loss(observations, z)
        grads = tape.gradient(loss, self._critic.trainable_variables)
        gradients = zip(grads, self._critic.trainable_variables)

        self._critic_optimizer.apply_gradients(gradients)

    def _truncate_densities(self, policies, old_policies):
        truncated_densities = []
        current_product = 1
        for policy, old_policy in zip(policies, old_policies):
            density = policy / old_policy
            current_product = current_product * density
            truncated_densities.append(min([current_product, self._b]))

        return truncated_densities

    def _fetch_offline_batch(self) -> List[Dict[str, Union[np.array, list]]]:
        trajectory_lens = [np.random.geometric(self._p) for _ in range(self._num_parallel_envs)]
        return self._memory.get(trajectory_lens)

    def _fetch_online_batch(self) -> List[Dict[str, Union[np.array, list]]]:
        trajectory_lens = [np.random.geometric(self._p) for _ in range(self._num_parallel_envs)]
        return self._memory.get_newest(trajectory_lens)













