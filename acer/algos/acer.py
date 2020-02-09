"""
Actor-Critic with Experience Replay algorithm.
Implements the algorithm from:

(1)
Wawrzyński P, Tanwani AK. Autonomous reinforcement learning with experience replay.
Neural Networks : the Official Journal of the International Neural Network Society.
2013 May;41:156-167. DOI: 10.1016/j.neunet.2012.11.007.

(2)
Wawrzyński, Paweł. "Real-time reinforcement learning by sequential actor–critics
and experience replay." Neural Networks 22.10 (2009): 1484-1497.
"""


from abc import ABC, abstractmethod
from typing import Optional, List, Tuple, Union, Dict

import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

from algos.base import Agent
from environment import BaseMultiEnv
from utils import flatten_experience, unflatten_batches, normc_initializer, RunningMeanVariance
from replay_buffer import MultiReplayBuffer


class Critic(tf.keras.Model):

    def __init__(self, observations_dim: int, layers: Optional[List[int]], tf_time_step: tf.Variable, *args, **kwargs):
        """Value function approximation as MLP network neural network.

        Args:
            observations_dim: dimension of observations space
            layers: list of hidden layers sizes, eg: for neural network with two layers with 10 and 20 hidden units
                pass: [10, 20]
            tf_time_step: time step as TensorFlow variable, required for TensorBoard summaries
        """
        super().__init__(*args, **kwargs)
        self.hidden_1 = tf.keras.layers.Dense(observations_dim, activation='tanh', kernel_initializer=normc_initializer())
        self.hidden_body = [tf.keras.layers.Dense(
            units, activation='tanh', kernel_initializer=normc_initializer()
        ) for units in layers]
        self.hidden_value = tf.keras.layers.Dense(1, kernel_initializer=normc_initializer())
        self._tf_time_step = tf_time_step

    # @tf.function
    def call(self, observations: tf.Tensor,  **kwargs) -> tf.Tensor:
        """Calculates value function given observations batch

        Args:
            observations: batch [batch_size, observations_dim] of observations vectors

        Returns:
            Tensor [batch_size, 1] with value function estimations

        """
        x = self.hidden_1(observations)
        for layer in self.hidden_body:
            x = layer(x)

        value = self.hidden_value(x)

        with tf.name_scope('critic'):
            tf.summary.scalar('batch_value_mean', tf.reduce_mean(value), step=self._tf_time_step)

        return value

    def loss(self, observations: np.array, z: np.array) -> tf.Tensor:
        """Computes Critic's loss.

        Args:
            observations: batch [batch_size, observations_dim] of observations vectors
            z: batch [batch_size, 1] of gradient update coefficient (summation term in the Equation (9)) from
                the paper (1))
        """

        loss = tf.reduce_mean(-tf.math.multiply(self.call(observations), z))

        with tf.name_scope('critic'):
            tf.summary.scalar('batch_loss', loss, step=self._tf_time_step)
        return loss


class Actor(ABC, tf.keras.Model):

    def __init__(self, observations_dim: int, actions_dim: int, layers: Optional[List[int]],
                 beta_penalty: float, tf_time_step: tf.Variable, *args, **kwargs):
        """Policy function as MLP neural network.

        Args:
            observations_dim: dimension of observations space
            layers: list of hidden layers sizes, eg: for neural network with two layers with 10 and 20 hidden units
                pass: [10, 20]
            beta_penalty: penalty coefficient. In discrete case, Actor is penalized for too executing too
                confident actions (no exploration), in the continuous case it is penalized for making actions
                that are out of allowed bounds
            tf_time_step: time step as TensorFlow variable, required for TensorBoard summaries
        """
        super().__init__(*args, **kwargs)

        self.hidden_1 = tf.keras.layers.Dense(
            observations_dim, activation='tanh', kernel_initializer=normc_initializer()
        )

        self.hidden_body = [tf.keras.layers.Dense(
            units, activation='tanh', kernel_initializer=normc_initializer()
        ) for units in layers]
        self.hidden_logits = tf.keras.layers.Dense(actions_dim, kernel_initializer=normc_initializer())

        self._actions_dim = actions_dim
        self._beta_penalty = beta_penalty
        self._tf_time_step = tf_time_step

    @property
    @abstractmethod
    def action_dtype(self):
        """Returns data type of the Actor's actions"""

    @abstractmethod
    def loss(self, observations: np.array, actions: np.array, z: np.array) -> tf.Tensor:
        """Computes Actor's loss

        Args:
            observations: batch [batch_size, observations_dim] of observations vectors
            actions: batch [batch_size, actions_dim] of actions vectors
            z: batch [batch_size, 1] of gradient update coefficient (summation term in the Equation (8)) from
                the paper (1))
        """

    @abstractmethod
    def prob(self, observations: np.array, actions: np.array) -> tf.Tensor:
        """Computes probabilities (or probability densities in continuous case) of executing passed actions

        Args:
            observations: batch [batch_size, observations_dim] of observations vectors
            actions: batch [batch_size, actions_dim] of actions vectors

        Returns:
             Tensor [batch_size, actions_dim, 1] with computed probabilities (densities)
        """

    @abstractmethod
    def call(self, observations: np.array, **kwargs) -> Tuple[tf.Tensor, tf.Tensor]:
        """Samples actions and computes their probabilities (or probability densities in continuous case)

        Args:
            observations: batch [batch_size, observations_dim] of observations vectors

        Returns:
            tuple with two Tensors:
                * actions [batch_size, actions_dim]
                * probabilities/densities [batch_size, 1]
        """

    @abstractmethod
    def act_deterministic(self, observations: np.array, **kwargs) -> tf.Tensor:
        """Samples actions without exploration noise.

        Args:
            observations: batch [batch_size, observations_dim] of observations vectors

        Returns:
            Tensor of actions [batch_size, actions_dim]
        """


class CategoricalActor(Actor):

    def __init__(self, observations_dim: int, actions_dim: int, layers: Optional[List[int]], *args, **kwargs):
        """Actor for discrete actions spaces. Uses Categorical Distribution"""
        super().__init__(observations_dim, actions_dim, layers, *args, **kwargs)

    def _call_logits(self, observations: tf.Tensor) -> tf.Tensor:
        x = self.hidden_1(observations)
        for layer in self.hidden_body:
            x = layer(x)
        return self.hidden_logits(x)

    @property
    def action_dtype(self):
        return tf.dtypes.int32

    def loss(self, observations: tf.Tensor, actions: tf.Tensor, z: tf.Tensor) -> tf.Tensor:
        logits = self._call_logits(observations)

        # TODO: remove hardcoded '10' and '20'
        logits_div = tf.divide(logits, 10)
        log_probs = tf.nn.log_softmax(logits_div)
        action_log_probs = tf.expand_dims(
            tf.gather_nd(log_probs, tf.expand_dims(actions, axis=1), batch_dims=1),
            axis=1
        )

        # penalty for making actions out of the allowed bounds
        penalty = tf.reduce_sum(
            tf.scalar_mul(
                self._beta_penalty,
                tf.square(tf.maximum(0.0, tf.abs(logits) - 20))
            ),
            axis=1,
            keepdims=True
        )
        total_loss = tf.reduce_mean(-tf.math.multiply(action_log_probs, z) + penalty)

        # entropy maximization penalty
        # penalty = self._beta_penalty * (-tf.reduce_sum(tf.math.multiply(probs, log_probs), axis=1))

        with tf.name_scope('actor'):
            tf.summary.scalar('batch_loss', total_loss, step=self._tf_time_step)
            tf.summary.scalar('batch_penalty_mean', tf.reduce_mean(penalty), step=self._tf_time_step)

        return total_loss

    # @tf.function
    def prob(self, observations: tf.Tensor, actions: tf.Tensor) -> tf.Tensor:
        # TODO: remove hardcoded '10' and '20'
        logits = tf.divide(self._call_logits(observations), 10)
        probs = tf.nn.softmax(logits)
        action_probs = tf.gather_nd(probs, tf.expand_dims(actions, axis=1), batch_dims=1)
        return action_probs

    # @tf.function
    def call(self, observations: tf.Tensor, **kwargs) -> Tuple[tf.Tensor, tf.Tensor]:

        # TODO: remove hardcoded '10' and '20'
        logits = tf.divide(self._call_logits(observations), 10)
        probs = tf.nn.softmax(logits)
        log_probs = tf.nn.log_softmax(logits)

        actions = tf.random.categorical(log_probs, num_samples=1, dtype=tf.dtypes.int32)
        actions_probs = tf.gather_nd(probs, actions, batch_dims=1)

        with tf.name_scope('actor'):
            # TODO: refactor
            tf.summary.histogram('action', actions, step=self._tf_time_step)

        return tf.squeeze(actions, axis=[1]), actions_probs

    def act_deterministic(self, observations: np.array, **kwargs) -> tf.Tensor:
        """Performs most probable action"""
        logits = self._call_logits(observations)

        actions = tf.argmax(logits, axis=1)
        return actions


class GaussianActor(Actor):

    def __init__(self, observations_dim: int, actions_dim: int, layers: Optional[List[int]], beta_penalty: float,
                 actions_bound: float, *args, **kwargs):
        """Actor for continuous actions space. Uses MultiVariate Gaussian Distribution as policy distribution.

        TODO: introduce [a, b] intervals as allowed actions bounds

        Args:
            observations_dim: dimension of observations space
            layers: list of hidden layer sizes
            beta_penalty: penalty for too confident actions coefficient
            actions_bound: upper (lower == '-actions_bound') bound for allowed actions,
             required in case of continuous actions
        """
        super().__init__(observations_dim, actions_dim, layers, beta_penalty, *args, **kwargs)

        self._actions_bound = actions_bound

        # change constant to Variable to make std a learned parameter
        self._log_std = tf.constant(
            tf.ones(shape=(actions_dim, )) * tf.math.log(0.25 * actions_bound),
            name="actor_std"
        )

    @property
    def action_dtype(self):
        return tf.dtypes.float32

    def loss(self, observations: np.array, actions: np.array, z: np.array) -> tf.Tensor:
        mean = self._call_mean(observations)
        dist = tfp.distributions.MultivariateNormalDiag(
            loc=mean,
            scale_diag=tf.exp(self._log_std)
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

        total_loss = tf.reduce_mean(-tf.math.multiply(action_log_probs, z) + bounds_penalty)

        with tf.name_scope('actor'):
            for i in range(self._actions_dim):
                tf.summary.scalar(f'std_{i}', tf.exp(self._log_std[i]), step=self._tf_time_step)
            tf.summary.scalar('batch_loss', total_loss, step=self._tf_time_step)
            tf.summary.scalar('batch_bounds_penalty_mean', tf.reduce_mean(bounds_penalty), step=self._tf_time_step)
            tf.summary.scalar('batch_entropy_mean', tf.reduce_mean(entropy), step=self._tf_time_step)

        return total_loss

    def _call_mean(self, observations: np.array) -> tf.Tensor:
        x = self.hidden_1(observations)
        for layer in self.hidden_body:
            x = layer(x)
        return self.hidden_logits(x)

    # @tf.function
    def prob(self, observations: tf.Tensor, actions: tf.Tensor) -> tf.Tensor:
        mean = self._call_mean(observations)
        dist = tfp.distributions.MultivariateNormalDiag(
            loc=mean,
            scale_diag=tf.exp(self._log_std)
        )

        return dist.prob(actions)

    # @tf.function
    def call(self, observations: tf.Tensor, **kwargs) -> Tuple[tf.Tensor, tf.Tensor]:
        mean = self._call_mean(observations)

        dist = tfp.distributions.MultivariateNormalDiag(
            loc=mean,
            scale_diag=tf.exp(self._log_std)
        )

        actions = dist.sample(dtype=tf.dtypes.float32)
        actions_probs = dist.prob(actions)

        with tf.name_scope('actor'):
            for i in range(self._actions_dim):
                tf.summary.scalar(f'batch_action_{i}_mean', tf.reduce_mean(actions[:, i]), step=self._tf_time_step)

        return actions, actions_probs

    def act_deterministic(self, observations: np.array, **kwargs) -> tf.Tensor:
        """Returns mean of the Gaussian"""
        mean = self._call_mean(observations)
        return mean


class ACER(Agent):
    def __init__(self, observations_dim: int, actions_dim: int, actor_layers: List[int], critic_layers: List[int],
                 num_parallel_envs: int, is_discrete: bool, gamma: int, memory_size: int, alpha: float, p: float,
                 b: float, c: int, c0: float, actor_lr: float, actor_beta_penalty: float,
                 actor_adam_beta1: float, actor_adam_beta2: float, actor_adam_epsilon: float, critic_lr: float,
                 critic_adam_beta1: float, critic_adam_beta2: float, critic_adam_epsilon: float,
                 actions_bound: Optional[float], standardize_obs: bool = True):
        """Actor-Critic with Experience Replay

        TODO: finish docstrings
        TODO: refactor converting to tensor
        """

        assert is_discrete or actions_bound is not None, "For continuous actions, " \
                                                         "'actions_bound' argument should be specified"

        self._tf_time_step = tf.Variable(name='tf_time_step', initial_value=1, dtype=tf.dtypes.int64)

        self._critic = Critic(observations_dim, critic_layers, self._tf_time_step)

        if is_discrete:
            self._actor = CategoricalActor(observations_dim, actions_dim, actor_layers,
                                           actor_beta_penalty, self._tf_time_step)
        else:
            self._actor = GaussianActor(observations_dim, actions_dim, actor_layers,
                                        actor_beta_penalty, actions_bound, self._tf_time_step)

        self._memory = MultiReplayBuffer(max_size=memory_size, num_buffers=num_parallel_envs)
        self._p = p
        self._alpha = alpha
        self._b = b
        self._gamma = gamma
        self._time_step = 0
        self._c = c
        self._c0 = c0
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

        if standardize_obs:
            self._running_mean_obs = RunningMeanVariance(shape=(observations_dim, ))
        else:
            self._running_mean_obs = None

    def save_experience(self, steps: List[
        Tuple[Union[int, float, list], np.array, float, np.array, np.array, bool, bool]
    ]):
        """Stores gathered experiences in a replay buffer. Accepts list of steps.

        Args:
            steps: List of steps, see ReplayBuffer.put() for a detailed format description
        """
        self._time_step += len(steps)
        self._tf_time_step.assign_add(len(steps))
        self._memory.put(steps)
        if self._running_mean_obs:
            for step in steps:
                self._running_mean_obs.update(step[1])

            with tf.name_scope('acer'):
                for i in range(self._running_mean_obs.mean.shape[0]):
                    tf.summary.scalar(
                        f'obs_{i}_running_mean', self._running_mean_obs.mean[i], step=self._tf_time_step
                    )
                    tf.summary.scalar(
                        f'obs_{i}_running_std', np.sqrt(self._running_mean_obs.var[i]), step=self._tf_time_step
                    )

    def reset(self):
        """Resets environments and neural network weights"""
        return NotImplementedError

    def predict_action(self, observations: np.array, is_deterministic: bool = False) \
            -> Tuple[np.array, Optional[np.array]]:
        """Predicts actions for given observations. Performs forward pass with Actor network.

        Args:
            observations: batch [batch_size, observations_dim] of observations vectors
            is_deterministic: True if mean actions (without exploration noise) should be returned

        Returns:
            Tuple of sampled actions and corresponding probabilities (probability densities) if action was sampled
                from the distribution, None otherwise
        """
        processed_obs = self._standardize_observations_if_turned_on(observations)
        if is_deterministic:
            return self._actor.act_deterministic(processed_obs).numpy(), None
        else:
            actions, policies = self._actor(tf.convert_to_tensor(processed_obs, dtype=tf.dtypes.float32))
            return actions.numpy(), policies.numpy()

    def learn(self):
        """
        Performs experience replay learning. Experience trajectory is sampled from every replay buffer once, thus
        single backwards pass batch consists of 'num_parallel_envs' trajectories.

        Every call executes N of backwards passes, where: N = min(c0 * time_step / num_parallel_envs, c).
        That means at the beginning experience replay intensity increases linearly with number of samples
        collected till c value is reached.
        """
        experience_replay_iterations = min([round(self._c0 * self._time_step / self._num_parallel_envs), self._c])

        for _ in range(experience_replay_iterations):
            offline_batch = self._fetch_offline_batch()
            self._learn_from_experience_batch(offline_batch)

    def _learn_from_experience_batch(self, experience_batches: List[Dict[str, Union[np.array, list]]]):
        """Backward pass with single batch of experience.

        See Equation (8) and Equation (9) in the paper (1).
        """

        policies_batches, values_batches, values_next_batches = self._get_processed_experience_batches(
            experience_batches
        )

        z = []
        # summation from the Equations (8) and (9) in the paper (1)
        for buffer_batch, policies, values, values_next in zip(experience_batches, policies_batches,
                                                               values_batches, values_next_batches):

            old_policies = buffer_batch['policies']
            densities = self._compute_truncated_ratios(policies, old_policies)

            z_t = 0

            for i in range(len(old_policies)):
                reward = buffer_batch['rewards'][i]
                next_state_value = values_next[i]
                value = values[i]
                coeff = (self._alpha * (1 - self._p)) ** i
                if buffer_batch['dones'][i]:
                    z_t += coeff * (reward - value) * densities[i]
                else:
                    z_t += coeff * (reward + next_state_value - value) * densities[i]
            z.append(z_t)

        observations = tf.convert_to_tensor(
            np.array([batch['observations'][0] for batch in experience_batches]),
            dtype=tf.dtypes.float32
        )
        observations = self._standardize_observations_if_turned_on(observations)
        actions = tf.convert_to_tensor(
            np.array([batch['actions'][0] for batch in experience_batches]),
            dtype=self._actor.action_dtype
        )

        self._backward_pass(observations, actions, tf.convert_to_tensor(z))

    def _get_processed_experience_batches(
            self, experience_batches: List[Dict[str, Union[np.array, list]]]
    ) -> Tuple[np.array, np.array, np.array]:
        """Computes value approximation and policy for samples from the replay buffers.

        experience_batches: batches [num_parallel_envs, trajectory_length] of experience from the buffers.
            each batch origins from different replay buffer. See ReplayBuffer specification for detailed
            experience batch description.

        Returns:
            Tuple with matrices:
                * batch [batch_size, 1] of policy function values (probabilities or probability densities)
                * batch [batch_size, 1] of value function approximation for "current" observations
                * batch [batch_size, 1] of value function approximation for "next" observations
        """

        observations_flatten, next_observations_flatten, actions_flatten = flatten_experience(experience_batches)
        observations_flatten = self._standardize_observations_if_turned_on(observations_flatten)
        next_observations_flatten = self._standardize_observations_if_turned_on(next_observations_flatten)
        # concatenate here to perform one single batch calculation
        values_flatten = self._critic(
            tf.convert_to_tensor(
                np.concatenate([observations_flatten, next_observations_flatten], axis=0),
                dtype=tf.dtypes.float32
            )
        )
        policies_flatten = self._actor.prob(
            tf.convert_to_tensor(observations_flatten, dtype=tf.dtypes.float32),
            tf.convert_to_tensor(actions_flatten, dtype=self._actor.action_dtype)
        ).numpy()
        values_next_flatten = values_flatten[len(observations_flatten):].numpy() * self._gamma
        values_flatten = values_flatten[:len(observations_flatten)].numpy()

        # back to the initial format
        policies_batches, values_batches, values_next_batches = unflatten_batches(
            values_flatten, values_next_flatten, policies_flatten, experience_batches
        )
        return policies_batches, values_batches, values_next_batches

    @tf.function
    def _backward_pass(self, observations: tf.Tensor, actions: tf.Tensor, z: tf.Tensor):
        """Performs backward pass in Actor's and Critic's networks

        TODO: computations bellow can be optimized, probably

        Args:
            observations: batch [batch_size, observations_dim] of observations vectors
            actions: batch [batch_size, actions_dim] of actions vectors
            z: batch [batch_size, observations_dim] of gradient update coefficient
                (summation terms in the Equations (8) and (9) from the paper (1))
        """
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

    def _compute_truncated_ratios(self, policies: np.array, old_policies: np.array) -> List[float]:
        """Computes truncated probability ratios (probability densities ratios) for the summation terms
        in Equations (8) and (9) from the paper (1)."""
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

    def _standardize_observations_if_turned_on(self, observations: np.array) -> np.array:
        """If standardization is turned on, observations are being standardized with running mean and variance."""
        if self._running_mean_obs:
            return (observations - self._running_mean_obs.mean) / np.sqrt(self._running_mean_obs.var)
        else:
            return observations













