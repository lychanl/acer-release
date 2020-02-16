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
# import os
# os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

import utils
from algos.base import Agent
from utils import normc_initializer, RunningMeanVariance
from replay_buffer import MultiReplayBuffer, BufferFieldSpec


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
        self.hidden_1 = tf.keras.layers.Dense(
            observations_dim,
            activation='tanh',
            kernel_initializer=normc_initializer(),
        )
        self.hidden_body = [tf.keras.layers.Dense(
            units, activation='tanh', kernel_initializer=normc_initializer()
        ) for units in layers]
        self.hidden_value = tf.keras.layers.Dense(1, kernel_initializer=normc_initializer())
        self._tf_time_step = tf_time_step

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
        """Returns data type of the Actor's actions (TensorFlow)"""

    @property
    @abstractmethod
    def action_dtype_np(self):
        """Returns data type of the Actor's actions (Numpy)"""

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

    @property
    def action_dtype_np(self):
        return np.int32

    def loss(self, observations: tf.Tensor, actions: tf.Tensor, z: tf.Tensor) -> tf.Tensor:
        logits = self._call_logits(observations)

        # TODO: remove hardcoded '10' and '20'
        logits_div = tf.divide(logits, 10)
        log_probs = tf.nn.log_softmax(logits_div)
        action_log_probs = tf.expand_dims(
            tf.gather_nd(log_probs, actions, batch_dims=1),
            axis=1
        )
        dist = tfp.distributions.Categorical(logits_div)

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
        # entropy = -tf.reduce_sum(tf.math.multiply(probs, log_probs), axis=1)
        # penalty = self._beta_penalty * (-tf.reduce_sum(tf.math.multiply(probs, log_probs), axis=1))

        with tf.name_scope('actor'):
            tf.summary.scalar('batch_entropy_mean', tf.reduce_mean(dist.entropy()), step=self._tf_time_step)
            tf.summary.scalar('batch_loss', total_loss, step=self._tf_time_step)
            tf.summary.scalar('batch_penalty_mean', tf.reduce_mean(penalty), step=self._tf_time_step)

        return total_loss

    def prob(self, observations: tf.Tensor, actions: tf.Tensor) -> tf.Tensor:
        # TODO: remove hardcoded '10' and '20'
        logits = tf.divide(self._call_logits(observations), 10)
        probs = tf.nn.softmax(logits)
        action_probs = tf.gather_nd(probs, actions, batch_dims=1)
        return action_probs

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
        logits = tf.divide(self._call_logits(observations), 10)
        probs = tf.nn.softmax(logits)

        actions = tf.argmax(probs, axis=1)
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
            tf.math.log(0.4 * actions_bound),
            name="actor_std",
        )

    @property
    def action_dtype(self):
        return tf.dtypes.float32

    @property
    def action_dtype_np(self):
        return np.float32

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

    def prob(self, observations: tf.Tensor, actions: tf.Tensor) -> tf.Tensor:
        mean = self._call_mean(observations)
        dist = tfp.distributions.MultivariateNormalDiag(
            loc=mean,
            scale_diag=tf.exp(self._log_std)
        )

        return dist.prob(actions)

    def call(self, observations: tf.Tensor, **kwargs) -> Tuple[tf.Tensor, tf.Tensor]:
        mean = self._call_mean(observations)

        dist = tfp.distributions.MultivariateNormalDiag(
            loc=mean,
            scale_diag=tf.exp(self._log_std)
        )

        actions = dist.sample(dtype=self.dtype)
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
                 actions_bound: Optional[float], standardize_obs: bool = False, rescale_rewards: bool = False,
                 batches_per_env: int = 5):
        """Actor-Critic with Experience Replay

        TODO: finish docstrings
        TODO: refactor converting to tensor
        """

        assert is_discrete or actions_bound is not None, "For continuous actions, " \
                                                         "'actions_bound' argument should be specified"

        self._tf_time_step = tf.Variable(initial_value=1, name='tf_time_step', dtype=tf.dtypes.int64, trainable=False)

        self._critic = Critic(observations_dim, critic_layers, self._tf_time_step)

        if is_discrete:
            self._actor = CategoricalActor(observations_dim, actions_dim, actor_layers,
                                           actor_beta_penalty, self._tf_time_step)
        else:
            self._actor = GaussianActor(observations_dim, actions_dim, actor_layers,
                                        actor_beta_penalty, actions_bound, self._tf_time_step)

        self._init_replay_buffer(actions_dim, observations_dim, is_discrete, memory_size, num_parallel_envs)
        self._p = p
        self._alpha = alpha
        self._b = b
        self._gamma = gamma
        self._batches_per_env = batches_per_env
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

        if rescale_rewards:
            self._running_mean_rewards = RunningMeanVariance(shape=(1, ))
        else:
            self._running_mean_rewards = None

    def _init_replay_buffer(self, actions_dim: int, observations_dim: int, is_discrete: bool,
                            memory_size: int, num_parallel_envs: int):
        if is_discrete:
            action_shape = (1, )
        else:
            action_shape = (actions_dim, )

        self._memory = MultiReplayBuffer(
            action_spec=BufferFieldSpec(shape=action_shape, dtype=self._actor.action_dtype_np),
            obs_spec=BufferFieldSpec(shape=(observations_dim,), dtype=np.float32),
            max_size=memory_size,
            num_buffers=num_parallel_envs
        )

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
            self._running_mean_obs.update(np.array([step[1] for step in steps]))
            with tf.name_scope('observations'):
                for i in range(self._running_mean_obs.mean.shape[0]):
                    tf.summary.scalar(
                        f'obs_{i}_running_mean', self._running_mean_obs.mean[i], step=self._tf_time_step
                    )
                    tf.summary.scalar(
                        f'obs_{i}_running_std', np.sqrt(self._running_mean_obs.var[i]), step=self._tf_time_step
                    )
        if self._running_mean_rewards:
            with tf.name_scope('rewards'):
                self._running_mean_rewards.update(np.expand_dims([step[2] for step in steps], axis=1))
                tf.summary.scalar(
                    f'rewards_running_std', np.sqrt(self._running_mean_rewards.var[0]), step=self._tf_time_step
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
        processed_obs = self._process_observations(observations)
        if is_deterministic:
            return self._actor.act_deterministic(processed_obs).numpy(), None
        else:
            actions, policies = self._actor(tf.convert_to_tensor(processed_obs))
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

            obs_flatten, obs_next_flatten, actions_flatten, policies_flatten, rewards_flatten, dones_flatten \
                = utils.flatten_experience(offline_batch)
            lengths = [len(batch['observations']) for batch in offline_batch]
            indices = tf.RaggedTensor.from_row_lengths(values=list(range(0, obs_flatten.shape[0])), row_lengths=lengths)

            obs_flatten = tf.convert_to_tensor(self._process_observations(obs_flatten))
            obs_next_flatten = tf.convert_to_tensor(self._process_observations(obs_next_flatten))
            actions_flatten = tf.convert_to_tensor(actions_flatten)
            policies_flatten = tf.convert_to_tensor(policies_flatten)
            rewards_flatten = tf.convert_to_tensor(self._process_rewards(rewards_flatten))
            dones_flatten = tf.convert_to_tensor(dones_flatten)

            first_obs = tf.convert_to_tensor(
                self._process_observations([batch['observations'][0] for batch in offline_batch])
            )
            first_actions = tf.convert_to_tensor([batch['actions'][0] for batch in offline_batch])

            self._learn_from_experience_batch(
                obs_flatten,
                obs_next_flatten,
                actions_flatten,
                policies_flatten,
                rewards_flatten,
                first_obs,
                first_actions,
                dones_flatten,
                indices
            )

    @tf.function(experimental_relax_shapes=True)
    def _learn_from_experience_batch(self, obs, obs_next, actions, old_policies,
                                     rewards, first_obs, first_actions, dones, batches_indices):
        """Backward pass with single batch of experience.

        Every experience replay requires sequence of experiences with random length, thus we have to use
        ragged tensors here.

        See Equation (8) and Equation (9) in the paper (1).
        """
        values = tf.squeeze(self._critic(obs))
        values_next = self._gamma * tf.squeeze(self._critic(obs_next)) * (1.0 - tf.cast(dones, tf.dtypes.float32))
        policies = tf.squeeze(self._actor.prob(obs, actions))
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
            tf.minimum(policies_ratio_product, self._b),
            batch_mask
        )
        coeffs_batches = tf.ones_like(truncated_densities).to_tensor() * (self._alpha * (1 - self._p))
        coeffs = tf.ragged.boolean_mask(
            tf.math.cumprod(coeffs_batches, axis=1, exclusive=True),
            batch_mask
        ).flat_values

        # flat tensors
        z_batches = coeffs * (rewards + values_next - values) * truncated_densities.flat_values
        # ragged
        z_batches = tf.gather_nd(z_batches, tf.expand_dims(indices, axis=2))
        # final summation over original batches
        z_batches = tf.stop_gradient(tf.reduce_sum(z_batches, axis=1))

        self._backward_pass(first_obs, first_actions, z_batches)

    def _backward_pass(self, observations: tf.Tensor, actions: tf.Tensor, z: tf.Tensor):
        """Performs backward pass in Actor's and Critic's networks

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

    def _fetch_offline_batch(self) -> List[Dict[str, Union[np.array, list]]]:
        trajectory_lens = [np.random.geometric(self._p) for _ in range(self._num_parallel_envs)]
        batch = []
        [batch.extend(self._memory.get(trajectory_lens)) for _ in range(self._batches_per_env)]
        return batch

    def _process_observations(self, observations: np.array) -> np.array:
        """If standardization is turned on, observations are being standardized with running mean and variance.
        Additional clipping is used to prevent performance spikes."""
        if self._running_mean_obs:
            return np.clip(
                (observations - self._running_mean_obs.mean) / np.sqrt(self._running_mean_obs.var + 1e-8),
                -10.0,
                10.0
            )
        else:
            return observations

    def _process_rewards(self, rewards: np.array) -> np.array:
        """Rescales returns with standard deviation. Additional clipping is used to prevent performance spikes."""
        if self._running_mean_rewards:
            return np.clip(
                rewards / np.sqrt(self._running_mean_rewards.var + 1e-8),
                -5.0,
                5.0
            )
        else:
            return rewards


