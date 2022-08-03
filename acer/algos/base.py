from abc import ABC, abstractmethod
from argparse import ArgumentError
from algos.common.automodel import AutoModel, AutoModelComponent
from pathlib import Path
from typing import Tuple, Union, List, Optional, Dict

import gym
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

import tf_utils
import utils
from models.cnn import build_cnn_network
from models.mlp import build_mlp_network
from replay_buffer import MultiReplayBuffer, BufferFieldSpec, ReplayBuffer

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)


class BaseModel(AutoModelComponent, tf.keras.Model):

    def __init__(
            self, observation_space: gym.spaces.Space, layers: Optional[Tuple[int]], output_dim: int,
            *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        if type(observation_space) != gym.spaces.Discrete:
            self._hidden_layers = self._build_layers(len(observation_space.shape), layers, output_dim)
            self._input_shape_len = len(observation_space.shape)
            self._forward = self._nn_forward
        else:
            self._array = tf.Variable(np.zeros((observation_space.n, output_dim)), dtype=tf.float32)
            self._forward = self._arr_forward

        self._output_dim = output_dim

        self.optimizer = None

    def _build_layers(
        self, input_shape_len: int, layers: Optional[Tuple[int]], output_dim: int
    ) -> List[tf.keras.Model]:
        hidden_layers = []

        if input_shape_len > 1:
            hidden_layers.extend(build_cnn_network())
        
        hidden_layers.extend(build_mlp_network(layers_sizes=layers))

        hidden_layers.append(tf.keras.layers.Dense(output_dim, kernel_initializer=utils.normc_initializer()))        

        return hidden_layers

    @tf.function(experimental_relax_shapes=True)
    def _nn_forward(self, input: np.array) -> tf.Tensor:
        shape = tf.shape(input)
        batch_dims = shape[1:-self._input_shape_len]
        input_dims = shape[-self._input_shape_len:]
        
        x = tf.reshape(input, tf.concat([[-1,], input_dims], axis=0))
        for layer in self._hidden_layers:
            x = layer(x)
        
        return tf.reshape(x, tf.concat([[-1,], batch_dims, [self._output_dim,]], axis=0))

    @tf.function(experimental_relax_shapes=True)
    def _arr_forward(self, input) -> tf.Tensor:
        return tf.gather_nd(self._array, tf.expand_dims(input, -1))

    def optimize(self, **loss_kwargs):
        with tf.GradientTape() as tape:
            loss = self.loss(**loss_kwargs)
        grads = tape.gradient(loss, self.trainable_variables)
        gradients = zip(grads, self.trainable_variables)

        self.optimizer.apply_gradients(gradients)

        return loss

    def init_optimizer(self, *args, **kwargs):
        self.optimizer = tf.keras.optimizers.Adam(*args, **kwargs)
        return self.optimizer

class BaseActor(BaseModel):

    def __init__(self, observations_space: gym.Space, actions_space: gym.Space, layers: Optional[Tuple[int]],
                 beta_penalty: float, tf_time_step: tf.Variable, *args, truncate: bool = True, b: float = 2, **kwargs):
        """Base abstract Actor class

        Args:
            observations_dim: dimension of observations space
            layers: list of hidden layers sizes, eg: for neural network with two layers with 10 and 20 hidden units
                pass: [10, 20]
            beta_penalty: penalty coefficient. In discrete case, BaseActor is penalized for too executing too
                confident actions (no exploration), in the continuous case it is penalized for making actions
                that are out of the allowed bounds
            tf_time_step: time step as TensorFlow variable, required for TensorBoard summaries
        """
        if type(actions_space) == gym.spaces.discrete.Discrete:
            actions_dim = actions_space.n
        else:
            actions_dim = actions_space.shape[0]

        super().__init__(observations_space, layers, actions_dim, *args, **kwargs)

        self.obs_shape = observations_space.shape
        self.actions_dim = actions_dim
        self.beta_penalty = beta_penalty
        self._tf_time_step = tf_time_step

        self._truncate = truncate
        self._b = b

        self.register_method('policies', self.policy, {'observations': 'obs', 'actions': 'actions'})
        self.register_method('actions', self.act_deterministic, {'observations': 'obs'})
        self.register_method('expected_probs', self.expected_probs, {'observations': 'obs'})
        self.register_method(
            'density', self._calculate_density, {
                'policies': 'actor.policies',
                'old_policies': 'policies',
                'mask': 'base.mask'
            }
        )
        self.register_method(
            'sample_weights', self._calculate_truncated_weights, {
                'density': 'actor.density',
                'priorities': 'priorities'
            }
        )
        self.register_method(
            'truncated_density', self._calculate_truncated_density, {
                'density': 'actor.density'
            }
        )

        self.register_method('optimize', self.optimize, {
            'observations': 'base.first_obs',
            'actions': 'base.first_actions',
            'd': 'base.weighted_td'
        })
        self.targets = ['optimize']

    def policy(self, observations, actions):
        return self.prob(observations, actions)[0]

    @property
    @abstractmethod
    def action_dtype(self):
        """Returns data type of the BaseActor's actions (TensorFlow)"""

    @property
    @abstractmethod
    def action_dtype_np(self):
        """Returns data type of the BaseActor's actions (Numpy)"""

    @abstractmethod
    def prob(self, observations: np.array, actions: np.array) -> tf.Tensor:
        """Computes probabilities (or probability densities in continuous case) and logarithms of it

        Args:
            observations: batch [batch_size, observations_dim] of observations vectors
            actions: batch [batch_size, actions_dim] of actions vectors

        Returns:
             Tensor [batch_size, actions_dim, 2] with computed probabilities (densities) and logarithms
        """

    @abstractmethod
    def act(self, observations: np.array, **kwargs) -> Tuple[tf.Tensor, tf.Tensor]:
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

    @abstractmethod
    def expected_probs(self, observations: tf.Tensor) -> tf.Tensor:
        """expected value of action probability"""

    @tf.function(experimental_relax_shapes=True)
    def _calculate_density(self, policies, old_policies, mask):
        policies_masked = policies * mask + (1 - mask) * tf.ones_like(policies)
        old_policies_masked = old_policies * mask + (1 - mask) * tf.ones_like(old_policies)

        policies_ratio = policies_masked / old_policies_masked
        policies_ratio_prod = tf.math.cumprod(policies_ratio, axis=-1)

        return policies_ratio_prod

    def _calculate_truncated_density(self, density):
        if self._truncate:
            density = tf.tanh(density / self._b) * self._b

        return density

    def _calculate_truncated_weights(self, density, priorities):
        weights = density / tf.reshape(priorities, (-1, 1))

        if self._truncate:
            weights = tf.tanh(weights / self._b) * self._b

        return weights

    def _update_ends(self, ends):
        pass

class BaseCritic(BaseModel):

    def __init__(self, observations_space: gym.Space, layers: Optional[Tuple[int]],
                 tf_time_step: tf.Variable, use_additional_input: bool = False, *args, **kwargs):
        """Value function approximation as MLP network neural network.

        Args:
            observations_dim: dimension of observations space
            layers: list of hidden layers sizes, eg: for neural network with two layers with 10 and 20 hidden units
                pass: [10, 20]
            tf_time_step: time step as TensorFlow variable, required for TensorBoard summaries
            additional_input_shape: shape of additional input variables
        """
        if len(observations_space.shape) > 1:
            assert not use_additional_input

        super().__init__(observations_space, layers, 1, *args, **kwargs)

        self.obs_shape = observations_space.shape
        self._tf_time_step = tf_time_step
        self._use_additional_input = use_additional_input

        self.register_method('value', self.value, {'observations': 'obs'})
        self.register_method('value_next', self.value, {'observations': 'obs_next'})

        self.register_method('optimize', self.optimize, {
            'observations': 'base.first_obs',
            'd': 'base.weighted_td'
        })
        self.targets = ['optimize']

    def call(self, inputs, training=None, mask=None, additional_input=None):
        return self.value(inputs, additional_input=additional_input)

    def value(self, observations: tf.Tensor, additional_input: tf.Tensor=None, **kwargs) -> tf.Tensor:
        """Calculates value function given observations batch

        Args:
            observations: batch [batch_size, observations_dim] of observations vectors


        Returns:
            Tensor [batch_size, 1] with value function estimations

        """
        x = tf.concat([observations, additional_input], axis=-1) if self._use_additional_input else observations
        return self._forward(x)


class Critic(BaseCritic):

    def __init__(self, observations_space: gym.Space, layers: Optional[Tuple[int]], tf_time_step: tf.Variable,
                 use_additional_input: bool = False, *args, **kwargs):
        """Basic Critic that outputs single value"""
        super().__init__(observations_space, layers, tf_time_step, *args, use_additional_input=use_additional_input, **kwargs)

    def loss(self, observations: np.array, d: np.array, additional_input=None) -> tf.Tensor:
        """Computes Critic's loss.

        Args:
            observations: batch [batch_size, observations_dim] of observations vectors
            d: batch [batch_size, 1] of gradient update coefficient (summation term in the Equation (9)) from
                the paper (1))
        """

        value = self.value(observations, additional_input=additional_input)
        loss = tf.reduce_mean(-tf.math.multiply(value, d))

        return loss


class CategoricalActor(BaseActor):

    def __init__(self, observations_space: gym.Space, actions_space: gym.Space, layers: Optional[Tuple[int]],
                 *args, entropy_coeff=0., **kwargs):
        """BaseActor for discrete actions spaces. Uses Categorical Distribution"""
        super().__init__(observations_space, actions_space, layers, *args, **kwargs)
        self._entropy_coeff = entropy_coeff
        self.n = actions_space.n

    @property
    def action_dtype(self):
        return tf.dtypes.int32

    @property
    def action_dtype_np(self):
        return np.int32

    def loss(self, observations: tf.Tensor, actions: tf.Tensor, d: tf.Tensor) -> tf.Tensor:
        probs, log_probs, action_probs, action_log_probs = self._prob(observations, actions)

        total_loss = tf.reduce_mean(-tf.math.multiply(tf.expand_dims(action_log_probs, 1), d))  # + penalty)

        # entropy maximization penalty
        entropy = -tf.reduce_sum(probs * log_probs, axis=1)
        # penalty = self.beta_penalty * (-tf.reduce_sum(tf.math.multiply(probs, log_probs), axis=1))

        return total_loss - entropy * self._entropy_coeff

    def prob(self, observations: tf.Tensor, actions: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        _, _, action_probs, action_log_probs = self._prob(observations, actions)
        return action_probs, action_log_probs

    def _prob(self, observations: tf.Tensor, actions: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        logits = self._forward(observations)  # tf.divide(self._forward(observations) , 10)

        action_shape = tf.shape(actions)[:-1]
        actions = tf.reshape(actions, (-1, 1))
        logits = tf.reshape(logits, (-1, tf.shape(logits)[-1]))

        probs = tf.nn.softmax(logits)
        log_probs = tf.nn.log_softmax(logits) 

        action_probs = tf.gather(probs, actions, batch_dims=1)
        action_log_probs = tf.gather(log_probs, actions, batch_dims=1)

        action_probs = tf.reshape(action_probs, action_shape)
        action_log_probs = tf.reshape(action_probs, action_shape)

        return probs, log_probs, action_probs, action_log_probs

    @tf.function(experimental_relax_shapes=True)
    def act(self, observations: tf.Tensor, **kwargs) -> Tuple[tf.Tensor, tf.Tensor]:

        # TODO: remove hardcoded '10' and '20'
        logits = self._forward(observations)  # tf.divide(self._forward(observations) , 10)
        probs = tf.nn.softmax(logits)
        log_probs = tf.nn.log_softmax(logits)

        dist = tfp.distributions.Categorical(log_probs, dtype=tf.int32)

        actions = dist.sample()
        actions = tf.clip_by_value(actions, 0, self.n - 1)  # there was some weird error

        action_probs = tf.reshape(tf.gather(
            tf.reshape(probs, (-1, self.n)),
            tf.reshape(actions, (-1, 1)),
            batch_dims=1
        ), tf.shape(actions))

        return actions, action_probs

    @tf.function(experimental_relax_shapes=True)
    def act_deterministic(self, observations: tf.Tensor, **kwargs) -> tf.Tensor:
        """Performs most probable action"""
        logits = self._forward(observations)  # tf.divide(self._forward(observations) , 10)

        actions = tf.argmax(logits, axis=1)
        return actions

    @tf.function(experimental_relax_shapes=True)
    def expected_probs(self, observations: tf.Tensor) -> tf.Tensor:
        logits = self._forward(observations)
        probs = tf.nn.softmax(logits)
        eprobs = tf.math.cumprod(tf.reduce_sum(probs ** 2, -1),axis=-1)

        return eprobs

class GaussianActor(BaseActor):

    def __init__(self, observations_space: gym.Space, actions_space: gym.Space, layers: Optional[Tuple[int]],
                 beta_penalty: float, actions_bound: float = None, std: float = None, *args, **kwargs):
        """BaseActor for continuous actions space. Uses MultiVariate Gaussian Distribution as policy distribution.

        TODO: introduce [a, b] intervals as allowed actions bounds

        Args:
            observations_dim: dimension of observations space
            layers: list of hidden layer sizes
            beta_penalty: penalty for too confident actions coefficient
            actions_bound: upper (lower == '-actions_bound') bound for allowed actions,
             required in case of continuous actions, deprecated (now taken as actions_space.high)
        """
        super().__init__(observations_space, actions_space, layers, beta_penalty, *args, **kwargs)

        self._actions_bound = actions_space.high
        self._k = actions_space.shape[0]

        if std:
            # change constant to Variable to make std a learned parameter
            self.log_std = tf.constant(
                tf.math.log([std] * self._k),
                name="actor_std",
            )
        else:
            self.log_std = tf.constant(
                tf.math.log(0.4 * self._actions_bound),
                name="actor_std",
            )

    @property
    def action_dtype(self):
        return tf.dtypes.float32

    @property
    def action_dtype_np(self):
        return np.float32

    def loss(self, observations: np.array, actions: np.array, d: np.array) -> tf.Tensor:
        mean = self._forward(observations)
        dist = tfp.distributions.MultivariateNormalDiag(
            loc=mean,
            scale_diag=tf.exp(self.log_std)
        )

        return self._loss(mean, dist, actions, d)

    def _loss(self, mean: tf.Tensor, dist: tf.Tensor, actions: np.array, d: np.array) -> tf.Tensor:
        action_log_probs = tf.expand_dims(dist.log_prob(actions), axis=1)

        bounds_penalty = tf.reduce_sum(
            tf.scalar_mul(
                self.beta_penalty,
                tf.square(tf.maximum(0.0, tf.abs(mean) - self._actions_bound))
            ),
            axis=1,
            keepdims=True
        )

        total_loss = tf.reduce_mean(-tf.math.multiply(action_log_probs, d) + bounds_penalty)

        return total_loss

    def _dist(self, observations: tf.Tensor) -> tf.Tensor:
        mean = self._forward(observations)
        dist = tfp.distributions.MultivariateNormalDiag(
            loc=mean,
            scale_diag=tf.exp(self.log_std)
        )
        return dist

    def prob(self, observations: tf.Tensor, actions: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        dist = self._dist(observations)
        return self._prob(dist, actions)

    def _prob(self, dist: tf.Tensor, actions: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        return dist.prob(actions), dist.log_prob(actions)

    @tf.function
    def act(self, observations: tf.Tensor, **kwargs) -> Tuple[tf.Tensor, tf.Tensor]:
        dist = self._dist(observations)
        return self._act(dist)

    @tf.function
    def _act(self, dist: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        actions = dist.sample(dtype=self.dtype)
        actions_probs = dist.prob(actions)

        return actions, actions_probs

    @tf.function
    def act_deterministic(self, observations: tf.Tensor, **kwargs) -> tf.Tensor:
        """Returns mean of the Gaussian"""
        mean = self._forward(observations)
        return mean

    @tf.function(experimental_relax_shapes=True)
    def expected_probs(self, observations: tf.Tensor) -> tf.Tensor:
        det = tf.exp(2 * tf.reduce_sum(self.log_std))
        mult = (4 * np.pi) ** tf.cast(self._k, tf.float32)

        return tf.math.cumprod(tf.ones(tf.shape(observations)[:2], tf.float32) / tf.sqrt(mult * det), axis=1)


class BaseACERAgent(AutoModelComponent, AutoModel):
    """Base ACER abstract class"""
    def __init__(self, observations_space: gym.Space, actions_space: gym.Space, actor_layers: Optional[Tuple[int]],
                 critic_layers: Tuple[int], gamma: int = 0.99, actor_beta_penalty: float = 0.001,
                 std: Optional[float] = None, memory_size: int = 1e6, num_parallel_envs: int = 10,
                 batches_per_env: int = 5, c: int = 10, c0: float = 0.3, actor_lr: float = 0.001,
                 actor_adam_beta1: float = 0.9, actor_adam_beta2: float = 0.999, actor_adam_epsilon: float = 1e-7,
                 critic_lr: float = 0.001, critic_adam_beta1: float = 0.9, critic_adam_beta2: float = 0.999,
                 critic_adam_epsilon: float = 1e-7, standardize_obs: bool = False, rescale_rewards: int = -1,
                 limit_reward_tanh: float = None, time_step: int = 1, gradient_norm: float = None,
                 gradient_norm_median_threshold: float = 4, learning_starts: int = 1000, 
                 additional_buffer_types: List = (), policy_spec: BufferFieldSpec = None, **kwargs):

        super().__init__()

        self._tf_time_step = tf.Variable(
            initial_value=time_step, name='tf_time_step', dtype=tf.dtypes.int64, trainable=False
        )
        self._observations_space = observations_space
        self._actions_space = actions_space
        self._std = std
        self._actor_beta_penalty = actor_beta_penalty
        self._c = c
        self._c0 = c0
        self._learning_starts = learning_starts
        self._actor_layers = tuple(actor_layers)
        self._critic_layers = tuple(critic_layers)
        self._gamma = gamma
        self._batches_per_env = batches_per_env
        self._time_step = 0
        self._num_parallel_envs = num_parallel_envs
        self._limit_reward_tanh = limit_reward_tanh
        self._gradient_norm = gradient_norm
        self._gradient_norm_median_threshold = gradient_norm_median_threshold
        self._batch_size = self._num_parallel_envs * self._batches_per_env
        self._memory_size = memory_size

        self._actor_gradient_norm_median = tf.Variable(initial_value=1.0, trainable=False)
        self._critic_gradient_norm_median = tf.Variable(initial_value=1.0, trainable=False)

        self._is_obs_discrete = type(observations_space) == gym.spaces.Discrete

        if type(actions_space) == gym.spaces.Discrete:
            self._is_discrete = True
            self._actions_bound = 0
        else:
            self._is_discrete = False
            self._actions_bound = actions_space.high

        self._actor = self._init_actor()
        self._critic = self._init_critic()

        self._init_replay_buffer(memory_size, policy_spec)

        self.register_method("first_obs", self._first_obs, {"obs": "obs"})
        self.register_method("first_actions", self._first_actions, {"actions": "actions"})

        self._init_automodel()

        self._init_data_loader(additional_buffer_types)

        self._actor_optimizer = self._actor.init_optimizer(
            lr=actor_lr,
            beta_1=actor_adam_beta1,
            beta_2=actor_adam_beta2,
            epsilon=actor_adam_epsilon
        )

        self._critic_optimizer = self._critic.init_optimizer(
            lr=critic_lr,
            beta_1=critic_adam_beta1,
            beta_2=critic_adam_beta2,
            epsilon=critic_adam_epsilon
        )

        if standardize_obs:
            self._running_mean_obs = tf_utils.RunningMeanVarianceTf(shape=observations_space.shape)
        else:
            self._running_mean_obs = None

        self._rescale_rewards = rescale_rewards
        if rescale_rewards == 0:
            self._running_mean_rewards = tf_utils.RunningMeanVarianceTf(shape=(1, ))
        else:
            self._running_mean_rewards = None

    def _first_obs(self, obs):
        return obs[:, 0]

    def _first_actions(self, actions):
        return actions[:, 0]

    def _init_automodel(self) -> None:
        pass  # call for automodel-compatibile classes (FastACER and subseq.) before immediatly before initializing data loader

    def _init_data_loader(self, additional_buffer_types) -> None:
        self._data_loader = tf.data.Dataset.from_generator(
            self._experience_replay_generator,
            (tf.dtypes.float32, tf.dtypes.float32, self._actor.action_dtype, tf.dtypes.float32, tf.dtypes.float32,
             tf.dtypes.float32, self._actor.action_dtype, tf.dtypes.bool, tf.dtypes.int32, *additional_buffer_types)
        )

    def _init_replay_buffer(self, memory_size: int, policy_spec: BufferFieldSpec = None):
        if type(self._actions_space) == gym.spaces.Discrete:
            actions_shape = (1, )
        else:
            actions_shape = self._actions_space.shape

        self._memory = MultiReplayBuffer(
            buffer_class=ReplayBuffer,
            action_spec=BufferFieldSpec(shape=actions_shape, dtype=self._actor.action_dtype_np),
            obs_spec=BufferFieldSpec(shape=self._observations_space.shape, dtype=self._observations_space.dtype),
            policy_spec=policy_spec,
            max_size=memory_size,
            num_buffers=self._num_parallel_envs
        )

    def _clip_gradient(
            self, grads: List[tf.Tensor], norm_variable: tf.Variable, scope: str
    ) -> List[tf.Tensor]:
        if self._gradient_norm == 0:
            grads_clipped, grads_norm = tf.clip_by_global_norm(
                grads,
                norm_variable * self._gradient_norm_median_threshold
            )
            update_sign = tf.pow(
                -1.0, tf.cast(tf.less(grads_norm, norm_variable), dtype=tf.float32)
            )
            norm_variable.assign_add(
                update_sign * norm_variable * 0.01
            )
        else:
            grads_clipped, grads_norm = tf.clip_by_global_norm(
                grads,
                self._gradient_norm
            )
        return grads_clipped

    def save_experience(self, steps: List[
        Tuple[Union[int, float, list], np.array, float, np.array, bool, bool]
    ]):
        """Stores gathered experiences in a replay buffer. Accepts list of steps.

        Args:
            steps: List of steps, see ReplayBuffer.put() for a detailed format description
        """
        self._time_step += len(steps)
        self._tf_time_step.assign_add(len(steps))
        self._memory.put(steps)

        if self._running_mean_obs:
            obs = np.array([step[1] for step in steps])
            self._update_obs_rms(obs)
        if self._running_mean_rewards:
            rewards = np.array([step[2] for step in steps])
            self._update_rewards_rms(rewards)

        self._actor.update_ends(np.array([[step[5]] for step in steps]))

    @tf.function(experimental_relax_shapes=True)
    def _update_obs_rms(self, obs):
        if self._running_mean_obs:
            self._running_mean_obs.update(obs)

    @tf.function(experimental_relax_shapes=True)
    def _update_rewards_rms(self, rewards):
        if self._running_mean_rewards:
            self._running_mean_rewards.update(tf.expand_dims(tf.cast(rewards, dtype=tf.float32), axis=1))

    def predict_action(self, observations: np.array, is_deterministic: bool = False) \
            -> Tuple[np.array, Optional[np.array]]:
        """Predicts actions for given observations. Performs forward pass with BaseActor network.

        Args:
            observations: batch [batch_size, observations_dim] of observations vectors
            is_deterministic: True if mean actions (without exploration noise) should be returned

        Returns:
            Tuple of sampled actions and corresponding probabilities (probability densities) if action was sampled
                from the distribution, None otherwise
        """
        processed_obs = tf.convert_to_tensor(self._process_observations(observations))
        if is_deterministic:
            return self._actor.act_deterministic(processed_obs).numpy(), None
        else:
            actions, policies = self._actor.act(processed_obs)
            return actions.numpy(), policies.numpy()

    def _experience_replay_generator(self):
        while True:
            offline_batch = self._fetch_offline_batch()

            obs_flatten, obs_next_flatten, actions_flatten, policies_flatten, rewards_flatten, dones_flatten \
                = utils.flatten_experience(offline_batch)

            lengths = [len(batch[0]['observations']) for batch in offline_batch]

            first_obs = [batch[0]['observations'][0] for batch in offline_batch]
            first_actions = [batch[0]['actions'][0] for batch in offline_batch]

            yield (
                obs_flatten,
                obs_next_flatten,
                actions_flatten,
                policies_flatten,
                rewards_flatten,
                first_obs,
                first_actions,
                dones_flatten,
                lengths
            )

    def _process_observations(self, observations: tf.Tensor) -> tf.Tensor:
        """If standardization is turned on, observations are being standardized with running mean and variance.
        Additional clipping is used to prevent performance spikes."""
        if self._running_mean_obs:
            centered = observations - self._running_mean_obs.mean
            standardized = centered / tf.sqrt(self._running_mean_obs.var + 1e-8)
            return tf.clip_by_value(
                standardized,
                -10.0,
                10.0
            )
        else:
            return observations

    def _process_rewards(self, rewards: tf.Tensor) -> tf.Tensor:
        """Rescales returns with standard deviation. Additional clipping is used to prevent performance spikes."""
        if self._rescale_rewards == 0:
            rewards = rewards / tf.sqrt(self._running_mean_rewards.var + 1e-8)
            if not self._limit_reward_tanh:
                rewards = tf.clip_by_value(
                    rewards / tf.sqrt(self._running_mean_rewards.var + 1e-8),
                    -5.0,
                    5.0
                )
        elif self._rescale_rewards >= 0:
            rewards = rewards / self._rescale_rewards

        if self._limit_reward_tanh:
            rewards = tf.tanh(rewards / self._limit_reward_tanh) * self._limit_reward_tanh

        return rewards

    @abstractmethod
    def _fetch_offline_batch(self) -> List[Tuple[Dict[str, Union[np.array, list]], int]]:
        ...

    @abstractmethod
    def learn(self):
        ...

    @abstractmethod
    def _init_actor(self) -> BaseActor:
        ...

    @abstractmethod
    def _init_critic(self) -> BaseCritic:
        ...

    def save(self, path: Path, **kwargs):
        actor_path = str(path / 'actor.tf')
        critic_path = str(path / 'critic.tf')
        buffer_path = str(path / 'buffer.pkl')

        self._actor.save_weights(actor_path, overwrite=True)
        self._critic.save_weights(critic_path, overwrite=True)

        if self._running_mean_obs:
            rms_obs_path = str(path / 'rms_obs.pkl')
            self._running_mean_obs.save(rms_obs_path)

        if self._running_mean_rewards:
            rms_rewards_path = str(path / 'rms_rewards.pkl')
            self._running_mean_rewards.save(rms_rewards_path)

        self._memory.save(buffer_path)
