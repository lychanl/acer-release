import datetime
import logging
from typing import Optional, List, Union, Tuple

import numpy as np
import tensorflow as tf

from algos.acer import ACER
from algos.base import Agent
from environment import SequentialEnv
import utils

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)-5.5s]  %(message)s",
)


def _get_agent(algorithm: str, parameters: Optional[dict], is_continuous: bool,
               observations_dim: int, actions_dim: int, actions_scale: float) -> Agent:
    """Initializes Agent object"""
    if not parameters:
        parameters = {}
    if algorithm == 'acer':
        return ACER(is_discrete=not is_continuous, observations_dim=observations_dim,
                    actions_dim=actions_dim, actions_bound=actions_scale, **parameters)
    else:
        raise NotImplemented


class Runner:
    def __init__(self, environment_name: str, algorithm: str = 'acer', algorithm_parameters: Optional[dict] = None,
                 num_parallel_envs: int = 5, evaluate_time_steps_interval: int = 1500,
                 num_evaluation_runs: int = 5, log_dir: str = 'logs/', max_time_steps: int = -1):
        """Trains and evaluates the agent.

        TODO: frames saving
        TODO: checkpoints saving

        Args:
            environment_name: environment to be created
            algorithm: algorithm name, one of the following: ['acer']
            algorithm_parameters: parameters of the agent
            num_parallel_envs: number of environments run in the parallel
            evaluate_time_steps_interval: number of time steps between evaluation runs, -1 if
                no evaluation should be done
            num_evaluation_runs: number of runs per one evaluation
            log_dir: TensorBoard logging directory
            max_time_steps: maximum number of training time steps
        """
        self._time_step = 0
        self._done_episodes = 0
        self._n_envs = num_parallel_envs
        self._evaluate_time_steps_interval = evaluate_time_steps_interval
        self._num_evaluation_runs = num_evaluation_runs
        self._max_time_steps = max_time_steps
        self._env_name = environment_name
        self._env = SequentialEnv(environment_name, num_parallel_envs)

        self._log_dir = log_dir + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensor_board_writer = tf.summary.create_file_writer(self._log_dir)
        tensor_board_writer.set_as_default()

        self._action_scale, self._actions_dim, self._observations_dim,\
            self._is_continuous, self._max_steps_in_episode = utils.get_env_variables(self._env)

        self._done_steps_in_a_episode = [0] * self._n_envs
        self._returns = [0] * self._n_envs
        self._agent = _get_agent(algorithm, algorithm_parameters, self._is_continuous,
                                 self._observations_dim, self._actions_dim, self._action_scale)
        self._current_obs = utils.reset_env_and_agent(self._agent, self._env)

        self._next_evaluation_timestamp = 0

    def run(self):
        """Performs training. If 'evaluate' is True, evaluation of the policy is performed. The evaluation
        uses optimized policy, not the one used in training (i.e. randomness is turned off)
        """
        while self._max_time_steps == -1 or self._time_step <= self._max_time_steps:

            if self._is_time_to_evaluate():
                self._evaluate()

            experience = self._step()
            self._agent.save_experience(experience)
            self._agent.learn()

    def _step(self) -> List[Tuple[Union[int, float], np.array, float, np.array, float, bool, bool]]:
        actions, policies = self._agent.predict_action(self._current_obs)
        steps = self._env.step(actions)
        rewards = []
        experience = []
        old_obs = self._current_obs
        self._current_obs = []

        for i, step in enumerate(steps):
            # 'is_done' from Gym does not take into account maximum number of steps in a single episode constraint
            self._time_step += 1
            rewards.append(step[1])
            self._done_steps_in_a_episode[i] += 1
            is_done_gym = step[2]
            is_maximum_number_of_steps_reached = self._max_steps_in_episode is not None \
                and self._max_steps_in_episode == self._done_steps_in_a_episode[i]

            is_done = is_done_gym and not is_maximum_number_of_steps_reached
            is_end = is_done or is_maximum_number_of_steps_reached

            # reward_clipped = float(np.clip(step[1], -5, 5))
            reward_clipped = step[1]
            experience.append(
                (actions[i], old_obs[i], reward_clipped, step[0], policies[i], is_done, is_end)
            )

            self._current_obs.append(step[0])
            self._returns[i] += step[1]

            if is_end:
                self._current_obs[i] = self._env.reset(i)
                self._done_episodes += 1

                logging.info(f"finished episode {self._done_episodes}, "
                             f"return: {self._returns[i]}, "
                             f"total time steps done: {self._time_step}")

                with tf.name_scope('rewards'):
                    tf.summary.scalar('return', self._returns[i], self._done_episodes)
                    tf.summary.scalar('episode length', self._done_steps_in_a_episode[i], self._done_episodes)

                self._returns[i] = 0
                self._done_steps_in_a_episode[i] = 0

        self._current_obs = np.array(self._current_obs)

        with tf.name_scope('rewards'):
            tf.summary.scalar('mean_reward', np.mean(rewards), self._time_step)
            tf.summary.scalar('max_reward', np.max(rewards), self._time_step)
            tf.summary.scalar('min_reward', np.min(rewards), self._time_step)

        return experience

    def _evaluate(self):
        self._next_evaluation_timestamp += self._evaluate_time_steps_interval

        returns = []
        env = SequentialEnv(self._env_name, 1)
        for _ in range(self._num_evaluation_runs):
            time_step = 0
            current_obs = np.expand_dims(env.reset(0), axis=0)
            evaluation_return = 0
            is_end = False

            while not is_end:
                time_step += 1
                actions, _ = self._agent.predict_action(current_obs, is_deterministic=True)
                steps = env.step(actions)
                evaluation_return += steps[0][1]
                current_obs = np.expand_dims(steps[0][0], axis=0)
                is_done_gym = steps[0][2]
                is_maximum_number_of_steps_reached = self._max_steps_in_episode is not None\
                    and self._max_steps_in_episode == time_step

                is_end = is_done_gym or is_maximum_number_of_steps_reached
            logging.info(f"evaluation run, "
                         f"return: {evaluation_return}")
            returns.append(evaluation_return)

        with tf.name_scope('rewards'):
            tf.summary.scalar('evaluation_return_mean', np.mean(returns), self._time_step)
            tf.summary.scalar('evaluation_return_std', np.std(returns), self._time_step)

    def _is_time_to_evaluate(self):
        return self._evaluate_time_steps_interval != -1 and self._time_step >= self._next_evaluation_timestamp
