import datetime
import json

import logging
import time
from typing import Optional, List, Union, Tuple

import gym
import numpy as np
import tensorflow as tf

from algos.classic_acer import ClassicACER
from algos.base import ACERAgent
from environment import SequentialEnv
from logger import CSVLogger

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)-5.5s]  %(message)s",
)


def _get_agent(algorithm: str, parameters: Optional[dict], observations_space: gym.Space,
               actions_space: gym.Space) -> ACERAgent:
    """Initializes Agent object"""
    if not parameters:
        parameters = {}
    if algorithm == 'classic':
        return ClassicACER(observations_space=observations_space, actions_space=actions_space, **parameters)
    else:
        raise NotImplemented


class Runner:

    MEASURE_TIME_TIME_STEPS = 1000

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
            log_dir: logging directory
            max_time_steps: maximum number of training time steps
        """
        self._elapsed_time_measure = 0
        self._time_step = 0
        self._done_episodes = 0
        self._next_evaluation_timestamp = 0
        self._n_envs = num_parallel_envs
        self._evaluate_time_steps_interval = evaluate_time_steps_interval
        self._num_evaluation_runs = num_evaluation_runs
        self._max_time_steps = max_time_steps
        self._env_name = environment_name
        self._env = SequentialEnv(environment_name, num_parallel_envs)
        self._done_steps_in_a_episode = [0] * self._n_envs
        self._returns = [0] * self._n_envs

        self._max_steps_in_episode = self._env.spec.max_episode_steps

        self._log_dir = f"{log_dir}/{environment_name}" \
                        f"_{algorithm}_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
        tensor_board_writer = tf.summary.create_file_writer(self._log_dir)
        tensor_board_writer.set_as_default()

        self._csv_logger = CSVLogger(self._log_dir + '/results.csv', ['time_step', 'eval_return_mean', 'eval_std_mean'])

        self._save_parameters(algorithm_parameters)

        self._agent = _get_agent(algorithm, algorithm_parameters, self._env.observation_space, self._env.action_space)
        self._current_obs = self._env.reset_all()

    def run(self):
        """Performs training. If 'evaluate' is True, evaluation of the policy is performed. The evaluation
        uses policy that is being optimized, not the one used in training (i.e. randomness is turned off)
        """
        while self._max_time_steps == -1 or self._time_step <= self._max_time_steps:

            if self._is_time_to_evaluate():
                self._evaluate()

            start_time = time.time()
            experience = self._step()
            self._agent.save_experience(experience)
            self._agent.learn()
            self._elapsed_time_measure += time.time() - start_time

        self._csv_logger.close()

    def _step(self) -> List[Tuple[Union[int, float], np.array, float, float, bool, bool]]:
        actions, policies = self._agent.predict_action(self._current_obs)
        steps = self._env.step(actions)
        rewards = []
        experience = []
        old_obs = self._current_obs
        self._current_obs = []

        for i, step in enumerate(steps):
            # 'is_done' from Gym does not take into account maximum number of steps in a single episode constraint
            self._time_step += 1
            if self._time_step % Runner.MEASURE_TIME_TIME_STEPS == 0:
                self._measure_time()

            rewards.append(step[1])
            self._done_steps_in_a_episode[i] += 1
            is_done_gym = step[2]
            is_maximum_number_of_steps_reached = self._max_steps_in_episode is not None \
                and self._max_steps_in_episode == self._done_steps_in_a_episode[i]

            is_done = is_done_gym and not is_maximum_number_of_steps_reached
            is_end = is_done or is_maximum_number_of_steps_reached

            reward = step[1]
            experience.append(
                (actions[i], old_obs[i], reward, policies[i], is_done, is_end)
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

        mean_returns = np.mean(returns)
        std_returns = np.std(returns)

        with tf.name_scope('rewards'):
            tf.summary.scalar('evaluation_return_mean', mean_returns, self._time_step)
            tf.summary.scalar('evaluation_return_std', std_returns, self._time_step)

        self._csv_logger.log_values(
            {'time_step': self._time_step, 'eval_return_mean': mean_returns, 'eval_std_mean': std_returns}
        )

    def _is_time_to_evaluate(self):
        return self._evaluate_time_steps_interval != -1 and self._time_step >= self._next_evaluation_timestamp

    def _measure_time(self):
        with tf.name_scope('acer'):
            tf.summary.scalar(
                'time steps per second',
                Runner.MEASURE_TIME_TIME_STEPS / self._elapsed_time_measure,
                self._time_step
            )
        self._elapsed_time_measure = 0

    def _save_parameters(self, algorithm_parameters: dict):
        with open(self._log_dir + '/parameters.json', 'wt') as f:
            json.dump(algorithm_parameters, f)
