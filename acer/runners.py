import datetime
import json

import logging
import time
from typing import Optional, List, Union, Tuple
from pathlib import Path

import gym
import pybullet_envs
import numpy as np
import tensorflow as tf
from gym import wrappers

from algos.acer import ACER
from algos.base import BaseACERAgent
from algos.eacer import EACER
from algos.pessimistic_acer import PACER
from algos.representative_acer import RepresentativeACER
from algos.quantile_acer import QACER
from algos.weighted_acer import WeightedACER
from logger import CSVLogger
from utils import is_atari

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)-5.5s]  %(message)s",
)


ALGOS = {
    'acer': ACER,
    'eacer': EACER,
    'pacer': PACER,
    'racer': RepresentativeACER,
    'qacer': QACER,
    'wacer': WeightedACER
}


def _get_agent(algorithm: str, parameters: Optional[dict], observations_space: gym.Space,
               actions_space: gym.Space) -> BaseACERAgent:
    if not parameters:
        parameters = {}
    
    if algorithm not in ALGOS:
        raise NotImplemented

    return ALGOS[algorithm](observations_space=observations_space, actions_space=actions_space, **parameters)

def _get_env(env_id: str, num_parallel_envs: int, asynchronous: bool = True) -> gym.vector.AsyncVectorEnv:
    if is_atari(env_id):
        def get_env_fn():
            return wrappers.AtariPreprocessing(
                gym.make(env_id),
            )
        builders = [get_env_fn for _ in range(num_parallel_envs)]
        env = gym.vector.AsyncVectorEnv(builders) if asynchronous else gym.vector.SyncVectorEnv(builders)
    else:
        env = gym.vector.make(env_id, num_envs=num_parallel_envs, asynchronous=asynchronous)
    return env


class Runner:

    MEASURE_TIME_TIME_STEPS = 1000

    def __init__(self, environment_name: str, algorithm: str = 'acer', algorithm_parameters: Optional[dict] = None,
                 num_parallel_envs: int = 5, evaluate_time_steps_interval: int = 1500,
                 num_evaluation_runs: int = 5, log_dir: str = 'logs/', max_time_steps: int = -1,
                 record: bool = True, experiment_name: str = None, asynchronous: bool = True):
        """Trains and evaluates the agent.

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
            record: True if video should be recorded after training
            asynchronous: True to use concurrent envs
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
        if experiment_name:
            self._log_dir = Path(
                f"{log_dir}/{environment_name}_{algorithm}_{experiment_name}"
                f"_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
            )
        else:
            self._log_dir = Path(
                f"{log_dir}/{environment_name}_{algorithm}_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
            )
        self._log_dir.mkdir(parents=True, exist_ok=True)

        self._record = record
        self._env = _get_env(environment_name, num_parallel_envs, asynchronous)
        self._evaluate_env = _get_env(environment_name, num_evaluation_runs, asynchronous)

        self._done_steps_in_a_episode = [0] * self._n_envs
        self._returns = [0] * self._n_envs
        self._rewards = [[] for _ in range(self._n_envs)]

        dummy_env = self._env.env_fns[0]()
        self._max_steps_in_episode = dummy_env.spec.max_episode_steps

        tensor_board_writer = tf.summary.create_file_writer(str(self._log_dir))
        tensor_board_writer.set_as_default()

        self._csv_logger = CSVLogger(
            self._log_dir / 'results.csv',
            keys=['time_step', 'eval_return_mean', 'eval_std_mean']
        )

        self._save_parameters(algorithm_parameters)
        self._agent = _get_agent(algorithm, algorithm_parameters, dummy_env.observation_space, dummy_env.action_space)
        self._current_obs = self._env.reset()

    def run(self):
        """Performs training. If 'evaluate' is True, evaluation of the policy is performed. The evaluation
        uses policy that is being optimized, not the one used in training (i.e. randomness is turned off)
        """
        while self._max_time_steps == -1 or self._time_step <= self._max_time_steps:

            if self._is_time_to_evaluate():
                self._evaluate()
                if self._time_step != 0:
                    self._save_checkpoint()

            start_time = time.time()
            experience = self._step()
            self._agent.save_experience(experience)
            self._agent.learn()
            self._elapsed_time_measure += time.time() - start_time

        self._csv_logger.close()
        if self._record:
            self.record_video()

    def _step(self) -> List[Tuple[Union[int, float], np.array, float, float, bool, bool]]:
        actions, policies = self._agent.predict_action(self._current_obs)
        steps = self._env.step(actions)
        rewards = []
        experience = []
        old_obs = self._current_obs
        self._current_obs = steps[0]

        for i in range(self._n_envs):
            # 'is_done' from Gym does not take into account maximum number of steps in a single episode constraint
            self._time_step += 1
            if self._time_step % Runner.MEASURE_TIME_TIME_STEPS == 0:
                self._measure_time()

            rewards.append(steps[1][i])
            self._done_steps_in_a_episode[i] += 1
            is_done_gym = steps[2][i]
            is_maximum_number_of_steps_reached = self._max_steps_in_episode is not None \
                and self._max_steps_in_episode == self._done_steps_in_a_episode[i]

            is_done = is_done_gym and not is_maximum_number_of_steps_reached
            is_end = is_done or is_maximum_number_of_steps_reached

            reward = steps[1][i]
            experience.append(
                (actions[i], old_obs[i], reward, policies[i], is_done, is_end)
            )

            self._returns[i] += steps[1][i]
            self._rewards[i].append(steps[1][i])

            if is_end:
                self._done_episodes += 1

                logging.info(f"finished episode {self._done_episodes}, "
                             f"return: {self._returns[i]}, "
                             f"total time steps done: {self._time_step}")

                with tf.name_scope('rewards'):
                    tf.summary.histogram('rewards', self._rewards[i], self._done_episodes)
                    tf.summary.scalar('return', self._returns[i], self._done_episodes)
                    tf.summary.scalar('episode length', self._done_steps_in_a_episode[i], self._done_episodes)

                self._returns[i] = 0
                self._rewards[i] = []
                self._done_steps_in_a_episode[i] = 0

        self._current_obs = np.array(self._current_obs)

        with tf.name_scope('rewards'):
            tf.summary.scalar('mean_reward', np.mean(rewards), self._time_step)
            tf.summary.scalar('max_reward', np.max(rewards), self._time_step)
            tf.summary.scalar('min_reward', np.min(rewards), self._time_step)

        return experience

    def _evaluate(self):
        self._next_evaluation_timestamp += self._evaluate_time_steps_interval

        returns = [0] * self._num_evaluation_runs
        envs_finished = [False] * self._num_evaluation_runs
        time_step = 0
        current_obs = self._evaluate_env.reset()

        while not all(envs_finished):
            time_step += 1
            actions, _ = self._agent.predict_action(current_obs, is_deterministic=True)
            steps = self._evaluate_env.step(actions)
            current_obs = steps[0]
            for i in range(self._num_evaluation_runs):
                if not envs_finished[i]:
                    returns[i] += steps[1][i]

                    is_done_gym = steps[2][i]
                    is_maximum_number_of_steps_reached = self._max_steps_in_episode is not None\
                        and self._max_steps_in_episode == time_step

                    is_end = is_done_gym or is_maximum_number_of_steps_reached
                    envs_finished[i] = is_end
                    if is_end:
                        logging.info(f"evaluation run, "
                                     f"return: {returns[i]}")

        mean_returns = np.mean(returns)
        std_returns = np.std(returns)

        with tf.name_scope('rewards'):
            tf.summary.scalar('evaluation_return_mean', mean_returns, self._time_step)
            tf.summary.scalar('evaluation_return_std', std_returns, self._time_step)

        self._csv_logger.log_values(
            {'time_step': self._time_step, 'eval_return_mean': mean_returns, 'eval_std_mean': std_returns}
        )

    def record_video(self):
        logging.info(f"saving video of the current model performance...")

        env = wrappers.Monitor(gym.make(self._env_name), self._log_dir / 'video',
                               force=True, video_callable=lambda x: True)
        is_end = False
        time_step = 0
        current_obs = np.array([env.reset()])

        while not is_end:
            time_step += 1
            actions, _ = self._agent.predict_action(current_obs, is_deterministic=True)
            steps = env.step(actions[0])
            current_obs = np.array([steps[0]])
            is_done_gym = steps[2]
            is_maximum_number_of_steps_reached = self._max_steps_in_episode is not None\
                and self._max_steps_in_episode == time_step

            is_end = is_done_gym or is_maximum_number_of_steps_reached

        env.close()

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
        with open(str(self._log_dir / 'parameters.json'), 'wt') as f:
            json.dump(algorithm_parameters, f)

    def _save_checkpoint(self):
        """Saves current state and model"""
        checkpoint_dir = self._log_dir / 'checkpoint'
        checkpoint_dir.mkdir(exist_ok=True)

        runner_dump = {
            'time_step': self._time_step,
            'done_episodes': self._done_episodes,
        }
        with open(str(checkpoint_dir / 'runner.json'), 'wt') as f:
            json.dump(runner_dump, f)

        self._agent.save(checkpoint_dir / 'model')

        self._csv_logger.dump()
        logging.info(f"saved evaluation results")
        logging.info(f"saved checkpoint in '{str(checkpoint_dir)}'")
    #
    # def flush(self):
    #     """Dumps checkpoint and CSVLogger output to disk"""
    #     logging.info(f"flushing data to disk...")
    #     self._csv_logger.close()
    #     self._save_checkpoint()
    #     if self._save_video_on_kill:
    #         self._record_video()
