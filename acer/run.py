import argparse
import datetime

import numpy as np
import tensorflow as tf

from environment import SequentialEnv
from models import ACER
from utils import get_env_variables, reset_env_and_agent

# handle command line arguments
parser = argparse.ArgumentParser(description='Actor-Critic with experience replay.')
parser.add_argument('--env_name', type=str, help='OpenAI Gym environment name', default="CartPole-v0")
# parser.add_argument('--runs_number', type=int, help='number of independent agent\'s runs', default=1)
# parser.add_argument('--max_episodes', type=int, help='maximum number of episodes in a single run', default=None)
# parser.add_argument('--output_dir', type=str, help='rewards log output directory', default="./")
# parser.add_argument('--render', help="true if frames should be rendered", action='store_true')
parser.add_argument('--gamma', type=float, help='discount factor', required=False)
parser.add_argument('--alpha', type=float, help='alpha value coefficient', required=False)
parser.add_argument('--p', type=float, help='prob. of success in geometric probability distribution, used to'
                                            'sample trajectory length while sampling from the buffer',
                    required=False)
parser.add_argument('--b', type=float, help='probability density truncation coefficient', required=False)
parser.add_argument('--actor_epsilon', type=float, help='ADAM optimizer epsilon parameter (Actor)',
                    required=False)
parser.add_argument('--actor_beta1', type=float, help='ADAM optimizer beta1 (Actor)', required=False)
parser.add_argument('--actor_beta2', type=float, help='ADAM optimizer beta2 (Actor)', required=False)
parser.add_argument('--critic_epsilon', type=float, help='ADAM optimizer epsilon (Critic)',
                    required=False)
parser.add_argument('--critic_beta1', type=float, help='ADAM optimizer beta1 (Critic)', required=False)
parser.add_argument('--critic_beta2', type=float, help='ADAM optimizer beta2 (Critic)', required=False)
parser.add_argument('--actor_lr', type=float, help='Actor learning rate', required=False)
parser.add_argument('--critic_lr', type=float, help='Critic learning rate', required=False)
parser.add_argument('--actor_beta_penalty', type=float, help='Actor penalty coefficient', default=0.1)
parser.add_argument('--c', type=int, help='experience replay intensity', required=False)
parser.add_argument('--c0', type=float, help='experience replay warm start coefficient', default=0.3)
parser.add_argument('--memory_size', type=int, help='memory buffer size', required=False)
parser.add_argument('--actor_layers', nargs='+', type=int, help='List of Actor\'s neural network hidden layers sizes',
                    required=False)
parser.add_argument('--critic_layers', nargs='+', type=int, help='List of Critic\'s neural network hidden layers sizes',
                    required=False)
parser.add_argument('--std', nargs='+', type=float, help='Actor\'s covariance diagonal (Gaussian policy)',
                    required=False, default=[1.0])
parser.add_argument('--num_parallel_envs', type=int, help='Number of environments to be run in a parallel', default=4,
                    required=True)


def run_acer(env, parameters, max_steps_in_episode=None):
    log_dir = "logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_writer = tf.summary.create_file_writer(log_dir)
    tensorboard_writer.set_as_default()

    agent = ACER(
        observations_dim=parameters['observations_dim'],
        actions_dim=parameters['actions_dim'],
        actions_bound=parameters['actions_bound'],
        actor_layers=parameters['actor_layers'],
        critic_layers=parameters['critic_layers'],
        num_parallel_envs=parameters['num_parallel_envs'],
        is_discrete=not parameters['continuous'],
        gamma=parameters['gamma'],
        memory_size=parameters['memory_size'],
        alpha=parameters['alpha'],
        p=parameters['p'],
        b=parameters['b'],
        c=parameters['c'],
        c0=parameters['c0'],
        actor_lr=parameters['actor_lr'],
        actor_beta_penalty=parameters['actor_beta_penalty'],
        actor_adam_beta1=parameters['actor_beta1'],
        actor_adam_beta2=parameters['actor_beta2'],
        actor_adam_epsilon=parameters['actor_epsilon'],
        critic_lr=parameters['critic_lr'],
        critic_adam_beta1=parameters['critic_beta1'],
        critic_adam_beta2=parameters['critic_beta2'],
        critic_adam_epsilon=parameters['critic_epsilon'],
        std=parameters['std']
    )

    n_steps = 0

    cumulated_rewards = [0] * parameters['num_parallel_envs']
    episode_num = 1
    done_steps_in_episode = [0] * parameters['num_parallel_envs']

    current_states = reset_env_and_agent(agent, env)
    while True:
        for i in range(0, len(current_states)):
            done_steps_in_episode[i] += 1
        actions, policies = agent.predict_action(current_states)
        steps = env.step(actions)

        with tf.name_scope('env'):
            rewards = [step[1] for step in steps]
            tf.summary.scalar('mean_reward', np.mean(rewards), n_steps)
            tf.summary.scalar('max_reward', np.max(rewards), n_steps)
            tf.summary.scalar('min_reward', np.min(rewards), n_steps)

        n_steps += parameters['num_parallel_envs']

        exp = []
        for i, step in enumerate(steps):
            is_done = step[2] and (not max_steps_in_episode or max_steps_in_episode != done_steps_in_episode[i])
            exp.append((actions[i], current_states[i], step[1], step[0], policies[i], is_done,
                        step[2] or max_steps_in_episode == done_steps_in_episode[i]))

        agent.save_experience(exp)
        agent.learn()

        next_states = np.array([step[0] for step in steps])
        current_states = next_states
        rewards = [step[1] for step in steps]

        for i in range(0, len(current_states)):
            cumulated_rewards[i] += rewards[i]
            if steps[i][2] or (max_steps_in_episode and max_steps_in_episode == done_steps_in_episode[i]):
                current_states[i] = env.reset(i)

                print(
                    "episode {}; return: {}, time step: {}".format(episode_num, cumulated_rewards[i], n_steps)
                )

                with tf.name_scope('env'):
                    tf.summary.scalar('return', cumulated_rewards[i], episode_num)
                cumulated_rewards[i] = 0
                done_steps_in_episode[i] = 0

                episode_num += 1


def main():
    args = parser.parse_args()
    env = SequentialEnv(args.env_name, args.num_parallel_envs)

    action_scale, actions_dim, observations_dim, continuous, max_steps_in_episode = get_env_variables(env)

    parameters = {"observations_dim": observations_dim,
                  "actions_dim": actions_dim,
                  "actions_bound": action_scale,
                  "continuous": continuous}

    cmd_parameters, unknown_args = parser.parse_known_args()
    if len(unknown_args):
        print("Not recognized arguments: ", str(vars(unknown_args)))
        return

    parameters.update(vars(cmd_parameters))

    # remove empty values
    parameters = {k: v for k, v in parameters.items() if v is not None}
    print(parameters)
    run_acer(env, parameters, max_steps_in_episode)


if __name__ == "__main__":
    main()
