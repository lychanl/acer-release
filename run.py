import argparse

import numpy as np
from acer import ACER
from environment import SequentialEnv
from utils import get_env_variables, reset_env_and_agent

# handle command line arguments
parser = argparse.ArgumentParser(description='Actor-Critic with experience replay.')
parser.add_argument('--env_name', type=str, help='OpenAI Gym environment name', default="CartPole-v0")
parser.add_argument('--runs_number', type=int, help='Number of independent agent\'s runs', default=1)
parser.add_argument('--max_episodes', type=int, help='Maximum number of episodes in each run.', default=None)
parser.add_argument('--output_dir', type=str, help='Output directory for rewards files', default="./")
# parser.add_argument('--use_gpu', help="Specify to use GPU", action='store_true')
parser.add_argument('--render', help="Specify to render frames", action='store_true')
parser.add_argument('--gamma', type=float, help='discount factor', required=False)
parser.add_argument('--alpha', type=float, help='alpha value (see from Actor-Critic(lambda) algorithm)', required=False)
parser.add_argument('--p', type=float, help='geometric probability distribution parameter (prob. of failure)',
                    required=False)
parser.add_argument('--b', type=float, help='probability density truncation parameter', required=False)
parser.add_argument('--actor_epsilon', type=float, help='ADAM optimizer epsilon parameter for Actor',
                    required=False)
parser.add_argument('--actor_beta1', type=float, help='ADAM optimizer beta1 parameter for Actor', required=False)
parser.add_argument('--actor_beta2', type=float, help='ADAM optimizer beta2 parameter for Actor', required=False)
parser.add_argument('--critic_epsilon', type=float, help='ADAM optimizer epsilon parameter for Critic',
                    required=False)
parser.add_argument('--critic_beta1', type=float, help='ADAM optimizer beta1 parameter for Critic', required=False)
parser.add_argument('--critic_beta2', type=float, help='ADAM optimizer beta2 parameter for Critic', required=False)
parser.add_argument('--actor_lr', type=float, help='Actor learning rate', required=False)
parser.add_argument('--critic_lr', type=float, help='Critic learning rate', required=False)
parser.add_argument('--actor_beta_penalty', type=float, help='Actor penalty factor', required=False)
parser.add_argument('--c', type=int, help='experience replay intensity', required=False)
parser.add_argument('--memory_size', type=int, help='Memory buffer size', required=False)
parser.add_argument('--min_steps_learn', type=float, help='initial experience replay intensity factor', required=False)
parser.add_argument('--actor_layers', nargs='+', type=int, help='Sizes of Actor\'s network hidden layers',
                    required=False)
parser.add_argument('--critic_layers', nargs='+', type=int, help='Sizes of Critic\'s network hidden layers',
                    required=False)
parser.add_argument('--std', nargs='+', type=float, help='Covariance diagonal',
                    required=False, default=[1.0])
parser.add_argument('--num_parallel_envs', type=int, default=4,
                    required=True)


def run_acer(env, parameters, runs_number=10, max_episodes=200, max_steps_in_episode=None):
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
        min_steps_learn=parameters['min_steps_learn'],
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

    for current_run_number in range(0, runs_number):

        print("STARTED RUN {}".format(current_run_number))

        cumulated_rewards = [0] * parameters['num_parallel_envs']
        episode_num = 1
        done_steps_in_episode = [0] * parameters['num_parallel_envs']

        current_states = reset_env_and_agent(agent, env)
        while True:
            for i in range(0, len(current_states)):
                done_steps_in_episode[i] += 1

            actions, policies = agent.predict_action(current_states)
            steps = env.step(actions)
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

                    print("episode {}/{}; cumulated reward: {}, timestep: {}".format(episode_num, max_episodes,
                                                                                     cumulated_rewards[i], n_steps))

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
    run_acer(env, parameters, args.runs_number, args.max_episodes, max_steps_in_episode)


if __name__ == "__main__":
    main()
