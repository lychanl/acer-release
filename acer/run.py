import argparse

# handle command line arguments
from runners import Runner

parser = argparse.ArgumentParser(description='Actor-Critic with experience replay.')
parser.add_argument('--env_name', type=str, help='OpenAI Gym environment name', default="CartPole-v0")
parser.add_argument('--gamma', type=float, help='discount factor', required=False)
parser.add_argument('--alpha', type=float, help='alpha value coefficient', required=False)
parser.add_argument('--p', type=float, help='prob. of success in geometric probability distribution, used to'
                                            'sample trajectory length while sampling from the buffer',
                    required=False)
parser.add_argument('--b', type=float, help='probability density truncation coefficient', required=False)
parser.add_argument('--actor_adam_epsilon', type=float, help='ADAM optimizer epsilon parameter (Actor)',
                    required=False)
parser.add_argument('--actor_adam_beta1', type=float, help='ADAM optimizer beta1 (Actor)', required=False)
parser.add_argument('--actor_adam_beta2', type=float, help='ADAM optimizer beta2 (Actor)', required=False)
parser.add_argument('--critic_adam_epsilon', type=float, help='ADAM optimizer epsilon (Critic)',
                    required=False)
parser.add_argument('--critic_adam_beta1', type=float, help='ADAM optimizer beta1 (Critic)', required=False)
parser.add_argument('--critic_adam_beta2', type=float, help='ADAM optimizer beta2 (Critic)', required=False)
parser.add_argument('--actor_lr', type=float, help='Actor learning rate', required=False)
parser.add_argument('--critic_lr', type=float, help='Critic learning rate', required=False)
parser.add_argument('--actor_beta_penalty', type=float, help='Actor penalty coefficient', default=0.1)
parser.add_argument('--c', type=int, help='experience replay intensity', required=False)
parser.add_argument('--c0', type=float, help='experience replay warm start coefficient', default=0.3)
parser.add_argument('--std', type=float, help='value on diagonal of Normal dist. covariance matrix. If not specified,'
                                              '0.4 * actions_bound is set.',
                    required=False)
parser.add_argument('--memory_size', type=int, help='memory buffer size (sum of all of the buffers from every env',
                    required=False)
parser.add_argument('--actor_layers', nargs='+', type=int, help='List of Actor\'s neural network hidden layers sizes',
                    required=False)
parser.add_argument('--critic_layers', nargs='+', type=int, help='List of Critic\'s neural network hidden layers sizes',
                    required=False)
parser.add_argument('--num_parallel_envs', type=int, help='Number of environments to be run in a parallel', default=4,
                    required=True)
parser.add_argument('--batches_per_env', type=int, help='Number of batches sampled from one environment buffer in one'
                                                        'backward pass',
                    default=5)
parser.add_argument('--standardize_obs', help='True, if observations should be standarized online'
                                              ' (and clipped between -5, 5)',
                    action='store_true')
parser.add_argument('--rescale_rewards', help='-1 to turn rescaling off, 0 to rescale automatically based on'
                                              'running variance, value greater than rescales the rewards in the way'
                                              'that they are divided by that value',
                    type=int, default=-1)
parser.add_argument('--evaluate_time_steps_interval', type=int, help='Number of time steps between evaluations. '
                                                                     '-1 to turn evaluation off',
                    default=3000)
parser.add_argument('--num_evaluation_runs', type=int, help='Number of evaluation runs in a single evaluation',
                    default=10)
parser.add_argument('--max_time_steps', type=int, help='Maximum number of time steps of agent learning. -1 means no '
                                                       'time steps limit',
                    default=-1)
parser.add_argument('--log_dir', type=str, help='TensorBoard logging directory', default='logs/')


def main():
    args = parser.parse_args()

    cmd_parameters, unknown_args = parser.parse_known_args()
    if len(unknown_args):
        print("Not recognized arguments: ", str(vars(unknown_args)))
        return

    parameters = {k: v for k, v in vars(cmd_parameters).items() if v is not None}
    parameters.pop('env_name')
    evaluate_time_steps_interval = parameters.pop('evaluate_time_steps_interval')
    num_evaluation_runs = parameters.pop('num_evaluation_runs')
    max_time_steps = parameters.pop('max_time_steps')
    log_dir = parameters.pop('log_dir')

    runner = Runner(
        environment_name=cmd_parameters.env_name,
        algorithm='acer',
        algorithm_parameters=parameters,
        num_parallel_envs=cmd_parameters.num_parallel_envs,
        log_dir=log_dir,
        max_time_steps=max_time_steps,
        num_evaluation_runs=num_evaluation_runs,
        evaluate_time_steps_interval=evaluate_time_steps_interval
    )
    runner.run()


if __name__ == "__main__":
    main()
