import argparse
import signal

from sys import argv
import gym
from numpy import float32

import tensorflow as tf

if '--debug' in argv:
    tf.function = lambda x=None, *args, **kwargs: x if x else tf.function

from runners import Runner, ALGOS, LEGACY_ALGOS
from utils import calculate_gamma, getDTChangedEnvName

from algos.legacy.acerac import AUTOCORRELATED_ACTORS


def prepare_legacy_parser(parser):
    parser.add_argument('--gamma', type=float, help='discount factor', required=False, default=0.99)
    parser.add_argument('--lam', type=float, help='lambda parameter', required=False, default=0.9)
    parser.add_argument('--b', type=float, help='probability density truncation coefficient',
                        required=False, default=3)
    parser.add_argument('--actor_adam_epsilon', type=float, help='ADAM optimizer epsilon parameter (BaseActor)',
                        required=False, default=None)
    parser.add_argument('--actor_adam_beta1', type=float, help='ADAM optimizer beta1 (BaseActor)',
                        required=False, default=0.9)
    parser.add_argument('--actor_adam_beta2', type=float, help='ADAM optimizer beta2 (BaseActor)',
                        required=False, default=0.999)
    parser.add_argument('--critic_adam_epsilon', type=float, help='ADAM optimizer epsilon (Critic)',
                        required=False, default=None)
    parser.add_argument('--critic_adam_beta1', type=float, help='ADAM optimizer beta1 (Critic)',
                        required=False, default=0.9)
    parser.add_argument('--critic_adam_beta2', type=float, help='ADAM optimizer beta2 (Critic)',
                        required=False, default=0.999)
    parser.add_argument('--actor_lr', type=float, help='BaseActor learning rate', required=False, default=3e-5)
    parser.add_argument('--critic_lr', type=float, help='Critic learning rate', required=False, default=1e-4)
    parser.add_argument('--actor_beta_penalty', type=float, help='BaseActor penalty coefficient', default=0.1)
    parser.add_argument('--c', type=int, help='experience replay intensity', required=False, default=1)
    parser.add_argument('--c0', type=float, help='experience replay warm start coefficient', default=1)
    parser.add_argument('--kappa', type=float, help='kappa parameter for qacer and time scaling')
    parser.add_argument('--atoms', type=int, help='number of atoms for qacer')
    parser.add_argument('--rho', type=float, help='Rho parameter.')
    parser.add_argument('--std_loss_mult', type=float, help='std loss to actor loss ration')
    parser.add_argument('--std_loss_delay', type=float, help='Delay of std optimization.')
    parser.add_argument('--diff_b', type=float, help='diff ratio truncation coefficient',
                        required=False, default=None)
    parser.add_argument('--diff_h', type=float, help='H scale for log std loss',
                        required=False, default=None)
    parser.add_argument('--entropy_coeff', type=float, help='Entropy coefficient for ExplorACER')
    parser.add_argument('--tau', type=int, help='Tau parameter for acerac')
    parser.add_argument('--n', type=int, help='N parameter for fast acerac')
    parser.add_argument('--keep_n', action='store_true', help='Do not autoadapt n', default=False)
    parser.add_argument('--noise_type', type=str, help='Type of noise for ACERAC',
                        choices=list(AUTOCORRELATED_ACTORS))
    parser.add_argument('--std', type=float, help='value on diagonal of Normal dist. covariance matrix. If not specified,'
                                                '0.4 * actions_bound is set.',
                        required=False, default=None)
    parser.add_argument('--learning_starts', type=int, help='experience replay warm start coefficient', default=10000)
    parser.add_argument('--memory_size', type=int, help='memory buffer size (sum of all of the buffers from every env',
                        required=False, default=1e6)
    parser.add_argument('--actor_layers', nargs='+', type=int, help='List of BaseActor\'s neural network hidden layers sizes',
                        required=False, default=(256, 256))
    parser.add_argument('--critic_layers', nargs='+', type=int, help='List of Critic\'s neural network hidden layers sizes',
                        required=False, default=(256, 256))
    parser.add_argument('--batches_per_env', type=int, help='Number of batches sampled from one environment buffer in one'
                                                            'backward pass',
                        default=5)
    parser.add_argument('--standardize_obs', help='True, if observations should be standarized online'
                                                ' (and clipped between -5, 5)',
                        action='store_true')
    parser.add_argument('--rescale_rewards', help='-1 to turn rescaling off, 0 to rescale automatically based on'
                                                'running variance; value greater than 0 rescales the rewards by'
                                                'dividing them by the value',
                        type=float, default=-1)
    parser.add_argument('--limit_reward_tanh', help='limits reward to [-value, value] using tanh function'
                                                    '0 to disable',
                        type=float, default=None)
    parser.add_argument('--td_clip', help='Temporal difference clipping threshold (ACERAC only)',
                        type=float, default=None)
    parser.add_argument('--gradient_norm', help='Global gradient clip norm, 0 to use learned median of the gradient',
                        type=float, default=None)
    parser.add_argument('--gradient_norm_median_threshold', help='Number of medians used to clip gradients by their norm',
                        type=float, default=4)
    parser.add_argument('--use_v', action='store_true',
                        help='If true then value instead of noise-value will be used (ACERAC only)')
    parser.add_argument('--alpha', type=float, help='Alpha parameter.')
    parser.add_argument('--reverse', action='store_true',
                        help='Reverse param for exploracer')
    parser.add_argument('--time_coeff', type=str, help='type of time-based coefficient for sigma learning', choices=("none", "linear", "exp", "power"))
    parser.add_argument('--no_window_adapt', help='don\'t adapt window size of time-based coefficient for sigma learning', action='store_true')
    parser.add_argument('--above_expected', help='adapt std to samples with positive adv only', action='store_true')


def prepare_parser():
    parser = argparse.ArgumentParser(description='BaseActor-Critic with experience replay.')
    parser.add_argument('--defaults', type=str, default=None)
    args, _ = parser.parse_known_args()

    defaults = []
    if args.defaults:
        with open(args.defaults) as defaults_file:
            defaults = defaults_file.read().split()

    parser.add_argument('--algo', type=str, help='Algorithm to be used', default="fastacer", choices=ALGOS)
    parser.add_argument('--env', type=str, help='OpenAI Gym environment name', default="HalfCheetahBulletEnv-v0")

    parser.add_argument('--evaluate_time_steps_interval', type=int, help='Number of time steps between evaluations. '
                                                                        '-1 to turn evaluation off',
                        default=10000)
    parser.add_argument('--num_evaluation_runs', type=int, help='Number of evaluation runs in a single evaluation',
                        default=10)
    parser.add_argument('--max_time_steps', type=int, help='Maximum number of time steps of agent learning. -1 means no '
                                                        'time steps limit',
                        default=-1)
    parser.add_argument('--log_dir', type=str, help='Logging directory', default='logs/')
    parser.add_argument('--no_checkpoint', help='Disable checkpoint saving', action='store_true')
    parser.add_argument('--no_tensorboard', help='Disable tensorboard logs', action='store_true')
    parser.add_argument('--log_values', help='Log values during training', type=str, nargs='*')
    parser.add_argument('--log_act_values', help='Log values during action prediction', type=str, nargs='*')
    parser.add_argument('--log_memory_values', help='Log values during training during memory update step (if any)', type=str, nargs='*')
    parser.add_argument('--log_to_file_values', help='Log values during training to a file every n steps, averaged over these steps', type=str, nargs='*')
    parser.add_argument('--log_to_file_memory_values', help='Log values during training to a file every n steps, averaged over these steps', type=str, nargs='*')
    parser.add_argument('--log_to_file_act_values', help='Log values during training to a file every n steps, averaged over these steps', type=str, nargs='*')
    parser.add_argument('--log_to_file_steps', help='Log values during training to a file every n steps, averaged over these steps', type=int, default=1000)
    parser.add_argument('--experiment_name', type=str, help='Name of the current experiment', default='')
    parser.add_argument('--save_video_on_kill', action='store_true',
                        help='True if SIGINT signal should trigger registration of the video')
    parser.add_argument('--record_time_steps', type=int, default=None,
                        help='Number of time steps between evaluation video recordings')
    parser.add_argument('--use_cpu', action='store_true',
                        help='True if CPU (instead of GPU) should be used')
    parser.add_argument('--synchronous', action='store_true',
                        help='True if not use asynchronous envs')
    parser.add_argument('--timesteps_increase', help='Timesteps per second increase. Affects gamma and max time steps', type=int, default=None)

    parser.add_argument('--dump', help='Dump memory and models on given timesteps', nargs='*', type=int)
    parser.add_argument('--debug', help='Disable tf functions', action='store_true')
    parser.add_argument('--force_periodic_log', help='Force logging every n timesteps instead of on episode finish etc', type=int, default=0)
    parser.add_argument('--n_step', type=int, help='experience replay frequency', required=False, default=1)
    parser.add_argument('--num_parallel_envs', type=int, help='Number of environments to be run in a parallel', default=1)
    default_args_partial, _ = parser.parse_known_args(defaults)
    parser.set_defaults(**default_args_partial.__dict__)
    partial_args, unparsed = parser.parse_known_args()

    if partial_args.algo in LEGACY_ALGOS:
        prepare_legacy_parser(parser)
    else:
        sample_env = gym.make(partial_args.env)
        ALGOS[partial_args.algo].prepare_parser(parser, sample_env, unparsed)
    try:
        default_args = parser.parse_args(defaults)
    except Exception as e:
        raise Exception("An exception occured while parsing defaults file") from e

    parser.set_defaults(**default_args.__dict__)
    return parser


def main():
    parser = prepare_parser()
    args = parser.parse_args()
    
    if args.algo not in ('acer', 'acerac', 'qacer', 'exploracer', 'qacerac', 'quantile_acer'):
        for old, new in [
            ('--n', '--buffer.n'),
            ('--std', '--actor.std'),
            ('--b', '--actor.b')
        ]:
            if old in argv:
                raise ValueError(f"Use {new} instead of {old}!")

    parameters = {k: v for k, v in args.__dict__.items() if v is not None}
    parameters.pop('env')
    evaluate_time_steps_interval = parameters.pop('evaluate_time_steps_interval')
    n_step = parameters.pop('n_step')
    num_evaluation_runs = parameters.pop('num_evaluation_runs')
    max_time_steps = parameters.pop('max_time_steps')
    save_video_on_kill = parameters.pop('save_video_on_kill')
    no_checkpoint = parameters.pop('no_checkpoint')
    no_tensorboard = parameters.pop('no_tensorboard')
    record_time_steps = parameters.pop('record_time_steps', None)
    experiment_name = parameters.pop('experiment_name')
    algorithm = parameters.pop('algo')
    log_dir = parameters.pop('log_dir')
    use_cpu = parameters.pop('use_cpu')
    synchronous = parameters.pop('synchronous')
    env_name = args.env
    log_to_file_values = args.log_to_file_values
    log_to_file_memory_values = args.log_to_file_memory_values
    log_to_file_act_values = args.log_to_file_act_values
    log_to_file_steps = parameters.pop('log_to_file_steps')
    debug = parameters.pop('debug')
    dump = args.dump or ()

    timesteps_increase = parameters.pop('timesteps_increase', None)
    if timesteps_increase and timesteps_increase != 1:
        parameters['gamma'] = calculate_gamma(parameters['gamma'], timesteps_increase)
        print(f'Auto-adapted gamma to {parameters["gamma"]}')
        parameters['lam'] = (1 - (1 - parameters['lam']) / timesteps_increase)
        print(f'Auto-adapted lam to {parameters["lam"]}')
        parameters['memory_size'] = parameters['memory_size'] * timesteps_increase
        print(f'Auto-adapted memory_size to {parameters["memory_size"]}')
        parameters['learning_starts'] = parameters['learning_starts'] * timesteps_increase
        print(f'Auto-adapted learning_starts to {parameters["learning_starts"]}')
        if not parameters['keep_n']:
            parameters['n'] = parameters['n'] * timesteps_increase
            print(f'Auto-adapted n to {parameters["n"]}')
        if 'alpha' in parameters:
            parameters['alpha'] = parameters['alpha'] ** (1 / timesteps_increase)
            print(f'Auto-adapted alpha to {parameters["alpha"]}')
        env_name = getDTChangedEnvName(env_name, timesteps_increase)
        max_time_steps *= timesteps_increase
        print(f'Auto-adapted max_time_steps to {max_time_steps}')
        evaluate_time_steps_interval *= timesteps_increase
        print(f'Auto-adapted evaluate_time_steps_interval to {evaluate_time_steps_interval}')
        n_step = n_step * timesteps_increase
        print(f'Auto-adapted n_step to {n_step}')

    if experiment_name:
        for param, value in parameters.items():
            experiment_name = experiment_name.replace(f'{{{param}}}', str(value))

    if use_cpu:
        tf.config.set_visible_devices([], 'GPU')

    runner = Runner(
        environment_name=env_name,
        algorithm=algorithm,
        algorithm_parameters=parameters,
        num_parallel_envs=args.num_parallel_envs,
        log_dir=log_dir,
        max_time_steps=max_time_steps,
        num_evaluation_runs=num_evaluation_runs,
        evaluate_time_steps_interval=evaluate_time_steps_interval,
        experiment_name=experiment_name,
        asynchronous=not synchronous,
        log_tensorboard=not no_tensorboard,
        do_checkpoint=not no_checkpoint,
        record_time_steps=record_time_steps,
        n_step=n_step,
        dump=dump,
        periodic_log=args.force_periodic_log,
        log_to_file_values=log_to_file_values,
        log_to_file_act_values=log_to_file_act_values,
        log_to_file_steps=log_to_file_steps,
        log_to_file_memory_values=log_to_file_memory_values,
        debug=debug
    )

    def handle_sigint(sig, frame):
        runner.record_video()

    if save_video_on_kill:
        signal.signal(signal.SIGINT, handle_sigint)

    runner.run()


if __name__ == "__main__":
    main()
