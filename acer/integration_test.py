
seed = 1
n_steps = 20000
save_ref = ()  # ('fastacer', 'fastacer+ISVB',)

old_layers = True
old_buffer = True
old_b = False

debug = False

import tensorflow as tf
import numpy as np
import pickle
import copy

with open('.git/HEAD', 'r') as f:
    githash = f.read().strip()
if githash.startswith('ref: '):
    gitref = githash.split()[1]
    with open('.git/' + gitref) as f:
        githash = f.read().strip()

print('git hash', githash)

if debug:
    tf.function = lambda x=None, *args, **kwargs: x if x else tf.function

import runners

runners.Runner.MEASURE_TIME_TIME_STEPS = n_steps + 1

get_env = runners._get_env
def get_seeded_env(*args, **kwargs):
    env = get_env(*args, **kwargs)
    env.seed(seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    return env
runners._get_env = get_seeded_env


values_list = [
    "obs", "actions", "policies", "base.mask", "actor.density", "actor.sample_weights", "actor.policies", "actor.optimize", "base.weighted_td", "critic.value", "critic.value_next", "critic.optimize"
]

base_params = {
    "asynchronous": False,
    "log_tensorboard": False,
    "log_dir": "test_logs",
    "num_parallel_envs": 1,
}

fastacer_params = {
    "environment_name": "HalfCheetahBulletEnv-v0",
    "algorithm": "fastacer",
    "algorithm_parameters": {
        "batches_per_env": 256,
        "gamma": 0.99,
        "actor_lr": 3e-5,
        "critic_lr": 1e-4,
        ("actor.layers" if not old_layers else 'actor_layers'): [256, 256],
        ("critic.layers" if not old_layers else 'critic_layers'): [256, 256],
        "buffer.n": 8,
        ("buffer.max_size" if not old_buffer else 'memory_size'): 1000000,
        ("actor.b" if not old_b else 'b'): 3,
        "actor.std": 0.4,
        "c": 1,
        "c0": 1,
        "learning_starts": -1,
        "actor_adam_beta1": 0.9,
        "actor_adam_beta2": 0.999,
        ("actor.beta_penalty" if not old_layers else "actor_beta_penalty"): 0.1,
        "num_parallel_envs": 1,
    },
}

fastacer_params.update(base_params)

isvb_params = copy.deepcopy(fastacer_params)
isvb_params["algorithm_parameters"].update({'learning_starts': 10000, 'buffer_type': 'is_dispersion_limiting', 'buffer.update_speed': 3, 'buffer.target_is_dispersion': 16})
isvb_values_list = values_list + ["memory.log_weights", "memory.is_dispersion", "memory.update_size_limit", "memory.is_dispersion", "memory.update_is_ref"]

"""
isvb_params["algorithm_parameters"] = {'gamma': 0.99,
  'lam': 0.9,
  'b': 3,
  'actor_adam_beta1': 0.9,
  'actor_adam_beta2': 0.999,
  'critic_adam_beta1': 0.9,
  'critic_adam_beta2': 0.999,
  'actor_lr': 3e-05,
  'critic_lr': 0.0001,
  'actor_beta_penalty' if old_layers else 'actor.beta_penalty': 0.1,
  'c': 1,
  'c0': 1,
  'std_diff_fun': 'KL',
  'keep_n': False,
  'learning_starts': 10000,
  'memory_size' if old_buffer else 'buffer.max_size': 1000000.0,
  'actor_layers' if old_layers else 'actor.layers': [256, 256],
  'critic_layers' if old_layers else 'critic.layers': [256, 256],
  'num_parallel_envs': 1,
  'batches_per_env': 256,
  'standardize_obs': False,
  'rescale_rewards': -1,
  'gradient_norm_median_threshold': 4,
  'use_v': False,
  'scale_td': False,
  'buffer_type': 'is_dispersion_limiting',
  'buffer.n': 8,
  'buffer.target_is_dispersion': 16.0,
  'buffer.update_speed': 3.0,
  'actor.std': 0.4,
  'actor.b': 3,
  'actor.limit_sustain_length': 8.0,
  'actor.modify_std': False,
  'actor.single_step_mask': False,
  'actor.scale_td': False,
  'actor.scale2_td2': False,
  'actor.no_clip_td2': False,
  'actor.clip_weighted': False,
  'actor.nn_std': False,
  'actor.single_std': False,
  'actor.mask_outliers': False,
  'actor.each_step_delay': False,
  'reverse': False,
  'no_window_adapt': False,
  'above_expected': False,
  'log_to_file_values': ['memory.update_size_limit'],
  'force_periodic_log': 1000,
  'nan_guard': False
}
"""

settings = (
    #  ('fastacer', 'test_references/fastacer.pickle', fastacer_params, values_list),
    ('fastacer+ISVB', 'test_references/fastacer_isvb.pickle', isvb_params, isvb_values_list),
)

for algo, ref, params, values_list  in settings:
    print(algo)
    params = copy.deepcopy(params)
    params['log_to_file_values'] = values_list
    params['algorithm_parameters']['log_to_file_values'] = values_list

    tf.random.set_seed(seed)
    np.random.seed(seed)

    runner = runners.Runner(**params)
    # runner._env.seed(seed)
    # runner._env.action_space.seed(seed)
    # runner._env.observation_space.seed(seed)


    for _ in range(n_steps):
        init_vars = None

        experience, _ = runner._step()

        if init_vars is None:
            init_vars = (
                *[v.numpy() for v in runner._agent._actor.trainable_variables],
                *[v.numpy() for v in runner._agent._critic.trainable_variables]
            )

        runner._agent.save_experience(experience)
        values = runner._agent.learn()
    values = [v.numpy() for v in values]

    step_vars = (
        *[v.numpy() for v in runner._agent._actor.trainable_variables],
        *[v.numpy() for v in runner._agent._critic.trainable_variables]
    )

    if algo in save_ref:
        with open(ref, 'wb') as rf:
            pickle.dump((githash, init_vars, step_vars, values), rf)

    else:
        with open(ref, 'rb') as rf:
            githash_ref, ref_init_vars, ref_step_vars, ref_values = pickle.load(rf)
        for ref_vars, vars, name in (ref_init_vars, init_vars, 'init'), (ref_step_vars, step_vars, 'step'):
            if not all((v1 == v2).all() for v1, v2 in zip(ref_vars, vars)):
                print('>>>>>>', 'Difference in', name)
            else:
                print('>>>>>>', name, 'OK')
        
        values_diff = []
        for val, ref_val, name in zip(values, ref_values, values_list):
            if not np.all(val == ref_val):
                values_diff.append(name)
        if values_diff:
            print('>>>>>>', 'Difference in values:', *values_diff)
        else:
            print('>>>>>>', 'No difference in values')

    print()
