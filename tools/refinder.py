"""
Sample usage:

tools/refinder.py --logdir logs --algo qacer --env Pendulum-v0 --diff --params kappa=1.0 critic_lr=0.05
"""
import argparse
import json
import os
import csv


def getDTChangedEnvName(base_env_name, timesteps_increase):
    return str.join('TS' + str(timesteps_increase) + '-', base_env_name.split('-'))


def check_name(fname, algo, env, ts):
    ret = False

    if not ts or ts == 1:
        ret = fname.startswith(f'{env}_{algo}_')
    if ts:
        dt_changed_env = getDTChangedEnvName(env, ts)
        ret = ret or fname.startswith(f'{dt_changed_env}_{algo}_')
    
    return ret


def load_params(params_file_path):
    with open(params_file_path, mode='r') as file:
        return json.load(file)


def load_results(results_file_path):
    with open(results_file_path, mode='r') as file:
        return list(csv.DictReader(file))


def check_params(params, required):
    return all([params[k] == v for k, v in required.items()]) or not params


def get_results_props(results):
    if len(results) == 0:
        return { "time_steps": 0, "last_eval_return_mean": None, }
    return {
        "time_steps": max(int(result['time_step']) for result in results),
        "last_eval_return_mean": [float(result['eval_return_mean']) for result in results][-1],
    }


def check_results(results_props, steps):
    return not steps or results_props['time_steps'] >= steps


def find_results(logdir, env, algo, ts, steps, params):
    matches = []
    all_runs_params = []
    all_results_props = []

    for f in os.listdir(logdir):
        fpath = os.path.join(logdir, f)
        params_path = os.path.join(fpath, 'parameters.json')
        results_path = os.path.join(fpath, 'results.csv')
        if os.path.isdir(fpath) and os.path.isfile(params_path) and os.path.isfile(results_path):
            run_params = load_params(params_path)
            results_props = get_results_props(load_results(results_path))
            if check_name(f, algo, env, ts) and check_params(run_params, params) and check_results(results_props, steps):
                matches.append(f)
                all_runs_params.append(run_params)
                all_results_props.append(results_props)
    
    return matches, all_runs_params, all_results_props


def parse_params(params):
    out = {}

    for p in params:
        k, v = p.split('=')
        out[k] = eval(v)
    
    return out


def make_diffs(params):
    keys = set().union(*(p.keys() for p in params))
    diff_keys = set()
    for key in keys:
        values = {str(p.get(key, None)) for p in params}
        if len(values) > 1:
            diff_keys.add(key)

    return [{k: p.get(k, None) for k in diff_keys} for p in params]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir', type=str, required=True)
    parser.add_argument('--env', type=str, required=True)
    parser.add_argument('--algo', type=str, required=True)
    parser.add_argument('--ts', type=int, required=False)
    parser.add_argument('--steps', type=int, required=False)
    parser.add_argument('--diff', action='store_true', default=False)
    parser.add_argument('--results', action='store_true', default=False)
    parser.add_argument('--params', type=str, nargs='*', default=[])
    
    args = parser.parse_args()

    params = parse_params(args.params)

    results, runs_params, all_res_props = find_results(args.logdir, args.env, args.algo, args.ts, args.steps, params)

    diffs = make_diffs(runs_params) if args.diff else [{}] * len(results)
    res_props = all_res_props if args.results else [{}] * len(results)

    for res, r, d in zip(results, res_props, diffs):
        print(
            res,
            *[f'{k}={v}' for k, v in r.items()],
            *[f'{k}={v}' for k, v in d.items()],
            sep='\t'
        )
