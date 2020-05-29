"""
Sample usage:

tools/refinder.py --logdir logs --algo qacer --env Pendulum-v0 --diff --params kappa=1.0 critic_lr=0.05
"""
import argparse
import json
import os


def check_name(fname, algo, env):
    return fname.startswith(f'{env}_{algo}_')


def load_params(params_file_path):
    with open(params_file_path, mode='r') as file:
        return json.load(file)


def check_params(params, required):
    return all([params[k] == v for k, v in required.items()])

def find_results(logdir, env, algo, params):
    matches = []
    all_runs_params = []

    for f in os.listdir(logdir):
        fpath = os.path.join(logdir, f)
        params_path = os.path.join(fpath, 'parameters.json')
        if os.path.isdir(fpath) and os.path.isfile(params_path):
            run_params = load_params(params_path)
            if check_name(f, algo, env) and check_params(run_params, params):
                matches.append(f)
                all_runs_params.append(run_params)
    
    return matches, all_runs_params


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
    parser.add_argument('--diff', action='store_true', default=False)
    parser.add_argument('--params', type=str, nargs='*', default=[])
    
    args = parser.parse_args()

    params = parse_params(args.params)

    results, runs_params = find_results(args.logdir, args.env, args.algo, params)

    diffs = {}
    if args.diff:
        diffs = make_diffs(runs_params)

    for res, d in zip(results, diffs):
        print(res, *[f'{k}={v}' for k, v in d.items()])
