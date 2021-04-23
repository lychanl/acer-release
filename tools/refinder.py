"""
Sample usage:

tools/refinder.py --logdir logs --algo qacer --env Pendulum-v0 --diff --params kappa=1.0 critic_lr=0.05
"""
import argparse
import json
import os
import csv
import sys


class Refinder:
    def __init__(self, return_col, timestep_col, sep):
        self.return_col = return_col
        self.timestep_col = timestep_col
        self.sep = sep


    def getDTChangedEnvName(self, base_env_name, timesteps_increase):
        return str.join('TS' + str(timesteps_increase) + '-', base_env_name.split('-'))


    def check_name(self, fname, algo, env, ts):
        ret = False

        if not ts or ts == 1:
            ret = fname.startswith(f'{env}_{algo}_')
        if ts:
            dt_changed_env = self.getDTChangedEnvName(env, ts)
            ret = ret or fname.startswith(f'{dt_changed_env}_{algo}_')
        
        return ret


    def load_params(self, params_file_path):
        with open(params_file_path, mode='r') as file:
            return json.load(file)


    def load_results(self, results_file_path):
        with open(results_file_path, mode='r') as file:
            return list(csv.DictReader(file, delimiter=self.sep))


    def check_param(self, params, param, value):
        if '.' in param:
            parts = param.split('.')
            if parts[0] in params:
                return self.check_param(params[parts[0]], '.'.join(parts[1:]), value)
            else:
                return value is None
        else:
            return params.get(param, None) == value


    def check_params(self, params, required):
        return all(self.check_param(params, k, v) for k, v in required.items()) or not params


    def get_results_props(self, results):
        if len(results) == 0:
            return { "time_steps": 0, "last_eval_return_mean": None, }
        return {
            "time_steps": max(int(result[self.timestep_col]) for result in results),
            "last_eval_return_mean": [float(result[self.return_col]) for result in results][-1],
        }


    def check_results(self, results_props, steps):
        return not steps or results_props['time_steps'] >= steps


    def list_dir(self, logdir):
        raise NotImplementedError


    def find_results(self, logdir, env, algo, ts, steps, params):
        matches = []
        all_runs_params = []
        all_results_props = []

        for fname, params_path, results_path in self.list_dir(logdir):
            try:
                run_params = self.load_params(params_path)
                results_props = self.get_results_props(self.load_results(results_path))
                if self.check_name(fname, algo, env, ts)\
                        and self.check_params(run_params, params)\
                        and self.check_results(results_props, steps):
                    matches.append(fname)
                    all_runs_params.append(run_params)
                    all_results_props.append(results_props)
            
            except Exception as e:
                print(f'Error processing {fname}', file=sys.stderr)
                print(e, file=sys.stderr)
        
        return matches, all_runs_params, all_results_props


class BenchmarkRefinder(Refinder):
    def __init__(self):
        super().__init__('eval_episode_reward_mean', 'timesteps_total', '\t')

    def list_dir(self, logdir):
        for f in os.listdir(logdir):
            fname, fext = os.path.splitext(f)
            if fext == '.csv':
                fpath = os.path.join(logdir, fname)
                params_path = fpath + os.path.extsep + 'json'
                results_path = fpath + os.path.extsep + 'csv'
                if os.path.isfile(params_path) and os.path.isfile(results_path):
                    yield f, params_path, results_path


class ACERRefinder(Refinder):
    def __init__(self):
        super().__init__('eval_return_mean', 'time_step', ',')

    def list_dir(self, logdir):
        for f in os.listdir(logdir):
            fpath = os.path.join(logdir, f)
            params_path = os.path.join(fpath, 'parameters.json')
            results_path = os.path.join(fpath, 'results.csv')
            if os.path.isdir(fpath) and os.path.isfile(params_path) and os.path.isfile(results_path):
                yield f, params_path, results_path


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

    results, runs_params, all_res_props = [a + b for a, b in zip(
        ACERRefinder().find_results(args.logdir, args.env, args.algo, args.ts, args.steps, params),
        BenchmarkRefinder().find_results(args.logdir, args.env, args.algo, args.ts, args.steps, params)
    )]

    diffs = make_diffs(runs_params) if args.diff else [{}] * len(results)
    res_props = all_res_props if args.results else [{}] * len(results)

    for res, r, d in sorted(zip(results, res_props, diffs)):
        print(
            res,
            *[f'{k}={v}' for k, v in r.items()],
            *[f'{k}={v}' for k, v in d.items()],
            sep='\t'
        )
