import argparse
import functools
import json
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


TIME_STEPS = 3000000
TIME_DELTA = 30000

EVAL_START = 2900000

TS_COL = 'step'
VALUE_COL = 'value'

PL = False

NAMES_DICT = {
    'STDEXPLORACER': 'StdACER',
    'EXPLORACER': 'ACERAX',
    'FASTACER': 'FastACER'
}


class Filter:
    def __init__(self, logdir, param, values) -> None:
        self.logdir = logdir
        self.param = param
        self.values = values

    def apply(self, logdir, params):
        if self.logdir is not None and self.logdir != logdir:
            return True
        return str(params.get(self.param, None)) in self.values


def parse_filter(str_filter):
    tokens = str_filter.split(':')
    assert len(tokens) <= 2, f"Too many ':' in filter {str_filter}"
    if len(tokens) == 2:
        logdir, algo_filter = tokens
    else:
        logdir, algo_filter = None, tokens[0]

    assert(algo_filter.count('=')) == 1, f"Expected a single '=' in filter {str_filter}"
    param, value = algo_filter.split('=')
    values = value.split('?')

    return Filter(logdir, param, values)


class LogDir(object):
    def __init__(
        self, path, name_pattern, sep,
        time_step_col, value_col, filters=()):
        self.path = path
        self.name_pattern = name_pattern
        self.sep = sep
        self.time_step_col = time_step_col
        self.value_col = value_col
        self.filters = filters

    def parse(self, name):
        try:
            return pd.read_csv(name, sep=self.sep, header=0, skip_blank_lines=True) if os.path.isfile(name) else None
        except:
            return None

    def list_files(self):
        if self.path is None:
            files = []
        elif isinstance(self.path, str):
            files = [(self.path, file) for file in os.listdir(self.path)]
        else:
            files = [(path, file) for path in self.path for file in os.listdir(path)]
        matches = [re.fullmatch(self.name_pattern, f[1]) for f in files]
        return tuple(zip(*[(*f, m) for f, m in zip(files, matches) if m]))
    
    def get_filter_and_variant(self, alg, path, file, variant_params):
        return True, None

    def get_env(self, match):
        env_name = match.group(1)
        ts_match = re.fullmatch(r'(.+)TS(\d+)-v(\d+)', env_name)
        if not ts_match:
            return env_name, 1
        
        else:
            return f'{ts_match.group(1)}-v{ts_match.group(3)}', int(ts_match.group(2))
    
    def get_alg(self, match, path, file, variant_params):
        alg = match.group(2).upper()
        filter_result, variant = self.get_filter_and_variant(alg, path, file, variant_params)

        return (alg, variant) if filter_result else None
    
    def extract_values(self, df, ts):
        if df is None:
            return None
        
        delta = TIME_DELTA * ts
        steps = TIME_STEPS * ts

        t = 0
        rows = []
        for i, row in df.iterrows():
            if row[self.time_step_col] > t:
                while row[self.time_step_col] > t:
                    t += delta
            elif rows:
                rows[len(rows) - 1] = False
            
            if row[self.time_step_col] <= steps:
                rows.append(True)
            else:
                rows.append(False)
        
        df = df[[self.time_step_col, self.value_col]][rows]

        if df.empty:
            return None

        df[self.time_step_col] = df[self.time_step_col].apply(lambda x: int(np.ceil(x / delta) * delta))

        return df
    
    def parse_files(self, variant_params):
        lf = self.list_files()
        if len(lf) == 0:
            return {}
        paths, files, matches = lf
        envs = [self.get_env(match) for match in matches]
        algs = [self.get_alg(match, path, file, variant_params) for match, path, file in zip(matches, paths, files)]
        values = [self.extract_values(self.parse(os.path.join(path, name)), e[1]) for path, name, e in zip(paths, files, envs)]

        ret = {}

        for e, a, v in zip(envs, algs, values):
            if v is not None and not v.empty and a is not None:
                v.columns = [TS_COL, VALUE_COL]
                if e not in ret:
                    ret[e] = {}
                if a not in ret[e]:
                    ret[e][a] = []
                ret[e][a].append(v)
        
        return ret


class ACERLogDir(LogDir):
    def __init__(self, path, filters, column):
        super(ACERLogDir, self).__init__(
            path, 
            r'(.+?)_(.+?)_.+', ',', 'time_step', column, filters)

    def parse(self, name):
        return super(ACERLogDir, self).parse(os.path.join(name, 'results.csv'))

    def get_filter_and_variant(self, alg, path, file, variant_params):
        params_file = os.path.join(path, file, 'parameters.json')
        if not os.path.isfile(params_file):
            print(f'Missing parameters file for {path}')
            return None

        with open(params_file, 'r') as vf:
            json_data = json.load(vf)
            return all(f.apply(os.path.split(self.path)[1], json_data) for f in self.filters),\
                tuple(str(json_data.get(param, default)) for param, default in variant_params)


class BenchmarkLogDir(LogDir):
    def __init__(self, path, filters):
        super(BenchmarkLogDir, self).__init__(
            path, 
            r'(.+?)_(.+?)_.*\.csv', '\t', 'timesteps_total', 'eval_episode_reward_mean', filters)


def to_arrays(df, stderr=False):
    values_cols = [c for c in df.columns if c != TS_COL]
    values = np.array(df[values_cols])
    divisor = np.sqrt(values.shape[1] - 1) if stderr else 1
    return (
        np.array(df[TS_COL]),  # time
        np.nanmean(values, axis=1),  # mean
        np.nanstd(values, axis=1) / divisor,  # std
        len(df.columns) - 1,  # num
        np.nanmean((values[1:] + values[:-1]) / 2),  # AULC
        np.nanstd(np.nanmean((values[1:] + values[:-1]) / 2, axis=0)) / divisor,  # AULC STD
    )


def make_order(values, first_algo):
    s = list(sorted(values.items()))
    first = []
    other = []

    for v in s:
        if first_algo and v[0].upper() == first_algo.upper():
            first.append(v)
        else:
            other.append(v)

    return first + other


def make_name(algo, variant, variant_params, mapping):
    name = NAMES_DICT.get(algo, algo)
    
    if variant:
        name = f'{name} ({", ".join(f"{param[0]}={value}" for param, value in zip(variant_params, variant))})'
    for base, target in mapping:
        name = name.replace(base, target)

    return name


def get_best_by_ts(data):
    out = {}
    bests = {}

    for (env, ts), v in data.items():
        ts = ts or 1
        if env not in out:
            out[env] = {}
            bests[env] = {}

        for algo, (time, mean, std, num, aulc, aulcs) in v.items():
            if algo not in out[env] or out[env][algo][1][-1] < mean[-1]:
                out[env][algo] = (time / ts, mean, std, num, aulc, aulcs)
                bests[env][algo] = ts

    for env, v in bests.items():
        print(f'Env {env}')
        for algo, ts in v.items():
            print(f'For algo {algo} best {ts}')

    out = {
        (env, None): {
            (algo, (*(variant or ()), bests[env][(algo, variant)])): d
            for (algo, variant), d in v.items()
        }
        for env, v in out.items()
    }
    return out


def join_dicts(dict1, dict2):
    dict1 = dict(dict1)
    for k, v in dict2.items():
        if k not in dict1:
            dict1[k] = dict(v)
        else:
            dict1[k].update(v)
    
    return dict1


def visualise(
        acer_dir_paths, benchmarks_dir_paths,
        out_dir, out_suffix, pl, first_algo,
        variant_params, mapping, limit_n_runs, show, best_by_ts,
        filters, filter_envs, legend_loc, stderr, column):
    acer_dirs = [ACERLogDir(dir_path, filters, column) for dir_path in acer_dir_paths or ()]
    benchmark_dirs = [BenchmarkLogDir(dir_path, filters) for dir_path in benchmarks_dir_paths or ()]
    acer_dir_data = [acer_dir.parse_files(variant_params) for acer_dir in acer_dirs]
    benchmark_dir_data = [benchmark_dir.parse_files(variant_params) for benchmark_dir in benchmark_dirs]

    dir_data = acer_dir_data + benchmark_dir_data
    data = functools.reduce(join_dicts, dir_data)

    data = {
        k: {
            kk: to_arrays(functools.reduce(
                lambda x, y: (None, pd.merge(x[1], y[1], on=TS_COL, how='outer', suffixes=(None, f'_{y[0]}')).sort_values(TS_COL)),
                enumerate(vv[:limit_n_runs])  # enumerate to avoid repeating column names
            )[1], stderr=stderr)
            for kk, vv in v.items()
        } for k, v in data.items()
    }
    
    if best_by_ts:
        data = get_best_by_ts(data)
        variant_params = (*variant_params, 'ts')

    tss = {ts for (_, ts), _ in data.items()}
    include_ts = len(tss) > 1
    
    plt.rc('font', size=12)

    if not show and out_dir:
        os.makedirs(out_dir, exist_ok=True)

    for (env, ts), v in data.items():
        if filter_envs:
            found = False
            for e in filter_envs:
                if env.lower().startswith(e.lower()):
                    found = True
            if not found:
                continue

        fig, ax = plt.subplots(figsize=(6, 4), nrows=1, ncols=1)
        plt.title(f'{env} ts {ts}' if include_ts else f'{env}')

        v = {
            make_name(algo, variant, variant_params, mapping): (time, mean, std, num, aulc, aulcs)
            for (algo, variant), (time, mean, std, num, aulc, aulcs) in v.items()
        }

        for name, (time, mean, std, num, aulc, aulcs) in make_order(v, first_algo):
            # name = algo[6:].replace('kappa', 'κ') if algo.startswith('qacer,kappa') else ''.join(filter(str.isalpha, algo.upper()))

            ax.plot(time / 1000000, mean, label=name)
            ax.fill_between(time / 1000000, mean - std, mean + std, alpha=0.2)

            # eval_mean = np.mean(mean[time >= EVAL_START * k[1]])
            # eval_std = np.linalg.norm(std[time >= EVAL_START * k[1]]) / len(std[time >= EVAL_START * k[1]])

            # print(f'{env} {ts}: {name}\tmean {eval_mean}\tstd {eval_std}')
            print(f'{env}{ " " + str(ts) if include_ts else ""}: {name}\tnum: {num} {mean[-1]:.2f} +- {std[-1]:.2f} AULC: {aulc:.2f} +- {aulcs:.2f}')
        ax.grid(which='major')
        #plt.minorticks_on()
        #plt.grid(which='minor', linestyle='--')
        ax.set_ylabel('Średnia suma nagród' if pl else 'Average return')
        ax.set_xlabel('Miliony kroków czasowych' if pl else 'Million timesteps')
        ax.legend(loc=legend_loc)
        ax.set_facecolor('white')
        fig.set_facecolor((0, 0, 0, 0))
        # plt.savefig(f'fig_{env}{out_suffix}.eps', format='eps', facecolor='white', bbox_inches='tight')
        if show:
            plt.show()
        else:
            figpath = os.path.join(out_dir, f'fig_{env}{f"_{ts}" if ts else ""}{out_suffix}.png')
            plt.savefig(figpath, format='png', bbox_inches='tight', transparent=False)
        plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--acer_dir', nargs='*', type=str, default=None)
    parser.add_argument('--benchmark_dir', nargs='*', type=str, default=None)

    parser.add_argument('--first_algo', type=str, default=None)
    parser.add_argument('--out_suffix', type=str, default='')
    parser.add_argument('--out_dir', type=str, default='')
    parser.add_argument('--variant_params', nargs='+', default=())
    parser.add_argument('--pl', default=False, action='store_true')
    parser.add_argument('--mapping', nargs='+', default=())
    parser.add_argument('--show', default=False, action='store_true')

    parser.add_argument('--best_by_ts', default=False, action='store_true')
    parser.add_argument('--filter', nargs='+', default=())
    parser.add_argument('--filter_envs', nargs='+')
    parser.add_argument('--legend_loc', default='upper left')
    parser.add_argument('--limit_n_runs', type=int, default=None)
    
    parser.add_argument('--stderr', default=False, action='store_true')
    parser.add_argument('--acer_column', type=str, default='eval_return_mean')

    args = parser.parse_args()

    acer_dir_path = args.acer_dir
    benchmarks_dir_path = args.benchmark_dir

    first_algo = args.first_algo
    out_suffix = args.out_suffix
    variant_params = args.variant_params
    variant_params = [vp.split(':') if ':' in vp else (vp, None) for vp in variant_params]
    pl = args.pl
    mapping = [r.split(':') if ':' in r else r.split('=') for r in args.mapping]
    show = args.show
    best_by_ts = args.best_by_ts
    out_dir = args.out_dir
    filter = [parse_filter(f) for f in args.filter]
    filter_envs = args.filter_envs
    legend_loc = args.legend_loc
    limit_n_runs = args.limit_n_runs
    stderr = args.stderr
    column = args.acer_column
    if column != 'eval_return_mean':
        assert not benchmarks_dir_path, 'Different column supported only for ACER'

    visualise(
        acer_dir_path, benchmarks_dir_path, 
        out_dir, out_suffix, pl, first_algo,
        variant_params, mapping, limit_n_runs, show, best_by_ts,
        filter, filter_envs, legend_loc, stderr, column)
