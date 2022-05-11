import argparse
import subprocess
import os
import sys
import datetime
import time
import select
import json


def cls():
    os.system('cls' if os.name=='nt' else 'clear')


def time_str(time):
    return time.strftime("%Y-%m-%d %H:%M:%S")


def dt_str(dt):
    return str(dt).split('.')[0]

def print_row(row, widths):
    print('|'.join(str(cell).ljust(width) for cell, width in zip(row, widths)))

def show_table(table, colnames):
    widths = [len(name) for name in colnames]

    for row in table:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))

    print_row(colnames, widths)
    print('+'.join('-' * width for width in widths))
    for row in table:
        print_row(row, widths)



GPUS_ENV = 'CUDA_VISIBLE_DEVICES'
DEFAULT_LOC = "~/acer_research/acer/run.py"


class RunGroup:
    def __init__(self, RunClss, name_params, name, params, log_outs=3, repeats=1) -> None:
        self.processes = [RunClss(name, name_params, params, log_outs) for _ in range(repeats)]

    def show(self):
        return [p.show() for p in self.processes]
            

class Remote:
    def __init__(self, server, username, key, max_procs, python, loc) -> None:
        self.server = server
        self.max_procs = max_procs
        self.python = python
        self.loc = loc
        self.username = username
        self.key = key

    def __str__(self) -> str:
        return self.server


class Run:
    def __init__(self, name, name_params, params, log_outs=3) -> None:
        self.name = name
        self.name_params = name_params
        self.params = params
        self.open = True
        self.process = None
        self.return_code = None
        self.log_outs = log_outs
        self.started = False
        self.resource = None

        self.started_t = None
        self.finished = None

        self.last_output = None
        self.last_err_line = "(No line)"
        self.last_err = None

        self.last_eval_outs = []
        self.last_eval_out_means = [None for _ in range(log_outs)]
        self.timesteps = 0

    def _start(self, verbose):
        raise NotImplementedError

    def start(self, resource, verbose):
        self.resource = resource

        self._start(verbose)

        self.started = True
        self.started_t = datetime.datetime.now()

        time.sleep(5)

    def refresh(self, verbose):
        if not self.alive():
            return

        self._refresh(verbose)

    def _refresh(self, verbose):
        out = self.process.stdout.readline()
        self.process.stdout.isatty
        self.process_output(out, verbose)

        self.return_code = self.process.poll()
        if self.return_code is not None:
            for out in self.process.stdout.readlines():
                self.process_output(out, verbose)
            self.open = False
            self.finished = datetime.datetime.now()

    def process_output(self, out, verbose):
        out = out.strip()
        if not out:
            return

        if verbose:
            print(out)

        if 'evaluation run, return' in out:
            self.last_eval_outs.append(float(out.split()[-1]))
        elif 'saved evaluation results' in out:
            last_eval = sum(self.last_eval_outs) / len(self.last_eval_outs)
            self.last_eval_out_means = [last_eval] + self.last_eval_out_means[:-1]
            self.last_eval_outs = []
        elif 'total time steps done' in out:
            self.timesteps = int(out.split()[-1])

        self.last_output = out
        if 'ERROR' in out.upper() or 'EXCEPTION' in out.upper():
            self.last_err = out
        if 'acer/' in out or 'acer\\' in out:
            self.last_err_line = out + " "

    def show(self, show_name=True):
        if not self.started:
            status = "AWAITING"
        elif self.return_code is None:
            status = "RUNNING"
        else:
            status = "EXITED: " + str(self.return_code)

        error = f" {self.last_err_line + self.last_err if self.last_err else self.last_output}" if self.return_code else "-"

        # descr = f'Timesteps: {self.timesteps} Last results: {" ".join(map(str, self.last_eval_out_means))}'
        # name = f"{self.name}:" if show_name else " " * (len(self.name) + 1)
        # print(f"{name}\t{descr}\t{status}{error}")

        return [
            *[param if show_name else "" for param in self.name_params],
            self.timesteps,
            *[f'{out:.2f}' if out is not None else "-" for out in self.last_eval_out_means],
            self.resource,
            status,
            time_str(self.started_t) if self.started_t else "-",
            time_str(self.finished) if self.finished else "-",
            dt_str(self.finished - self.started_t if self.finished else datetime.datetime.now() - self.started_t) if self.started_t else "-",
            error,
        ]

    def alive(self):
        return self.process and self.open

    def interrupt(self):
        if self.process:
            self.process.terminate()


class LocalRun(Run):

    def _start(self, verbose):
        exe = sys.executable
        run = os.path.join(os.path.dirname(__file__), 'run.py')

        if verbose:
            print(f'Run: python {exe} script {run} {" ".join(self.params)}')

        if self.resource is None:
            env = None
        else:
            env = {GPUS_ENV: self.resource}

        self.process = subprocess.Popen(
            [exe, "-u", run, *self.params],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env
        )


class RemoteRun(Run):
    def _start(self, verbose):
        exe = sys.executable
        run = os.path.join(os.path.dirname(__file__), 'runmp_remote.py')

        cmd = f"{self.resource.python} {self.resource.loc} {' '.join(self.params)} 2>&1"

        if verbose:
            print(f'Run: Call: {exe} {run} on {self.resource.server} Remote command: {cmd}')

        self.process = subprocess.Popen(
            [exe, "-u", run, self.resource.server, self.resource.username, self.resource.key, cmd],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
        )

        if verbose:
            print('Started')


def make_procs(base_params, params, repeats, remote, param_names):
    return RunGroup(RemoteRun if remote else LocalRun, name_params(params, param_names), name(params), base_params + params, repeats=repeats)


def name_params(params, param_names):
    splitted = [par.split() for par in ' '.join(params).split('--') if par]
    out = {name: False for name in param_names}
    for par in splitted:
        if len(par) > 1:
            out[par[0]] = ' '.join(par[1:])
        else:
            out[par[0]] = True
    return [out[par] for par in param_names]


def name(params):
    return ' '.join([p[2:] if p.startswith('--') else p for p in params])


def poll(processes, verbose):
    alive = []
    finished = []

    if os.name == 'nt':
        num_alive = 0
        for p in processes:
            p.refresh(verbose)
            if p.alive():
                alive.append(p)
            else:
                finished.append(p)
    else:
        possible = [p.process.stdout for p in processes if p.alive()]
        to_refresh, _, _ = select.select(possible, [], [])

        for p in processes:
            if p.process.stdout in to_refresh:
                p.refresh(verbose)
            if p.alive():
                alive.append(p)
            else:
                finished.append(p)

    return alive, finished


def get_next_resource(resources, running):
    if not resources:
        return None

    counters = {r: 0 for r in resources}

    for p in running:
        counters[p.resource] += 1

    available = {r: limit - counters[r] for r, limit in resources.items() if counters[r] < limit}

    return max(available, key=counters.get)


def run(base_params, splitted_sets, repeats, resources, max_procs, remote, param_names, verbose):
    columns = param_names + ["Timesteps", "Out (-1)", "Out (-2)", "Out (-3)", "Resource", "Status", "Start", "Finish", "Duration", "Error"]

    processes_groups = [make_procs(base_params, set, repeats, remote, param_names) for set in splitted_sets]
    processes = [p for g in processes_groups for p in g.processes]

    running = []
    awaitng = []
    finished = []

    for i, p in enumerate(processes):
        if len(running) < max_procs:
            res = get_next_resource(resources, running)

            print(f"Starting process {i+1} of {len(processes)} (resource: {str(res)})...")
            p.start(res, verbose)
            running.append(p)
        else:
            awaitng.append(p)

    start = datetime.datetime.now()

    try:
        while running:
            running, new_finished = poll(running, verbose)
            finished.extend(new_finished)

            if not verbose:
                cls()
            now = datetime.datetime.now()
            print(f"START: {start} ACTUALIZATION: {now} DURATION: {now - start}")
            print("Base params: " + " ".join(map(str, base_params)))
            print(f"Running: {len(running)}\tAwaiting: {len(awaitng)}\tFinished: {len(finished)}")
            print()
            show_table([p for g in processes_groups for p in g.show()], columns)
            
            if verbose:
                print()
            
            while len(running) < max_procs and awaitng:
                res = get_next_resource(resources, running)
                print(f"Starting process (resource spec: {res})...")
                p = awaitng.pop()
                p.start(res, verbose)
                running.append(p)
    except KeyboardInterrupt:
        for p in processes:
            p.interrupt()


def split_run_params(optimized_params, optimized_flags):
    if not optimized_params and not optimized_flags:
        return [[]]

    runs = []
    if optimized_params:
        param = optimized_params[0]

        name = "--" + param[0]
        values = param[1:]

        for value in values:
            param_sets = split_run_params(optimized_params[1:], optimized_flags)
            for s in param_sets:
                s.append(name)
                s.append(value)
            runs.extend(param_sets)

        return runs
    else:
        param_sets = split_run_params([], optimized_flags[1:])
        flag = '--' + optimized_flags[0]
        return [[flag] + params for params in param_sets] + param_sets


def load_remotes(remote):
    with open(remote) as remote_list:
        raw = json.load(remote_list)

    return {r: r.max_procs for r in [Remote(
        spec['server'],
        spec['username'],
        spec['key'],
        spec['max_procs'],
        spec.get('python', 'python'),
        spec.get('loc', DEFAULT_LOC)
    ) for spec in raw]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--optim', action='append', nargs='+', type=str, default=[])
    parser.add_argument('--optim_flag', action='append', type=str, default=[])
    parser.add_argument('--verbose', action='store_true', default=False)
    parser.add_argument('--gpus', nargs='+', default=None, type=str)
    parser.add_argument('--repeats', default=1, type=int)
    parser.add_argument('--max_procs', default=1, type=int)
    parser.add_argument('--remote', type=str)

    args, params = parser.parse_known_args()

    splitted_params = split_run_params(args.optim, args.optim_flag)

    if args.remote:
        resources = load_remotes(args.remote)
    elif args.gpus:
        resources = {
            gpu: args.max_procs // len(args.gpus) for gpu in args.gpus
        }
    else:
        resources = {None: args.max_procs}
    max_procs = sum(resources.values())

    param_names = [par[0] for par in args.optim]
    run(params, splitted_params, args.repeats, resources, max_procs, args.remote is not None, param_names, args.verbose)


if __name__ == "__main__":
    main()
