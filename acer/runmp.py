import argparse
import subprocess
import os
import sys
import datetime
import time
import select


def cls():
    os.system('cls' if os.name=='nt' else 'clear')


GPUS_ENV = 'CUDA_VISIBLE_DEVICES'


class RunGroup:
    def __init__(self, name, params, log_outs=3, repeats=1) -> None:
        self.processes = [Run(name, params, log_outs) for _ in range(repeats)]

    def show(self):
        for i, p in enumerate(self.processes):
            p.show(show_name=(i == 0))


class Run:
    def __init__(self, name, params, log_outs=3) -> None:
        self.name = name
        self.params = params
        self.open = True
        self.process = None
        self.return_code = None
        self.log_outs = log_outs

        self.last_output = None

        self.last_eval_outs = []
        self.last_eval_out_means = [None for _ in range(log_outs)]
        self.timesteps = 0

    def start(self, gpu, verbose):
        exe = sys.executable
        run = os.path.join(os.path.dirname(__file__), 'run.py')

        if verbose:
            print(f'Run: python {exe} script {run} {" ".join(self.params)}')

        if gpu is None:
            env = None
        else:
            env = {GPUS_ENV: gpu}

        self.process = subprocess.Popen(
            [exe, run, *self.params],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env
        )
        time.sleep(5)

    def refresh(self, verbose):
        if not self.alive():
            return

        out = self.process.stdout.readline()
        self.process.stdout.isatty
        self.process_output(out, verbose)

        self.return_code = self.process.poll()
        if self.return_code is not None:
            for out in self.process.stdout.readlines():
                self.process_output(out, verbose)
            self.open = False

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

    def show(self, show_name=True):
        status = "RUNNING" if self.return_code is None else "EXITED: " + str(self.return_code)

        error = f" ({self.last_output})" if self.return_code else ""

        descr = f'Timesteps: {self.timesteps} Last results: {" ".join(map(str, self.last_eval_out_means))}'
        name = "{self.name}:" if show_name else " " * (show_name + 1)
        print(f"{name}\t{descr}\t{status}{error}")

    def alive(self):
        return self.process and self.open

    def interrupt(self):
        self.process.terminate()


def make_procs(base_params, params, repeats):
    return RunGroup(name(params), base_params + params, repeats=repeats)


def name(params):
    return ' '.join([p[2:] if p.startswith('--') else p for p in params])


def poll(processes, verbose):
    if os.name == 'nt':
        num_alive = 0
        for p in processes:
            p.refresh(verbose)
            if p.alive():
                num_alive += 1

        return num_alive
    else:
        alive = [p.process.stdout for p in processes if p.alive()]
        to_refresh, _, _ = select.select(alive, [], [])

        for p in processes:
            if p.process.stdout in to_refresh:
                p.refresh()

        return len([p for p in alive if p.alive()])

def run(base_params, splitted_sets, repeats, gpus, verbose):
    processes_groups = [make_procs(base_params, set, repeats) for set in splitted_sets]
    processes = [p for g in processes_groups for p in g.processes]

    for i, p in enumerate(processes):
        if gpus:
            gpu = gpus[i % len(gpus)]
        else:
            gpu = None

        print(f"Starting process {i+1} of {len(processes)} (gpu spec: {gpu})...")

        p.start(gpu, verbose)

    start = datetime.datetime.now()

    try:
        num_alive = len(processes)

        while num_alive:
            num_alive = poll(processes, verbose)

            if not verbose:
                cls()
            now = datetime.datetime.now()
            print(f"START: {start} ACTUALIZATION: {now} DURATION: {now - start}")
            print("Base params: " + " ".join(map(str, base_params)))
            print()
            for g in processes_groups:
                g.show()
            print()
    except KeyboardInterrupt:
        for p in processes:
            p.interrupt()


def split_run_params(optimized_params):
    if not optimized_params:
        return [[]]

    runs = []
    param = optimized_params[0]

    name = "--" + param[0]
    values = param[1:]

    for value in values:
        param_sets = split_run_params(optimized_params[1:])
        for s in param_sets:
            s.append(name)
            s.append(value)
        runs.extend(param_sets)

    return runs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--optim', action='append', nargs='+', type=str)
    parser.add_argument('--verbose', action='store_true', default=False)
    parser.add_argument('--gpus', nargs='+', default=None, type=str)
    parser.add_argument('--repeats', default=1, type=int)

    args, params = parser.parse_known_args()

    splitted_params = split_run_params(args.optim)

    run(params, splitted_params, args.repeats, args.gpus, args.verbose)


if __name__ == "__main__":
    main()
