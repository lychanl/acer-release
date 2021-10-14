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
        self.started = False
        self.gpu = None

        self.last_output = None
        self.last_err = None

        self.last_eval_outs = []
        self.last_eval_out_means = [None for _ in range(log_outs)]
        self.timesteps = 0

    def start(self, gpu, verbose):
        self.gpu = gpu

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
        self.started = True

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
        if 'ERROR' in out.upper() or 'EXCEPTION' in out.upper():
            self.last_err = out

    def show(self, show_name=True):
        if not self.started:
            status = "AWAITING"
        elif self.return_code is None:
            status = "RUNNING"
        else:
            status = "EXITED: " + str(self.return_code)

        error = f" ({self.last_err or self.last_output})" if self.return_code else ""

        descr = f'Timesteps: {self.timesteps} Last results: {" ".join(map(str, self.last_eval_out_means))}'
        name = f"{self.name}:" if show_name else " " * (len(self.name) + 1)
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
        alive = [p.process.stdout for p in processes if p.alive()]
        to_refresh, _, _ = select.select(alive, [], [])

        for p in processes:
            if p.process.stdout in to_refresh:
                p.refresh(verbose)
            if p.alive():
                alive.append(p)
            else:
                finished.append(p)
    
    return alive, finished


def get_next_gpu(gpus, running):
    if not gpus:
        return None

    counters = {g: 0 for g in gpus}

    for p in running:
        counters[p.gpu] += 1

    return min(counters, key=counters.get)


def run(base_params, splitted_sets, repeats, gpus, max_procs, verbose):
    processes_groups = [make_procs(base_params, set, repeats) for set in splitted_sets]
    processes = [p for g in processes_groups for p in g.processes]

    running = []
    awaitng = []
    finished = []

    for i, p in enumerate(processes):
        if len(running) < max_procs:
            if gpus:
                gpu = gpus[i % len(gpus)]
            else:
                gpu = None

            print(f"Starting process {i+1} of {len(processes)} (gpu spec: {gpu})...")
            p.start(gpu, verbose)
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
            print()
            for g in processes_groups:
                g.show()
            print()
            
            while len(running) < max_procs and awaitng:
                gpu = get_next_gpu(gpus, running)
                print(f"Starting process (gpu spec: {gpu})...")
                p.start(gpu, verbose)
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
    parser.add_argument('--max_procs', default=1, type=int)

    args, params = parser.parse_known_args()

    splitted_params = split_run_params(args.optim)

    run(params, splitted_params, args.repeats, args.gpus, args.max_procs, args.verbose)


if __name__ == "__main__":
    main()
