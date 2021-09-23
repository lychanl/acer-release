import argparse
import subprocess
import os
import sys
import datetime
import time


def cls():
    os.system('cls' if os.name=='nt' else 'clear')


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

    def start(self, verbose):
        exe = sys.executable
        run = os.path.join(os.path.dirname(__file__), 'run.py')

        if verbose:
            print(f'Run: python {exe} script {run} {" ".join(self.params)}')

        self.process = subprocess.Popen(
            [exe, run, *self.params],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True, text=True
        )
        time.sleep(5)

    def refresh(self, verbose):
        if not self.alive():
            return

        out = self.process.stdout.readline()
        self.process_output(out, verbose)

        self.return_code = self.process.poll()
        if self.return_code is not None:
            for out in self.process.stdout.readlines():
                self.process_output(out)
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

    def show(self):
        status = "RUNNING" if self.return_code is None else "EXITED: " + str(self.return_code)
        descr = f'Timesteps: {self.timesteps} Last results: {" ".join(map(str, self.last_eval_out_means))}'
        print(f"{self.name}:\t{descr}\t{status}")

    def alive(self):
        return self.process and self.open

    def interrupt(self):
        self.process.terminate()


def make_proc(base_params, params):
    return Run(name(params), base_params + params)


def name(params):
    return ' '.join([p[2:] if p.startswith('--') else p for p in params])


def run(base_params, splitted_sets, verbose):
    processes = [make_proc(base_params, set) for set in splitted_sets]

    for p in processes:
        p.start()

    start = datetime.datetime.now()

    try:
        num_alive = len(processes)

        while num_alive:
            num_alive = 0
            for p in processes:
                p.refresh(verbose)
                if p.alive():
                    num_alive += 1

            if not verbose:
                cls()
            print(f"START: {start} ACTUALIZATION: {datetime.datetime.now()}")
            for p in processes:
                p.show()
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

    args, params = parser.parse_known_args()

    splitted_params = split_run_params(args.optim)

    run(params, splitted_params, args.verbose)


if __name__ == "__main__":
    main()
