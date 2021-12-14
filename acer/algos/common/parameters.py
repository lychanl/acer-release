import functools
from algos.common.automodel import AutoModelComponent
import tensorflow as tf

import re


def get_adapts_from_kwargs(kwargs, params):
    return {
        f'{param}.adapt': kwargs[f'{param}.adapt']
        for param in params
        if f'{param}.adapt' in kwargs
    }


class Adaptation:
    def __init__(self, arg, adaptation, adaptation_args) -> None:
        self.arg = arg
        self.adaptation = adaptation
        self.adaptation_args = adaptation_args


class Parameters(AutoModelComponent):
    def __init__(self, name, **kwargs) -> None:
        super().__init__()
        self.name = name
        self.params = {}
        self.adaptations = {}

        self.FUNCTIONS = {
            'exp_decay': {
                'func': self.exp_decay,
                'args': {'time': 'base.time_step'},
                'ctypes': (float, float),
                'cdefaults': (0.99, 1.)
            },
            'linear': {
                'func': self.linear,
                'args': {'time': 'base.time_step'},
                'ctypes': (float, float),
                'cdefaults': (None, None)
            }
        }

        for arg, value in kwargs.items():
            if '.' in arg:
                continue

            if f'{arg}.adapt' in kwargs:
                self.params[arg] = tf.Variable(value)
                self.adaptations[arg] = self._parse_adaptation(arg, value, kwargs[f'{arg}.adapt'])
                self.register_method(f'adapt_{arg}', self.adaptations[arg].adaptation, self.adaptations[arg].adaptation_args)
                self.targets.append(f'adapt_{arg}')
            else:
                self.params[arg] = tf.constant(value)
            self.register_method(arg, tf.function(functools.partial(self.get_value, arg)), {})

    def get_value(self, param):
        return tf.identity(self.params[param])

    def _parse_adaptation(self, arg, initial, adaptation):
        assert re.match(r'\w+(\((\w+,\s*)*(\w+\s*)\))?', adaptation), "Invalid adaptation format, should be func("
        func, params_str = adaptation.rstrip(')').split('(')

        spec = self.FUNCTIONS[func]

        params = tuple(t(v) for v, t in zip(params_str.split(','), spec['ctypes']))
        params = params + spec['cdefaults'][len(params):]

        args = {'param': f'{self.name}.{arg}'}
        args.update(spec['args'])

        return Adaptation(arg, functools.partial(spec['func'], var=self.params[arg], initial=initial, cparams=params), args)

    def exp_decay(self, var, param, initial, cparams, time):
        coeff, limit = cparams

        value = tf.maximum(initial * coeff ** tf.cast(time, tf.float32), limit)

        var.assign(tf.cast(value, var.dtype))

    def linear(self, var, param, initial, cparams, time):
        to_val, to_time = cparams

        value = initial + (to_val - initial) * tf.minimum(tf.cast(time, tf.float32) / to_time, 1.)

        var.assign(tf.cast(value, var.dtype))

    def __getitem__(self, param):
        return self.params[param].numpy()
