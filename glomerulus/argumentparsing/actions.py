import argparse
import os

from argparse import Action, BooleanOptionalAction, SUPPRESS




class MultiTargetBooleanOptionalAction(BooleanOptionalAction):
    """Multi-target Boolean Option."""
    def __init__(self, option_strings, dest, **kwargs):
        kwargs['default'] = SUPPRESS
        kwargs['metavar'] = None
        kwargs['type']    = bool
        kwargs.pop('nargs', None)
        dest = [dest] if isinstance(dest, str) else dest
        dest = tuple(d.replace('-', '_') for d in dest)
        self.dests = dest
        super().__init__(option_strings, argparse.SUPPRESS, **kwargs)

    def __call__(self, parser, ns, values, option_string=None):
        v = not option_string.startswith('--no-')
        for d in self.dests:
            setattr(ns, d, v)

class DistRankAction(Action):
    def __init__(self, option_strings, dest, **kwargs):
        N = os.environ.get('RANK',
            os.environ.get('SLURM_PROCID', '0'))
        kwargs['type']    = int
        kwargs['choices'] = None
        kwargs['default'] = int(N)
        kwargs.pop       ('nargs',    None)
        kwargs.setdefault('metavar', 'RANK')
        kwargs.setdefault('help',    'Set distributed process rank.')
        super().__init__(option_strings, dest, **kwargs)

    def __call__(self, parser, ns, values, option_string=None):
        assert values>=0, f'Distributed rank must be >= 0, but is {values}!'
        super().__call__(parser, ns, values, option_string=option_string)
        os.environ['RANK'] = str(values)

class DistLocalRankAction(Action):
    def __init__(self, option_strings, dest, **kwargs):
        N = os.environ.get('LOCAL_RANK',
            os.environ.get('SLURM_LOCALID', '0'))
        kwargs['type']    = int
        kwargs['choices'] = None
        kwargs['default'] = int(N)
        kwargs.pop       ('nargs',    None)
        kwargs.setdefault('metavar', 'LOCAL_RANK')
        kwargs.setdefault('help',    'Set distributed process local rank.')
        super().__init__(option_strings, dest, **kwargs)

    def __call__(self, parser, ns, values, option_string=None):
        assert values>=0, f'Distributed local rank must be >= 0, but is {values}!'
        super().__call__(parser, ns, values, option_string=option_string)
        os.environ['LOCAL_RANK'] = str(values)

class DistWorldSizeAction(Action):
    def __init__(self, option_strings, dest, **kwargs):
        N = os.environ.get('WORLD_SIZE',
            os.environ.get('SLURM_NPROCS', '1'))
        kwargs['type']    = int
        kwargs['choices'] = None
        kwargs['default'] = int(N)
        kwargs.pop       ('nargs',    None)
        kwargs.setdefault('metavar', 'WORLD_SIZE')
        kwargs.setdefault('help',    'Set distributed process world size.')
        super().__init__(option_strings, dest, **kwargs)

    def __call__(self, parser, ns, values, option_string=None):
        assert values>=1, f'Distributed world size must be >= 1, but is {values}!'
        super().__call__(parser, ns, values, option_string=option_string)
        os.environ['WORLD_SIZE'] = str(values)

class DistMasterPortAction(Action):
    def __init__(self, option_strings, dest, **kwargs):
        kwargs['type']    = int
        kwargs['choices'] = None
        kwargs.pop       ('nargs',    None)
        kwargs.setdefault('default',  int(os.environ.get('MASTER_PORT',
                                          kwargs.get('default', '29400'))))
        kwargs.setdefault('metavar', 'PORT')
        kwargs.setdefault('help',    'Set distributed process master port number.')
        super().__init__(option_strings, dest, **kwargs)

    def __call__(self, parser, ns, values, option_string=None):
        assert values>=1 and values<=65535, f'TCP port number must be 1 <= t <= 65535, but is {values}!'
        super().__call__(parser, ns, values, option_string=option_string)
