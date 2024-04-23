"""Common argument groups for benchmarks."""


import argparse
import importlib

import numpy as np
import jax.config


def add_jax_group(parser):
    group = parser.add_argument_group('jax', 'JAX configuration.')
    group.add_argument(
        '--jax-x64', dest='jax_x64', action=argparse.BooleanOptionalAction,
        help='Use double precision (64bits) in JAX.',
    )
    group.add_argument(
        '--jax-platform', dest='jax_platform', choices=['cpu', 'gpu'],
        help='JAX platform (processing unit) to use',
    )
    return group


def add_testing_group(parser):
    group = parser.add_argument_group('testing', 'Interactive testing config.')
    group.add_argument(
        '--reload', default=[], nargs='*', help='Modules to reload'
    )
    return group


def add_random_group(parser):
    group = parser.add_argument_group(
        'random', 'Random number generator configuration.'
    )
    group.add_argument('--seed', default=0, type=int, help='Random seed.')
    return group


def add_output_group(parser):
    group = parser.add_argument_group('output', 'Output configuration.')
    group.add_argument(
        '--matout', type=str, help='File name to save data in MATLAB format.',
    )
    group.add_argument(
        '--pickleout', type=argparse.FileType('wb'), help='Pickle output file.',
    )
    group.add_argument(
        '--txtout', type=argparse.FileType('w'), help='Text output file.',
    )
    return group


def process(args):
    if 'jax_x64' in args and args.jax_x64:
        jax.config.update('jax_enable_x64', True)
    if 'jax_platform' in args and args.jax_platform:
        jax.config.update('jax_platform_name', args.jax_platform)
    if 'reload' in args:
        for module_name in args.reload:
            module = importlib.import_module(module_name)
            importlib.reload(module)
    if 'seed' in args:
        np.random.seed(args.seed)