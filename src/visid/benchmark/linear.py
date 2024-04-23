#!/usr/bin/env python3

"""Identification of a simulated linear--Gaussian system."""


import argparse
import collections
import importlib

import flax.linen as nn
import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
from scipy import interpolate, optimize, signal, sparse, stats

from visid import vi, gvi, modeling
from visid.benchmark import arggroups


def program_args():
    parser = argparse.ArgumentParser(
        description=__doc__, fromfile_prefix_chars='@'
    )
    parser.add_argument(
        '--nx', default=10, type=int,
        help='Number of model states.',
    )
    parser.add_argument(
        '--ny', default=5, type=int,
        help='Number of model outputs.',
    )
    parser.add_argument(
        '--nu', default=5, type=int,
        help='Number of model exogenous inputs.',
    )
    parser.add_argument(
        '--N', default=1000, type=int,
        help='Number samples per batch.',
    )
    parser.add_argument(
        '--max_pole_radius', default=0.95, type=float,
        help='Largest pole radius for the system.',
    )
    parser.add_argument(
        '--D_sparsity', default=0.2, type=float,
        help='Sparsity of the D matrix.',
    )
    parser.add_argument(
        '--sqrt_q', default=0.1, type=float, nargs='+',
        help='Square root of diagonal of process noise covariance matrix.',
    )
    parser.add_argument(
        '--sqrt_r', default=0.1, type=float, nargs='+',
        help='Square root of diagonal of measurement noise covariance matrix.',
    )
    parser.add_argument(
        '--nkern', default=50, type=int,
        help='Length of convolution smoother kernel.',
    )


    # Add common benchmark argument groups
    arggroups.add_jax_group(parser)
    arggroups.add_testing_group(parser)
    arggroups.add_random_group(parser)
    arggroups.add_output_group(parser)

    # Parse command-line arguments
    args = parser.parse_args()

    # Process common benchmark argument groups
    arggroups.process(args)

    # Return parsed arguments
    return args


def sample_system(args):
    """Sample a linear--Gaussian system."""
    # Generate the system matrices
    A = np.random.rand(args.nx, args.nx)
    B = np.random.randn(args.nx, args.nu)
    C = np.random.randn(args.ny, args.nx)
    D = np.random.randn(args.ny, args.nu)
    D[np.random.rand(*D.shape) > args.D_sparsity] = 0.0

    # Generate the system noise covariance matrices
    sq = np.zeros(args.nx)
    sr = np.zeros(args.ny)
    sq[:] = args.sqrt_q
    sr[:] = args.sqrt_r
    sQ = np.diag(sq)
    sR = np.diag(sr)
    Q = sQ @ sQ.T
    R = sR @ sR.T

    # Determine the system poles (eigenvalues)
    eigval, eigvec = np.linalg.eig(A)
    complex_eigval = np.iscomplex(eigval).nonzero()[0]
    eigval_angle = np.exp(1j * np.angle(eigval))

    # Generate a new set of replacement eigenvalues for poles with large radii
    stab_eigval = args.max_pole_radius * np.random.rand(args.nx) * eigval_angle

    # Ensure that complex eigenvalues are conjugate pairs
    stab_eigval[complex_eigval[1::2]] = stab_eigval[complex_eigval[::2]].conj()

    # Replace eigenvalues with large radii
    keep = np.abs(eigval) < args.max_pole_radius
    eigval = np.where(keep, eigval, stab_eigval)

    # Reconstruct the state transition matrix with updated eigenvalues
    A = np.real(eigvec @ np.diag(eigval) @ np.linalg.inv(eigvec))

    # Create dynamic system models
    sys = signal.StateSpace(A, B, C, D, dt=True)

    # Create augmented system model for simulating with process noise
    B_aug = np.c_[B, sQ]
    D_aug = np.c_[D, np.zeros((args.ny, args.nx))]
    sys_aug = signal.StateSpace(A, B_aug, C, D_aug, dt=True)

    # Return system matrices
    mat_dict = dict(A=A, B=B, C=C, D=D, Q=Q, R=R, sq=sq, sr=sr)
    mats = collections.namedtuple('SystemMatrices', mat_dict)(**mat_dict)
    return sys, sys_aug, mats


def gen_data(sys_aug, mats, args):
    # Generate the simulation signals
    u = np.where(np.random.rand(args.N, args.nu) > 0.5, -1.0, 1.0)
    w = np.random.randn(len(u), args.nx)
    v = mats.sr * np.random.randn(len(u), args.ny)

    # Simulate the system
    tout, yclean, x = signal.dlsim(sys_aug, np.c_[u, w])
    y = yclean + v
    return u, y, x


class VIModel(vi.VIBase):
    nx: int
    nu: int
    ny: int
    nkern: int

    def setup(self):
        self.model = modeling.LinearModel(self.nx, self.nu, self.ny)
        self.smoother = gvi.LinearConvolutionSmoother(self.nkern, self.nx)
        self.sampler = gvi.SigmaPointSampler(self.nx)


if __name__ == '__main__':
    args = program_args()
    sys, sys_aug, mats = sample_system(args)
    u, y, x = gen_data(sys_aug, mats, args)

    model = VIModel(args.nx, args.nu, args.ny, args.nkern)
