#!/usr/bin/env python3

"""Identification of a simulated linear--Gaussian system."""


import argparse
import collections
import importlib
import typing

import flax.linen as nn
import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
import optax
from scipy import interpolate, optimize, signal, sparse, stats

from visid import gvi, modeling, vi
from visid.benchmark import arggroups


def program_args():
    parser = argparse.ArgumentParser(
        description=__doc__, fromfile_prefix_chars='@'
    )
    parser.add_argument(
        '--nx', default=4, type=int,
        help='Number of model states.',
    )
    parser.add_argument(
        '--ny', default=2, type=int,
        help='Number of model outputs.',
    )
    parser.add_argument(
        '--nu', default=2, type=int,
        help='Number of model exogenous inputs.',
    )
    parser.add_argument(
        '--N', default=1000, type=int,
        help='Number samples per batch.',
    )
    parser.add_argument(
        '--Nbatch', default=1000, type=int,
        help='Number of batches.',
    )
    parser.add_argument(
        '--max_pole_radius', default=0.95, type=float,
        help='Largest pole radius for the system.',
    )
    parser.add_argument(
        '--C-given', action=argparse.BooleanOptionalAction, dest='C_given',
        help='Whether the C matrix is given.',
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
        '--nkern', default=51, type=int,
        help='Length of convolution smoother kernel.',
    )
    parser.add_argument(
        '--nimpulse', default=100, type=int,
        help='Horizon to compute impulse response error ratio.',
    )

    # Add common benchmark argument groups
    arggroups.add_jax_group(parser)
    arggroups.add_testing_group(parser)
    arggroups.add_random_group(parser)
    arggroups.add_output_group(parser)
    arggroups.add_stoch_optim_group(parser)

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
    Ntotal = args.N * args.Nbatch + args.nkern - 1
    u = np.where(np.random.rand(Ntotal, args.nu) > 0.5, -1.0, 1.0)
    w = np.random.randn(len(u), args.nx)
    v = mats.sr * np.random.randn(len(u), args.ny)

    # Simulate the system
    tout, yclean, x = signal.dlsim(sys_aug, np.c_[u, w])
    y = yclean + v
    return u, y, x


def get_sys(estimator):
    """Create a state-space system from the bound VI estimator."""
    model = estimator.model
    return signal.StateSpace(model.A, model.B, model.C, model.D, dt=True)


def ier(sys, sys_ref, n):
    """Average impulse response error ratio."""
    h = np.array(sys.impulse(n=n)[1])
    h_ref = np.array(sys_ref.impulse(n=n)[1])
    err = h - h_ref
    err_norm = np.sqrt(np.sum(err**2, axis=1))
    sig_norm = np.sqrt(np.sum(h_ref**2, axis=1))
    eratio = np.mean(err_norm / sig_norm)
    return eratio, h, h_ref


class Estimator(vi.VIBase):
    nx: int
    nu: int
    ny: int
    nkern: int
    C_given: typing.Optional[jax.Array] = None

    Data = gvi.LinearConvolutionSmoother.Data

    def setup(self):
        C_given = 0.0 if self.C_given is None else self.C_given
        C_free = self.C_given is None
        self.model = modeling.LinearGaussianModel(
            self.nx, self.nu, self.ny, C_given=C_given, C_free=C_free
        )
        self.smoother = gvi.LinearConvolutionSmoother(self.nkern, self.nx)
        self.sampler = gvi.SigmaPointSampler(self.nx)


if __name__ == '__main__':
    args = program_args()
    sys_true, sys_aug, mats = sample_system(args)
    u, y, x = gen_data(sys_aug, mats, args)

    # Create the PRNG keys
    key = jax.random.PRNGKey(args.seed)
    key, init_key = jax.random.split(key)

    # Instantiate the model
    C_given = None if args.C_given else mats.C
    estimator = Estimator(
        args.nx, args.nu, args.ny, args.nkern, C_given=C_given
    )

    # Create the data objects
    skip0 = (args.nkern - 1) // 2
    skip1 = args.nkern - 1 - skip0
    data = [None] * args.Nbatch
    for i in range(args.Nbatch):
        start = i * args.N + skip0
        end = (i + 1) * args.N + skip0
        data[i] = estimator.Data(
            y=y[start:end], u=u[start:end], 
            conv_y=y[start-skip0:end+skip1], 
            conv_u=u[start-skip0:end+skip1],
        )

    # Initialize the Estimator
    v = estimator.init(init_key, data[0])

    # Create gradient and cost functions
    obj_fun = jax.jit(jax.value_and_grad(lambda v, d: estimator.apply(v, d)))

    # Build the optimizer
    sched = args.lrate_sched
    tx = optax.adam(learning_rate=sched)
    opt_state = tx.init(v)

    # Run the optimization loop
    steps = 0
    for e in range(args.epochs):
        for i in np.random.permutation(args.Nbatch):
            obj, grad = obj_fun(v, data[i])
            updates, opt_state = tx.update(grad, opt_state)
            v = optax.apply_updates(v, updates)

            # Print progress
            if steps % args.display_skip == 0:
                sys_est = jax.apply(get_sys, estimator)(v)
                eratio = ier(sys_est, sys_true, args.nimpulse)[0]
                print(
                    f'{e}', f'{sched(steps):1.1e}', f'{obj:1.2e}', 
                    f'{eratio:1.2e}', sep='\t'
                )
            
            steps += 1

    sys_opt = jax.apply(get_sys, estimator)(v)
