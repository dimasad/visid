#!/usr/bin/env python3

"""Identification of a simulated linear--Gaussian system using mini-batches."""


import argparse
import datetime
import functools
import importlib
import itertools
import json
import os
import pathlib
import pickle
import time

import hedeut
import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
import optax
import scipy.io
import scipy.linalg
from scipy import interpolate, optimize, signal, sparse, stats

import gvispe.stats
from gvispe import common, estimators, modeling, sde


def impulse_err(imp_true, model, dec):
    """Calculate the impulse response error."""
    n = imp_true.shape[1]
    qdict = model.q_packer.unpack(dec.q)
    sys = signal.StateSpace(qdict['A'], qdict['B'], qdict['C'], qdict['D'], dt=1)
    y = np.array(signal.dimpulse(sys, n=n)[1])
    err = y - imp_true
    rmserr = np.sqrt(np.sum(err**2, axis=1))
    rmssig = np.sqrt(np.sum(imp_true**2, axis=1))
    eratio = np.mean(rmserr/rmssig)
    return eratio, y, qdict


def save_progress(args, start, siminfo, model, dec):  
    imp_true, param_true = siminfo  
    eratio, y, qdict = impulse_err(imp_true, model, dec)
    if args.txtout is not None:
        secs = (datetime.datetime.today() - start).total_seconds()
        args.txtout.seek(0)
        args.txtout.truncate()
        print(
            args.nx, args.nu, args.ny, args.N, args.Nbatch, 
            eratio, secs, args.seed, file=args.txtout
        )
        args.txtout.flush()
    if args.pickleout is not None:
        outdata = dict(
            params=qdict, param_true=param_true,
            nx=args.nx, nu=args.nu, ny=args.ny, N=args.N, Nbatch=args.Nbatch,
            seed=args.seed, imp_true=imp_true, y=y
        )
        args.pickleout.seek(0)
        args.pickleout.truncate()
        pickle.dump(outdata, args.pickleout, protocol=-1)
        args.pickleout.flush()
    return eratio


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=__doc__, fromfile_prefix_chars='@'
    )
    parser.add_argument(
        '--jax-x64', dest='jax_x64', action=argparse.BooleanOptionalAction,
        help='Use double precision (64bits) in JAX.',
    )
    parser.add_argument(
        '--jax-platform', dest='jax_platform', choices=['cpu', 'gpu'],
        help='JAX platform (processing unit) to use',
    )
    parser.add_argument(
        '--reload', default=[], nargs='*',
        help='Modules to reload'
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
        '--Nbatch', default=1000, type=int,
        help='Number of batches.',
    )
    parser.add_argument(
        '--max_pole_radius', default=1.0, type=float,
        help='Largest pole radius for the system.',
    )
    parser.add_argument(
        '--lrate0', default=5e-2, type=float,
        help='Stochastic optimization initial learning rate.',
    )
    parser.add_argument(
        '--transition_steps', default=1000, type=float,
        help='Learning rate "transition_steps" parameter.',
    )
    parser.add_argument(
        '--decay_rate', default=0.8, type=float,
        help='Learning rate "decay_rate" parameter.',
    )
    parser.add_argument(
        '--epochs', default=100, type=int,
        help='Optimization epochs.',
    )
    parser.add_argument(
        '--seed', default=0, type=int,
        help='Random number generator seed.',
    )
    parser.add_argument(
        '--nwin', type=int, default=101, help='Convolution window size.',
    )
    parser.add_argument(
        '--savemat', type=str, 
        help='File name to save data in MATLAB format.',
    )
    parser.add_argument(
        '--pickleout', type=argparse.FileType('wb'), help='Pickle output file.',
    )
    parser.add_argument(
        '--txtout', type=argparse.FileType('w'), help='Text output file.',
    )
    parser.add_argument(
        '--estimator', choices=['gvi', 'pem'],
        default='gvi', help='Parameter estimation method.',
    )
    args = parser.parse_args()

    # Apply JAX config options
    if args.jax_x64:
        jax.config.update("jax_enable_x64", True)
    if args.jax_platform:
        jax.config.update('jax_platform_name', args.jax_platform)

    # Reload modules
    for modname in args.reload:
        importlib.reload(globals()[modname])

    # Seed the RNG
    np.random.seed(args.seed)

    # Define model dimensions
    nx = args.nx
    nu = args.nu
    ny = args.ny
    ntrily = ny * (ny + 1) // 2
    N = args.N
    Nbatch = args.Nbatch

    # Determine if GVI will be used
    use_gvi = args.estimator == 'gvi'

    # Generate the model
    A = np.random.rand(nx, nx)
    L, V = np.linalg.eig(A)
    complex_L = np.iscomplex(L).nonzero()[0]
    stab_L = args.max_pole_radius * np.random.rand(nx) * np.exp(1j*np.angle(L))
    stab_L[complex_L[1::2]] = stab_L[complex_L[::2]].conj()
    L = np.where(np.abs(L) < args.max_pole_radius, L, stab_L)
    A = np.real(V @ np.diag(L) @ np.linalg.inv(V))
    B = np.random.randn(nx, nu)
    C = np.random.randn(ny, nx)
    D = np.random.randn(ny, nu) * (np.random.rand(ny, nu) > 0.8)
    sQ = np.diag(np.repeat(0.1, nx))
    sR = np.diag(np.repeat(0.15, ny))
    Q = sQ @ sQ.T
    R = sR @ sR.T

    # Simulate
    x = np.zeros((Nbatch*N, nx))
    y = np.zeros((Nbatch*N, ny))
    u = np.where(np.random.rand(Nbatch*N, nu) > 0.5, -1.0, 1.0)
    w = np.random.randn(Nbatch*N, nx)
    v = np.random.randn(Nbatch*N, ny)
    for k in range(N*Nbatch-1):
        x[k+1] = x[k] @ A.T + u[k] @ B.T + w[k] @ sQ.T
    y = x @ C.T + v @ sR.T + u @ D.T

    # Save data to MATLAB file
    if args.savemat is not None:
        matdata = {'A': A, 'B': B, 'C': C, 'D': D, 'y': y, 'u': u}
        scipy.io.savemat(args.savemat, matdata)

    # Divide data into batches
    data = []
    for i in range(Nbatch):
        offset_start = -args.nwin//2 if i > 0 and use_gvi else 0
        offset_end = args.nwin//2 if i+1 < Nbatch and use_gvi else 0
        batch_slice = slice(i*N + offset_start, (i+1)*N + offset_end)
        datum = estimators.Data(y[batch_slice], u[batch_slice])
        data.append(datum)

    # Create model and data objects
    model = modeling.LinearGaussianModel(nx, nu, ny)
    q_true = model.q_packer.pack(
        A=A, B=B, C=C, D=D,
        vech_log_sQ=common.vech(scipy.linalg.logm(sQ)), 
        vech_log_sR=common.vech(scipy.linalg.logm(sR))
    )
    sys_true = signal.StateSpace(A, B, C, D, dt=1)
    imp_true = np.array(signal.dimpulse(sys_true, n=100)[1])
    siminfo = imp_true, dict(A=A, B=B, C=C, D=D, sQ=sQ, sR=sR, Q=Q, R=R)

    if use_gvi:
        p = estimators.SmootherKernel(model, args.nwin, elbo_multiplier=-1)
        K0 = np.zeros((nx, ny+nu, args.nwin))
        K0[:, :, args.nwin//2] = np.random.randn(nx, ny+nu) * 1e-3
        dec0 = p.Decision(
            q=jnp.zeros(model.nq),
            K=jnp.array(K0),
            vech_log_S_cond=jnp.zeros(p.ntrilx),
            S_cross=jnp.zeros((model.nx, model.nx))
        )

        # Obtain integration coefficients
        pair_us_dev, xpair_w = gvispe.stats.sigmapts(2*nx)
        coeff = estimators.ExpectationCoeff(
            estimators.XCoeff(*gvispe.stats.sigmapts(nx)),
            estimators.XPairCoeff(pair_us_dev[:, :nx], pair_us_dev[:, nx:], xpair_w)
        )

        # JIT the cost and gradient functions
        value_and_grad = jax.jit(
            lambda dec, data: jax.value_and_grad(p.elbo)(dec, data, coeff)
        )
    else:
        p = estimators.PEM(model)
        K0 = np.random.randn(nx, ny) * 1e-3
        dec0 = p.Decision(
            q=jnp.zeros(model.nq),
            K=jnp.array(K0),
            vech_log_sR=jnp.zeros(p.ntrily),
            x0=jnp.zeros(0),
        )
        value_and_grad = jax.jit(jax.value_and_grad(p.cost))

    # Do the JIT
    value_and_grad(dec0, data[0])

    # Initialize solver
    dec = dec0
    sched = optax.exponential_decay(
        init_value=args.lrate0, 
        transition_steps=args.transition_steps,
        decay_rate=args.decay_rate,
    )
    optimizer = optax.adam(sched)
    opt_state = optimizer.init(dec)

    # Run optimization
    start = datetime.datetime.today()
    steps = 0
    for epoch in range(args.epochs):
        for i in np.random.permutation(Nbatch):
            # Calculate cost and gradient
            cost, grad = value_and_grad(dec, data[i])

            if steps % 100 == 0:
                fooc = p.Decision(*[jnp.sum(v**2) ** 0.5 for v in grad])
                eratio = save_progress(args, start, siminfo, model, dec)
                print(
                    f'{epoch}', f'sched={sched(steps):1.1e}', 
                    f'{cost=:1.2e}',
                    f'{fooc.K=:1.2e}', f'{fooc.q=:1.2e}', 
                    f'{eratio=:1.2e}',
                    sep='\t'
                )                

            if any(jnp.any(~jnp.isfinite(v)) for v in grad):
                print('Non-finite gradient in detected, stopping.')
                break

            updates, opt_state = optimizer.update(grad, opt_state)
            dec = optax.apply_updates(dec, updates)
            steps += 1
    
    # Save results
    save_progress(args, start, siminfo, model, dec)
