"""Classes for representing dynamic system models."""


import dataclasses
from typing import Literal

import flax.linen as nn
import hedeut as utils
import jax
import jax.numpy as jnp
import jax.scipy as jsp
import scipy.linalg

from . import common, stats, vi


class StochasticStateSpaceBase(nn.Module):
    """Base class for stochastic state-space models."""

    def path_trans_logpdf(self, xnext, x, u):
        """Joint log-density of the transitions of a state path.

        L = sum_{k=1}^N log p(x_k | x_{k-1}, u_{k-1})
        """
        logpdf = jax.vmap(self.trans_logpdf)(xnext, x, u)
        return jnp.sum(logpdf, axis=0)

    def path_meas_logpdf(self, y, x, u):
        """Joint log-likelihood of the measurements of a state path.

        L = sum_{k=0}^N log p(y_k | x_k, u_k)
        """
        logpdf = jax.vmap(self.meas_logpdf)(y, x, u)
        return jnp.sum(logpdf, axis=0)

    def avg_path_trans_logpdf(self, xnext, x, u, w=None):
        """Weighted average of path logpdf over a batch of samples."""
        if w is None:
            nsamp = len(x)
            w = jnp.ones(nsamp) / nsamp
        logpdf = jax.vmap(self.path_trans_logpdf, in_axes=(0,0,None))(xnext,x,u)
        return jnp.sum(w * logpdf, axis=0)

    def avg_path_meas_logpdf(self, y, x, u, w=None):
        """Weighted average of path log-likelihood over a batch of samples."""
        if w is None:
            nsamp = len(x)
            w = jnp.ones(nsamp) / nsamp
        logpdf = jax.vmap(self.path_meas_logpdf, in_axes=(None,0,None))(y, x, u)
        return jnp.sum(w * logpdf, axis=0)

    def free_sim(self, x0, u):
        """Simulate the system without noise."""
        scanfun = lambda x, u: (self.f(x, u), [x, self.h(x, u)])
        carry, (xpath, ypath) = jax.lax.scan(scanfun, x0, u)
        return xpath, ypath

    def filter(self, x0: jax.Array, data: vi.Data, K=None):
        """Run one-step-ahead predictor with linear corrections."""
        if K is None:
            K = self.ss_kf(x0, jnp.mean(data.u, axis=0))
    	
        def scanfun(xpred, datum):
            ypred = self.h(xpred, datum.u)
            err = datum.y - ypred
            xcorr = xpred + K @ err
            xpred_next = self.f(xcorr, datum.u)
            return xpred_next, (xpred, xcorr, ypred, err)
        carry, paths = jax.lax.scan(scanfun, x0, data)
        return paths

    def ss_kf(self, x=None, u=None):
        """Compute steady-state Kalman gain."""
        x = jnp.zeros(self.nx) if x is None else x
        u = jnp.zeros(self.nu) if u is None else u
        A = jax.jacfwd(self.f)(x, u)
        C = jax.jacfwd(self.h)(x, u)
        Q = jnp.linalg.inv(self.trans_info()())
        R = jnp.linalg.inv(self.meas_info()())

        Ppred = scipy.linalg.solve_discrete_are(A.T, C.T, Q, R)
        K = Ppred @ C.T @ jnp.linalg.inv(C @ Ppred @ C.T + R)
        return K


class MVNTransition(StochasticStateSpaceBase):
    """Multivariate normal process noise model."""

    nx: int
    """Number of states."""

    trans_info_repr: stats.PositiveDefiniteRepr = 'ldlt'
    """Representation of the state transition noise information matrix."""

    def setup(self):
        super().setup()
        if self.trans_info_repr == 'ldlt':
            self.trans_info = stats.LDLTParam(self.nx)
        elif self.trans_info_repr == 'log_chol':
            self.trans_info = stats.LogCholParam(self.nx)

    @utils.jax_vectorize_method(signature='(x),(x),(u)->()')
    def trans_logpdf(self, xnext, x, u):
        """Log-density of a state transition, log p(x_{k+1} | x_k, u_k)."""
        mean = self.f(x, u)
        return stats.mvn_logpdf_info(xnext, mean, self.trans_info())


class MVNMeasurement(StochasticStateSpaceBase):
    """Multivariate normal measurement noise model."""

    ny: int
    """Number of outputs."""

    meas_info_repr: stats.PositiveDefiniteRepr = 'ldlt'
    """Representation of the state measurement noise information matrix."""

    def setup(self):
        super().setup()
        if self.meas_info_repr == 'ldlt':
            self.meas_info = stats.LDLTParam(self.ny)
        elif self.meas_info_repr == 'log_chol':
            self.meas_info = stats.LogCholParam(self.ny)

    @utils.jax_vectorize_method(signature='(y),(x),(u)->()')
    def meas_logpdf(self, y, x, u):
        """Log-density of a measurement, log p(y_k | x_k, u_k)."""
        mean = self.h(x, u)
        return stats.mvn_logpdf_info(y, mean, self.meas_info())


class GaussianTransition(StochasticStateSpaceBase):
    """Independent Gaussian process noise model."""

    nx: int
    """Number of states."""

    init_trans_info: float | jax.Array = 0.0
    """Initial value of the transition noise information matrix log-diagonal."""

    def setup(self):
        super().setup()
        initializer = nn.initializers.constant(self.init_trans_info)
        self.trans_info = stats.LogDiagParam(self.nx, initializer=initializer)

    @utils.jax_vectorize_method(signature='(x),(x),(u)->()')
    def trans_logpdf(self, xnext, x, u):
        """Log-density of a state transition, log p(x_{k+1} | x_k, u_k)."""
        mean = self.f(x, u)
        return stats.mvn_logpdf_info(xnext, mean, self.trans_info())


class GaussianMeasurement(StochasticStateSpaceBase):
    """Independent Gaussian measurement noise model."""

    ny: int
    """Number of outputs."""

    def setup(self):
        super().setup()
        self.meas_log_sigma = self.param(
            'meas_log_sigma', nn.initializers.zeros, (self.ny,)
        )

    @utils.jax_vectorize_method(signature='(y),(x),(u)->()')
    def meas_logpdf(self, y, x, u):
        """Log-density of a measurement, log p(y_k | x_k, u_k)."""
        mean = self.h(x, u)
        sigma = jnp.exp(self.meas_log_sigma)

        # Mask out nans
        unmasked = ~jnp.isnan(y)
        y_masked = jnp.where(unmasked, y, 0)

        logpdf = jsp.stats.norm.logpdf(y_masked, mean, sigma)
        return jnp.sum(unmasked * logpdf)
    
    @property
    def meas_info(self):
        return lambda: stats.LogDiagMatrix(-self.meas_log_sigma)


class LinearTransitions(nn.Module):
    """Discrete-time dynamic system with linear state transition model."""

    nx: int
    """Number of states."""

    nu: int
    """Number of exogenous (external) inputs."""

    ny: int
    """Number of outputs."""

    A_free: jax.Array | bool = True
    """Which entries of the state transition matrix are free parameters."""

    B_free: jax.Array | bool = True
    """Which entries of the input matrix are free parameters."""

    A_given: jax.Array | float = 0.0
    """Given values of the state transition matrix are free parameters."""

    B_given: jax.Array | float = 0.0
    """Given values of the input matrix are free parameters."""


    def setup(self):
        super().setup()

        nx = self.nx
        nu = self.nu

        self.A = common.ArrayParam((nx, nx), self.A_free, self.A_given)
        self.B = common.ArrayParam((nx, nu), self.B_free, self.B_given)

    @utils.jax_vectorize_method(signature='(x),(u)->(x)')
    def f(self, x, u):
        return self.A() @ x + self.B() @ u


class LinearMeasurements(nn.Module):
    """Discrete-time dynamic system with linear measurement model."""

    nx: int
    """Number of states."""

    nu: int
    """Number of exogenous (external) inputs."""

    ny: int
    """Number of outputs."""

    C_free: jax.Array | bool = True
    """Which entries of the ouput matrix are free parameters."""

    D_free: jax.Array | bool = True
    """Which entries of the feedthrough matrix are free parameters."""

    C_given: jax.Array | float = 0.0
    """Given values of the ouput matrix are free parameters."""

    D_given: jax.Array | float = 0.0
    """Given values of the feedthrough matrix are free parameters."""


    def setup(self):
        super().setup()

        nx = self.nx
        nu = self.nu
        ny = self.ny

        self.C = common.ArrayParam((ny, nx), self.C_free, self.C_given)
        self.D = common.ArrayParam((ny, nu), self.D_free, self.D_given)

    @utils.jax_vectorize_method(signature='(x),(u)->(y)')
    def h(self, x, u):
        return self.C() @ x + self.D() @ u


class LinearModel(LinearTransitions, LinearMeasurements):
    """Discrete-time linear dynamic system model."""

class LinearMVNModel(MVNTransition, MVNMeasurement, LinearModel):
    """Discrete-time linear system model with multivariate normal noise."""


class LinearGaussianModel(MVNTransition, GaussianMeasurement, LinearModel):
    """Discrete-time dynamic system with independent Gaussian measurements."""


def compare(model, states, datasets):
    """Get model responses against datasets"""
    # Initialize Results
    y = [None] * len(datasets)
    ysim = [None] * len(datasets)
    ypred = [None] * len(datasets)
    xtrans = [None] * len(datasets)
    mdltrans = [None] * len(datasets)

    # Iterate over datasets
    for i, (x, data) in enumerate(zip(states, datasets)):
        # Output path
        y[i] = model.h(x, data.u)

        # Free simulation
        ysim[i] = model.free_sim(x[0], data.u)[1]

        # One-step-ahead predictor
        ypred[i] = model.filter(x[0], data)[2]

        # Smoothed transition error
        if hasattr(model, 'fc'):
            xtrans[i] = jnp.diff(x, axis=0) / model.dt
            mdltrans[i] = model.fc(x, data.u)[:-1]
        else:
            xtrans[i] = x[1:]
            mdltrans[i] = model.f(x, data.u)[:-1]

    return y, ysim, ypred, xtrans, mdltrans
