"""Classes for representing dynamic system models."""


import dataclasses
from typing import Literal

import flax.linen as nn
import hedeut as utils
import jax
import jax.numpy as jnp
import jax.scipy as jsp

from . import common, stats


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
        unmasked = ~jnp.isnan(y)
        logpdf = jsp.stats.norm.logpdf(y, mean, jnp.exp(self.meas_log_sigma))
        return jnp.sum(jnp.where(unmasked, logpdf, 0))


class LinearModel(nn.Module):
    """Discrete-time linear dynamic system model."""

    nx: int
    """Number of states."""

    nu: int
    """Number of exogenous (external) inputs."""

    ny: int
    """Number of outputs."""

    def setup(self):
        super().setup()

        nx = self.nx
        nu = self.nu
        ny = self.ny

        self.A = self.param('A', nn.initializers.zeros, (nx, nx))
        self.B = self.param('B', nn.initializers.zeros, (nx, nu))
        self.C = self.param('C', nn.initializers.zeros, (ny, nx))
        self.D = self.param('D', nn.initializers.zeros, (ny, nu))

    @utils.jax_vectorize_method(signature='(x),(u)->(x)')
    def f(self, x, u):
        return self.A @ x + self.B @ u

    @utils.jax_vectorize_method(signature='(x),(u)->(y)')
    def h(self, x, u):
        return self.C @ x + self.D @ u


class LinearMVNModel(MVNTransition, MVNMeasurement, LinearModel):
    """Discrete-time linear system model with multivariate normal noise."""


class LinearGaussianModel(MVNTransition, GaussianMeasurement, LinearModel):
    """Discrete-time dynamic system with independent Gaussian measurements."""
