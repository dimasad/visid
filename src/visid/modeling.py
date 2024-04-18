"""Classes for representing dynamic system models."""


import dataclasses
from typing import Literal

import flax.linen as nn
import hedeut as utils
import jax.numpy as jnp
import jax.scipy as jsp

from . import common, stats


CovarianceSpec = (
    Literal['log_chol'] | Literal['L_expD_info']
)
"""Specifier of covariance matrix representation.

- 'L_expD_info' stands for the `L @ expm(D) @ L.T` decomposition of the information 
  matrix, where L is unitriangular and D is diagonal.
- 'log_chol' stands for the lower-triangular matrix logarithm of the Cholesky 
  factor of the covariance matrix R, that is, 
  `R == expm(log_chol) @ expm(log_chol).T`.
"""


class GaussianModel(nn.Module):
    """Discrete-time dynamic system model with Gaussian noise."""    

    @utils.jax_vectorize_method(signature='(x),(x),(u)->()') 
    def trans_logpdf(self, xnext, x, u):
        """Log-density of a state transition, log p(x_{k+1} | x_k, u_k)."""
        mean = self.f(x, u)
        inv_chol_cov = self.isQ(x, u)
        return stats.mvn_logpdf_ichol(xnext, mean, inv_chol_cov)

    @utils.jax_vectorize_method(signature='(y),(x),(u)->()')
    def meas_logpdf(self, y, x, u):
        """Log-density of a measurement, log p(y_k | x_k, u_k)."""
        mean = self.h(x, u)
        inv_chol_cov = self.isR(x, u)
        return stats.mvn_logpdf_ichol(y, mean, inv_chol_cov)


class TimeInvariantGaussianModel(GaussianModel):
    """Discrete-time dynamic system model with time-invariant Gaussian noise."""

    nx: int
    """Number of states."""

    ny: int
    """Number of outputs."""

    cov_type: CovarianceSpec = 'L_expD_info'
    """Type of covariance representation."""

    def setup(self):
        super().setup()

        nx = self.nx
        ny = self.ny
        zero_init = nn.initializers.zeros

        if self.cov_type == 'L_expD_info':
            self.d_iQ = self.param('d_iQ', zero_init, (nx,))
            self.d_iR = self.param('d_iR', zero_init, (ny,))
            self.L_iQ = self.param('L_iQ', zero_init, (nx, nx))
            self.L_iR = self.param('L_iR', zero_init, (ny, ny))
        elif self.cov_type == 'log_chol':
            self.log_chol_Q = self.param('log_chol_Q', zero_init, (nx, nx))
            self.log_chol_R = self.param('log_chol_R', zero_init, (ny, ny))
        else:
            raise ValueError(f'Unknown covariance type: {self.cov_type}')
        
    def isQ(self, x=None, u=None):
        if self.cov_type == 'L_expD_info':
            L = jnp.tril(self.L_iQ, k=-1) + jnp.identity(self.nx)
            sqrt_exp_d = jnp.exp(0.5 * self.d_iQ)
            return sqrt_exp_d[:, None] * L
        elif self.cov_type == 'log_chol':
            log_chol = jnp.tril(self.log_chol_Q)
            return jsp.linalg.expm(-log_chol)
        else:
            raise ValueError(f'Unknown covariance type: {self.cov_type}')

    def isR(self, x=None, u=None):
        if self.cov_type == 'L_expD_info':
            L = jnp.tril(self.L_iR, k=-1) + jnp.identity(self.nx)
            sqrt_exp_d = jnp.exp(0.5 * self.d_iR)
            return sqrt_exp_d[:, None] * L
        elif self.cov_type == 'log_chol':
            log_chol = jnp.tril(self.log_chol_R)
            return jsp.linalg.expm(-log_chol)
        else:
            raise ValueError(f'Unknown covariance type: {self.cov_type}')
        
    def sQ(self, x=None, u=None):
        """Cholesky factor of the state transition noise covariance matrix."""
        return jnp.linalg.inv(self.isQ(x, u))

    def sR(self, x=None, u=None):
        """Cholesky factor of the measurement noise covariance matrix."""
        return jnp.linalg.inv(self.isR(x, u))

    def Q(self, x=None, u=None):
        """State transition noise covariance matrix."""
        S = self.sQ(x, u)
        return S @ S.T

    def R(self, x=None, u=None):
        """Measurement noise covariance matrix."""
        S = self.sR(x, u)
        return S @ S.T


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


class LinearGaussianModel(TimeInvariantGaussianModel, LinearModel):
    """Discrete-time linear dynamic system model with Gaussian noise."""
