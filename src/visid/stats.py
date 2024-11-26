"""Statistical helper functions."""


import abc
import math
import typing
from inspect import signature

import flax.linen as nn
import hedeut as utils
import jax
import jax.numpy as jnp
import jax.scipy as jsp
import jax_dataclasses as jdc
import numpy as np
from scipy import special

from visid import common

PositiveDefiniteRepr = typing.Literal['log_chol', 'ldlt']
"""Representation of a positive-definite matrix."""


class PositiveDefiniteMatrix(abc.ABC):
    """Positive definite matrix base class."""

    @property
    @abc.abstractmethod
    def logdet(self):
        """Logarithm of the matrix determinant."""
        
    @property
    @abc.abstractmethod
    def chol(self):
        """Lower triangular Cholesky factor `L` of the matrix `P = L @ L.T`."""

    @abc.abstractmethod
    def __call__(self):
        """Return the positive-definite matrix."""


@jdc.pytree_dataclass
class LogDiagMatrix(PositiveDefiniteMatrix):
    """Diagonal PD matrix represented by its log-diagonal."""
    
    log_d: jax.Array
    """Logarithm of the diagonal elements."""

    def __post_init__(self):
        """Check that the dimensions are compatible."""
        assert self.log_d.ndim == 1, "Broadcasting not supported yet."

    @property
    def logdet(self):
        """Logarithm of the matrix determinant."""
        return self.log_d.sum()

    @property
    def chol(self):
        """Lower triangular Cholesky factor `S` of the matrix `P = S @ S.T`."""
        sqrt_d = jnp.exp(0.5 * self.log_d)
        return jnp.diag(sqrt_d)
    
    def __call__(self):
        """The underlying positive-definite matrix."""
        return jnp.diag(jnp.exp(self.log_d))


@jdc.pytree_dataclass
class LogCholMatrix(PositiveDefiniteMatrix):
    """PD matrix represented by the matrix logarithm of its Cholesky factor."""
    
    vech_log_chol: jax.Array
    """Elements at and below the main diagonal (using function vech) of the
    matrix logarithm of the Cholesky factor of a positive definite matrix."""

    @property
    def logdet(self):
        """Logarithm of the matrix determinant."""
        return 2 * common.matl_diag(self.vech_log_chol).sum()

    @property
    def chol(self):
        """Lower triangular Cholesky factor `L` of the matrix `P = L @ L.T`."""
        log_chol = common.matl(self.vech_log_chol)
        chol = jsp.linalg.expm(log_chol)
        return chol
    
    def __call__(self):
        """The underlying positive-definite matrix."""
        chol_T = self.chol.swapaxes(-1, -2)
        return self.chol @ chol_T


@jdc.pytree_dataclass
class LDLTMatrix(PositiveDefiniteMatrix):
    """PD matrix represented by its LDL^T decomposition.
    
    The matrix L is unitriangular, and D is diagonal with strictly positive
    entries. Only the elements below the unit diagonal of L are stored. The
    logarithm of the diagonal elements of D are stored.
    """
    
    vech_L: jax.Array
    """Elements strictly below the main diagonal (using function vech) of L."""

    log_d: jax.Array
    """Logarithm of the diagonal elements."""

    def __post_init__(self):
        """Check that the dimensions of L and D are compatible."""
        n_d = self.log_d.shape[-1]
        n_L = common.matl_size(self.vech_L.shape[-1]) + 1
        assert n_d == n_L, 'Incompatible dimensions for L and D.'

    @property
    def logdet(self):
        """Logarithm of the matrix determinant."""
        return self.log_d.sum()

    @property
    def _L(self):
        """The underlying unitriangular matrix `L`."""
        n = self.log_d.shape[-1]
        base_shape = jnp.broadcast_shapes(
            self.log_d.shape[:-1], self.vech_L.shape[:-1]
        )
        L = jnp.zeros(base_shape + (n, n)).at[...].set(jnp.identity(n))
        return L.at[..., 1:, :-1].add(common.matl(self.vech_L))

    @property
    def chol(self):
        """Lower triangular Cholesky factor `S` of the matrix `P = S @ S.T`."""
        sqrt_d = jnp.exp(0.5 * self.log_d)
        return self._L * sqrt_d[..., None, :]
    
    def __call__(self):
        """The underlying positive-definite matrix."""
        D = jnp.exp(self.log_d[..., None, :])
        L = self._L
        return (L * D) @ L.swapaxes(-1, -2)


class LogDiagParam(nn.Module):
    """Positive definite matrix parameter using log-diagonal representation."""

    n: int
    """Dimension of the positive definite matrix."""

    extra_shape: tuple[int, ...] = ()
    """Extra broadcast shape."""

    initializer: nn.initializers.Initializer = nn.initializers.zeros
    """Initializer for the log_d parameter."""

    def setup(self):
        assert self.extra_shape == (), 'extra_shape not supported yet.'
        self.log_d = self.param(
            'log_d', self.initializer, self.extra_shape + (self.n,)
        )
    
    def __call__(self):
        return LogDiagMatrix(self.log_d)


class LogCholParam(nn.Module):
    """Positive definite matrix parameter using log-chol representation."""

    n: int
    """Dimension of the positive definite matrix."""

    extra_shape: tuple[int, ...] = ()
    """Extra broadcast shape."""

    initializer: nn.initializers.Initializer = nn.initializers.zeros
    """Initializer for the vech_log_chol parameter."""

    def setup(self):
        ntril = self.n * (self.n + 1) // 2
        self.vech_log_chol = self.param(
            'vech_log_chol', self.initializer, self.extra_shape + (ntril,)
        )

    def __call__(self):
        return LogCholMatrix(self.vech_log_chol)


class LDLTParam(nn.Module):
    """Positive definite matrix parameter using LDL^T representation."""

    n: int
    """Dimension of the positive definite matrix."""

    extra_shape: tuple[int, ...] = ()
    """Extra broadcast shape."""

    d_initializer: nn.initializers.Initializer = nn.initializers.zeros
    """Initializer for the d parameter."""

    L_initializer: nn.initializers.Initializer = nn.initializers.zeros
    """Initializer for the L parameter."""

    def setup(self):
        ntril = self.n * (self.n - 1) // 2
        self.log_d = self.param(
            'log_d', self.d_initializer, self.extra_shape + (self.n,)
        )
        self.vech_L = self.param(
            'vech_L', self.L_initializer, self.extra_shape + (ntril,)
        )

    def __call__(self):
        return LDLTMatrix(self.vech_L, self.log_d)


def mvn_logpdf_info(x: jax.Array, mu: jax.Array, info: PositiveDefiniteMatrix):
    """Multivariate normal log-density using information matrix."""
    cte = -0.5 * len(x) * jnp.log(2 * jnp.pi) 
    normalized_dev = info.chol.T @ (x - mu)
    return -0.5 * jnp.sum(normalized_dev ** 2) + 0.5 * info.logdet + cte


def ghcub(order, dim):
    """Gauss-Hermite nodes and weights for Gaussian cubature."""
    x, w_unnorm = special.roots_hermitenorm(order)
    w = w_unnorm / w_unnorm.sum()
    xrep = [x] * dim
    wrep = [w] * dim
    xmesh = np.meshgrid(*xrep)
    wmesh = np.meshgrid(*wrep)
    X = np.hstack(tuple(xi.reshape(-1,1) for xi in xmesh))
    W = math.prod(wmesh).flatten()
    return X, W


def sigmapts(dim: int):
    """Sigma points and weights for unscented transform without center point."""
    X = np.r_[np.eye(dim), -np.eye(dim)] * np.sqrt(dim)
    W = 0.5 / dim
    return X, W
