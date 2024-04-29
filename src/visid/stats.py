"""Statistical helper functions."""


import abc
import math
from inspect import signature

import hedeut as utils
import jax
import jax.numpy as jnp
import jax.scipy as jsp
import jax_dataclasses as jdc
import numpy as np
from scipy import special

from visid import common


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


@jdc.dataclass
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
        return self.chol @ self.chol.T


@jdc.dataclass
class LexpDLTMatrix(PositiveDefiniteMatrix):
    """PD matrix represented as `P=L@diag(exp(d))@L.T` for unitriangular L."""
    
    vech_L: jax.Array
    """Elements strictly below the main diagonal (using function vech) of L."""

    d: jax.Array
    """Diagonal elements."""

    @property
    def logdet(self):
        """Logarithm of the matrix determinant."""
        return self.d.sum()

    @property
    def _L(self):
        """The underlying unitriangular matrix `L`."""
        n = common.matl_size(len(self.vech_L)) + 1
        L = jnp.identity(n)
        return L.at[1:, :-1].set(common.matl(self.vech_L))

    @property
    def chol(self):
        """Lower triangular Cholesky factor `L` of the matrix `P = L @ L.T`."""
        sqrt_exp_d = jnp.exp(0.5 * self.d)
        return self._L * sqrt_exp_d[None]
    
    def __call__(self):
        """The underlying positive-definite matrix."""
        exp_D = jnp.diag(jnp.exp(self.d))
        L = self._L
        return L @ exp_D @ L.T


@utils.jax_vectorize(signature='(x),(x),(x,x),()->()')
def mvn_logpdf(x, mu, chol_info, logdet_info):
    """Multivariate normal log-density using decomposed information matrix.
    
    The information matrix satisfies `info == chol_info @ chol_info.T` and
    `logdet_info == log(det(info))`.
    """
    unmasked = ~jnp.isnan(x)
    cte = -0.5 * jnp.log(2 * jnp.pi) * unmasked.sum()
    dev = jnp.where(unmasked, x - mu, 0)
    normalized_dev = chol_info.T @ dev
    ##### How do we mask logdet???
    return -0.5 * jnp.sum(normalized_dev ** 2) + 0.5 * logdet_info + cte


@utils.jax_vectorize(signature='(x),(x),(x,x)->()')
def mvn_logpdf_ichol(x, mu, inv_chol_cov):
    """Multivariate normal log-density with inverse of Cholesky factor of cov.
    
    inv_chol_cov does not have to be triangular, as long as the covariance R
    satisfies `inv(R) == inv_chol_cov.T @ inv_chol_cov`, which is equivalent to
    `R == inv(inv_chol_cov) @ inv(inv_chol_cov).T`.
    """
    unmasked = ~jnp.isnan(x)
    cte = -0.5 * jnp.log(2 * jnp.pi) * unmasked.sum()
    dev = jnp.where(unmasked, x - mu, 0)
    normdev = inv_chol_cov @ dev
    logdet = jnp.sum(jnp.log(inv_chol_cov.diagonal()) * unmasked)
    return -0.5 * jnp.sum(normdev ** 2) + logdet + cte


@utils.jax_vectorize(signature='(x),(x),(x,x)->()')
def mvn_logpdf_logchol(x, mu, log_chol_cov):
    """Multivariate normal log-density with log-chol covariance matrix.
    
    `log_chol_cov` is the matrix logarithm of the Cholesky factor of the 
    covariance matrix.
    """
    inv_chol_cov = jsp.linalg.expm(-log_chol_cov)
    cte = -0.5 * len(x) * jnp.log(2 * jnp.pi)
    dev = x - mu
    normdev = inv_chol_cov @ dev
    return -0.5 * jnp.sum(normdev ** 2) - jnp.trace(log_chol_cov) + cte


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
