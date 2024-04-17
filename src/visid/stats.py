"""Statistical helper functions."""


from inspect import signature
import jax.numpy as jnp
import jax.scipy as jsp
import hedeut as utils
import numpy as np
import math
from scipy import special


@utils.jax_vectorize(signature='(x),(x),(x,x)->()')
def mvn_logpdf_ichol(x, mu, inv_chol_cov):
    """Multivariate normal log-density with inverse of Cholesky factor of cov.
    
    inv_chol_cov does not have to be triangular, as long as the covariance R
    satisfies `inv(R) == inv_chol_cov.T @ inv_chol_cov`, which is equivalent to
    `R == inv(inv_chol_cov) @ inv(inv_chol_cov).T`.
    """
    unmasked = ~jnp.isnan(x)
    cte = -0.5 * jnp.log(2 * jnp.pi) * unmasked.sum()
    dev = (x - mu) * unmasked
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
