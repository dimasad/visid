"""Gaussian variational inference for state-space models."""

import abc
from typing import Literal

import flax.linen as nn
import jax.flatten_util
import jax.numpy as jnp
import jax.scipy as jsp
import jax_dataclasses as jdc
from numpy.typing import NDArray

from . import common, stats, vi
from .vi import Data, SampleWeights, StatePathPosterior

CovarianceRepr = Literal['log_chol', 'L_expD']
"""Covariance matrix representation.

- 'L_expD' stands for the `R = L @ expm(D) @ L.T` decomposition of the
  covariance matrix, where L is unitriangular and D is diagonal.
- 'log_chol' stands for the lower-triangular matrix logarithm of the Cholesky 
  factor of the covariance matrix R, that is,
  `R == expm(log_chol) @ expm(log_chol).T`.
"""

TriaRoutine = Literal['qr', 'chol']
"""Matrix triangularization routine option."""


@jdc.pytree_dataclass
class GaussianStatePathBase(StatePathPosterior):
    """Base class for Gaussian state-path posteriors."""

    mu: NDArray
    """State samples."""

    def sample_marg(self, us_dev):
        """Sample the states from the marginal posterior at each time index."""
        dev = self.scale_marg_samples(us_dev)
        if dev.ndim == 2:
            dev = dev[:, None]
        return self.mu + dev

    def sample_pairs(self, next_us_dev, curr_us_dev):
        """Sample the states from the marginal posterior at each time index."""
        dev_pair = self.scale_xpair_samples(next_us_dev, curr_us_dev)
        if dev_pair[0].ndim == 2:
            dev_pair = [dev[:, None] for dev in dev_pair]
        xnext = self.mu[1:] + dev_pair[0]
        xcurr = self.mu[:-1] + dev_pair[1]
        return xnext, xcurr

    @abc.abstractmethod
    def scale_marg_samples(self, norm_dev):
        """Scale normalized deviations from mean by the state marginals."""

    @abc.abstractmethod
    def scale_xpair_samples(self, next_us_dev, curr_us_dev):
        """Scale normalized deviations from mean of consecutive state pairs."""


@jdc.pytree_dataclass
class GaussianSteadyStatePosterior(GaussianStatePathBase):
    """Gaussian steady-state state-path using representation."""

    mu: NDArray
    """State path mean."""

    cond_scale: tuple[NDArray, NDArray] | NDArray
    """Scaling of the conditional state of a time sample, given the previous."""

    cross: NDArray
    """Normalized cross-correlation between consecutive states."""

    cov_repr: jdc.Static[CovarianceRepr] = 'L_expD'
    """Representation of the covariance matrix."""

    tria: jdc.Static[TriaRoutine] = 'qr'
    """Matrix triangularization routine."""

    def scale_marg_samples(self, norm_dev):
        return jnp.inner(norm_dev, self.chol_marg_cov)

    def scale_xpair_samples(self, next_us_dev, curr_us_dev):
        xcurr_dev = jnp.inner(curr_us_dev, self.chol_marg_cov)
        xnext_dev = (jnp.inner(curr_us_dev, self.cross)
                     + jnp.inner(next_us_dev, self.chol_cond_cov))
        return (xnext_dev, xcurr_dev)
    
    @property
    def chol_cond_cov(self):
        """Cholesky factor of the conditional covariance."""
        if self.cov_repr == 'L_expD':
            L, d = self.cond_scale
            L = jnp.tril(L, k=-1) + jnp.identity(len(d))
            return L @ jnp.diag(jnp.exp(0.5 * d))
        elif self.cov_repr == 'log_chol':
            return jsp.linalg.expm(jnp.tril(self.cond_scale))
        else:
            raise ValueError("Invalid covariance representation.")

    @property
    def chol_marg_cov(self):
        """Cholesky factor of the marginal covariance."""
        if self.tria == 'qr':
            return common.tria2_qr(self.chol_cond_cov, self.cross)
        elif self.tria == 'chol':
            return common.tria2_chol(self.chol_cond_cov, self.cross)
        else:
            raise ValueError("Invalid triangularization routine.")
    
    def entropy(self, xnext: NDArray, xcurr: NDArray, w: SampleWeights):
        """Entropy of the state-path posterior."""
        N = len(self.mu)
        if self.cov_repr == 'L_expD':
            L, d = self.cond_scale
            return N / 2  * jnp.sum(d)
        elif self.cov_repr == 'log_chol':
            return N * jnp.trace(self.cond_scale)
        else:
            raise ValueError("Invalid covariance representation.")


class LinearConvolutionSmoother(nn.Module):
    """Linear convolution smoother."""

    nkern: int
    """Length of the convolution kernel."""

    nx: int
    """Number of states."""

    conv_mode: Literal['full', 'same', 'valid'] = 'valid'
    """Mode argument passed to `jax.numpy.convolve`."""

    cov_repr: CovarianceRepr = 'L_expD'
    """Representation of the covariance matrix."""

    tria: TriaRoutine = 'qr'
    """Matrix triangularization routine."""

    @jdc.pytree_dataclass
    class Data(vi.Data):
        """Data for linear convolution smoother."""

        conv_u: NDArray
        """Exogenous inputs for convolution."""

        conv_y: NDArray
        """Measurements for convolution."""


    def setup(self):
        nx = self.nx
        self.cross = self.param('cross', nn.initializers.zeros, (nx, nx))
        if self.cov_repr == 'L_expD':
            L = self.param('L', nn.initializers.zeros, (nx, nx))
            d = self.param('d', nn.initializers.zeros, (nx,))
            self.cond_scale = (L, d)
        elif self.cov_repr == 'log_chol':
            self.cond_scale = self.param('S', nn.initializers.zeros, (nx, nx))
        else:
            raise ValueError("Invalid covariance representation.")

    @nn.compact
    def __call__(self, data: Data):
        """Apply the linear convolution smoother."""
        # Retrieve and concatenate the convolution inputs
        u = getattr(data, 'conv_u', data.u)
        y = getattr(data, 'conv_y', data.y)
        sig = jnp.c_[y, u].T

        # Retrieve and initialize the convolution kernel
        K_shape = (self.nx, len(sig), self.nkern)
        K = self.param('K', nn.initializers.normal(), K_shape)

        # Apply kernels and sum to obtain mean
        mu = common.bvconv(sig, K, mode=self.conv_mode).sum(1).T

        return GaussianSteadyStatePosterior(
            mu=mu, cond_scale=self.cond_scale, cross=self.cross,
            cov_repr=self.cov_repr, tria=self.tria
        )


class SigmaPointSampler(nn.Module):
    """Sigma point sampler for Gaussian distributions."""

    nx: int
    """Number of states."""

    def marginals(self, posterior: GaussianStatePathBase, seed=None):
        """Sample from the assumed density."""
        us_dev, w = stats.sigmapts(self.nx)
        x = posterior.sample_marg(us_dev)
        return x, w

    def pairs(self, posterior: GaussianStatePathBase, seed=None):
        """Sample from the assumed density."""
        us_dev, w = stats.sigmapts(2*self.nx)
        next_us_dev = us_dev[:, :self.nx]
        curr_us_dev = us_dev[:, self.nx:]
        xpair = posterior.sample_pairs(next_us_dev, curr_us_dev)
        return *xpair, w
