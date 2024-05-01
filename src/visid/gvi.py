"""Gaussian variational inference for state-space models."""

import abc
from typing import Literal

import flax.linen as nn
import jax.flatten_util
import jax.numpy as jnp
import jax.scipy as jsp
import jax_dataclasses as jdc

from . import common, stats, vi
from .vi import Data, SampleWeights, StatePathPosterior


TriaRoutine = Literal['qr', 'chol']
"""Matrix triangularization routine option."""


@jdc.pytree_dataclass
class GaussianStatePathBase(StatePathPosterior):
    """Base class for Gaussian state-path posteriors."""

    mu: jax.Array
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

    mu: jax.Array
    """State path mean."""

    cond_cov: stats.PositiveDefiniteMatrix
    """Conditional covariance of the state at a time, given the previous."""

    norm_cross_cov: jax.Array
    """Normalized cross covariance between consecutive states."""

    tria: jdc.Static[TriaRoutine] = 'qr'
    """Matrix triangularization routine."""

    def scale_marg_samples(self, norm_dev):
        return jnp.inner(norm_dev, self.chol_marg_cov)

    def scale_xpair_samples(self, next_us_dev, curr_us_dev):
        xcurr_dev = jnp.inner(curr_us_dev, self.chol_marg_cov)
        xnext_dev = (jnp.inner(curr_us_dev, self.norm_cross_cov)
                     + jnp.inner(next_us_dev, self.cond_cov.chol))
        return (xnext_dev, xcurr_dev)
    
    @property
    def chol_marg_cov(self):
        """Cholesky factor of the marginal covariance."""
        if self.tria == 'qr':
            return common.tria2_qr(self.cond_cov.chol, self.norm_cross_cov)
        elif self.tria == 'chol':
            return common.tria2_chol(self.cond_cov.chol, self.norm_cross_cov)
        else:
            raise ValueError("Invalid triangularization routine.")
    
    def entropy(self, xnext: jax.Array, xcurr: jax.Array, w: SampleWeights):
        """Entropy of the state-path posterior."""
        cte = 0.5 * jnp.size(self.mu) * jnp.log(2 * jnp.pi * jnp.e)
        logdet = self.cond_cov.logdet
        if logdet.ndim == 0:
            N = len(self.mu)
            path_logdet = logdet * N
        else:
            path_logdet = jnp.sum(logdet, -1)
        return 0.5 * path_logdet + cte


class LinearConvolutionSmoother(nn.Module):
    """Linear convolution smoother."""

    nkern: int
    """Length of the convolution kernel."""

    nx: int
    """Number of states."""

    conv_mode: Literal['full', 'same', 'valid'] = 'valid'
    """Mode argument passed to `jax.numpy.convolve`."""

    cov_repr: stats.PositiveDefiniteRepr = 'ldlt'
    """Representation of the covariance matrix."""

    tria: TriaRoutine = 'qr'
    """Matrix triangularization routine."""

    @jdc.pytree_dataclass
    class Data(vi.Data):
        """Data for linear convolution smoother."""

        conv_y: jax.Array
        """Measurements for convolution."""

        conv_u: jax.Array
        """Exogenous inputs for convolution."""

    def setup(self):
        nx = self.nx
        self.norm_cross_cov = self.param(
            'norm_cross_cov', nn.initializers.zeros, (nx, nx)
        )
        if self.cov_repr == 'ldlt':
            self.cond_cov = stats.LDLTParam(nx)
        elif self.cov_repr == 'log_chol':
            self.cond_cov = stats.LogCholParam(nx)
        else:
            raise ValueError("Invalid covariance representation.")

    @nn.compact
    def __call__(self, data: Data):
        """Apply the linear convolution smoother."""
        # Retrieve and concatenate the convolution inputs
        u = getattr(data, 'conv_u', data.u)
        y = getattr(data, 'conv_y', data.y)
        y_masked = jnp.where(jnp.isnan(y), 0, y)
        sig = jnp.c_[y_masked, u].T

        # Retrieve and initialize the convolution kernel
        K_shape = (self.nx, len(sig), self.nkern)
        K = self.param('K', nn.initializers.normal(), K_shape)

        # Apply kernels and sum to obtain mean
        mu = common.bvconv(sig, K, mode=self.conv_mode).sum(1).T

        return GaussianSteadyStatePosterior(
            mu=mu, cond_cov=self.cond_cov(), norm_cross_cov=self.norm_cross_cov,
            tria=self.tria
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
