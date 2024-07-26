"""Base classes for variational inference in state-space models."""

import abc
from typing import Literal

import flax.linen as nn
import jax.flatten_util
import jax.numpy as jnp
import jax.scipy as jsp
import jax_dataclasses as jdc
from jax import Array
from jax.experimental import checkify

from . import common, stats

SampleWeights = Array | None
"""Sample weights for computing expectation."""


@jdc.pytree_dataclass
class Data:
    """Data for estimation problems."""

    y_buffer: Array
    """Measurements."""

    u_buffer: Array
    """Exogenous inputs."""

    start: int = 0
    """Start of data slice (nonnegative)."""

    stop: int
    """Stop of data slice (nonnegative)."""

    def __len__(self):
        """Number of samples."""
        return len(self.y)
    
    @property
    def y(self):
        """Measurements."""
        return self.y_buffer[self.start:self.stop]

    @property
    def u(self):
        """Measurements."""
        return self.y_buffer[self.start:self.stop]
    
    @classmethod
    def buffer(cls, y, u, start=None, stop=None):
        """Convert to jax.Array and build object from buffer."""
        # Deal with optional arguments
        start = 0 if start is None else start
        stop = len(y) if stop is None else stop

        # Check inputs
        assert len(y) == len(y)
        assert start >= 0
        assert stop >= 0
        assert stop <= len(y)

        # Return object
        return cls(jnp.asarray(y), jnp.asarray(u), start, stop)
    
    def enlarge(self, n):
        """Enlarge data slice by reducing start and increasing stop by n."""
        start = self.start - n
        stop = self.stop + n
        return self.buffer(self.y_buffer, self.u_buffer, start, stop)

    def shrink(self, n):
        """Shrink data slice by increasing start and decreasing stop by n."""
        return self.enlarge(-n)


@jdc.pytree_dataclass
class StatePathPosterior(abc.ABC):

    @abc.abstractmethod
    def entropy(self, xnext: Array, xcurr: Array, w: SampleWeights):
        """Entropy of the state posterior."""


class VIBase(nn.Module):
    """Base class for Variational Inference estimators."""

    def elbo(self, data: Data, seed=None):
        """Compute the Evidence Lower Bound."""
        # Obtain the posterior assumed density
        posterior = self.smoother(data)

        # Sample from the assumed density
        xmarg, wmarg = self.sampler.marginals(posterior, seed)
        *xpair, wpair = self.sampler.pairs(posterior, seed)

        # Compute the elements of the complete-data log-density
        trans = self.model.avg_path_trans_logpdf(*xpair, data.u[:-1], wpair)
        meas = self.model.avg_path_meas_logpdf(data.y, xmarg, data.u, wmarg)

        # Compute the entropy
        entropy = posterior.entropy(*xpair, wpair)

        # Add all terms and return
        return trans + meas + entropy
    
    def __call__(self, data: Data, seed=None):
        """Loss function for minimization."""
        return -self.elbo(data, seed)
