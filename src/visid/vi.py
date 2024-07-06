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

    start: int | None = None
    """Start of data slice."""

    stop: int | None = None
    """Stop of data slice."""

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
        return cls(jnp.asarray(y), jnp.asarray(u), start, stop)
    
    def __getitem__(self, key):
        """Slice the data."""
        assert isinstance(key, slice)
        assert key.step is None

        # Get integer start and stop indices
        start = self.start or 0
        stop = self.stop or len(self.y_buffer)

        # Compute new start
        if key.start is None:
            new_start = start
        elif key.start >= 0:
            new_start = start + key.start
        elif key.start < 0:
            new_start = stop + key.start

        # Compute new stop
        if key.stop is None:
            new_stop = stop
        elif key.stop >= 0:
            new_stop = start + key.stop
        elif key.stop < 0:
            new_stop = stop + key.stop
        
        # Create new object and return
        with jdc.copy_and_mutate(self) as new:
            new.start = new_start
            new.stop = new_stop
        return new
    
    def pad(self, nkern):
        """Pad the data for convolution."""
        npad = nkern // 2

        # Get integer start and stop indices
        start = self.start or 0
        stop = self.stop or len(self.y_buffer)

        # Check that there is enough data to pad
        assert start > npad
        assert stop < len(self.y_buffer) - npad

        # Create padded object and return
        with jdc.copy_and_mutate(self) as padded:
            padded.start = start - npad
            padded.stop = stop + npad
        return padded


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
