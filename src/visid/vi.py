"""Base classes for variational inference in state-space models."""

import abc
from typing import Literal

import flax.linen as nn
import jax.flatten_util
import jax.numpy as jnp
import jax.scipy as jsp
import jax_dataclasses as jdc
import numpy as np
from jax import Array

from . import common, stats


SampleWeights = Array | None
"""Sample weights for computing expectation."""


@jdc.pytree_dataclass
class Data:
    """Data for estimation problems."""

    y: Array
    """Measurements."""

    u: Array
    """Exogenous inputs."""

    def __len__(self):
        """Number of samples."""
        assert len(self.y) == len(self.u)
        return len(self.y)
    
    def split(self, n):
        """Split the dataset into `n` sections of equal or near-equal size."""
        y_split = jnp.array_split(self.y, n)
        u_split = jnp.array_split(self.u, n)
        return [Data(y, u) for y, u in zip(y_split, u_split)]

    def pad(self, n):
        unpadded = np.s_[n:-n]
        return PaddedData(self.y[unpadded], self.u[unpadded], self)


@jdc.pytree_dataclass
class PaddedData(Data):
    """Data with a padded buffer for, e.g., convolution."""

    padded: Data
    """The buffer augmented with padded start and end."""

    @property
    def npad(self):
        """Number of samples padded to both ends."""
        nextra = len(self.padded) - len(self)
        assert nextra % 2 == 0, "Padding at left and right must be the same."
        return nextra // 2

    def split(self, n):
        npad = self.npad
        unpadded = super().split(n)
        unpadded_lens = np.array([len(d) for d in unpadded])
        padded_starts = np.r_[0, np.cumsum(unpadded_lens[:-1])]
        padded_stops = padded_starts + unpadded_lens + 2*npad
        
        ret = []
        for base, start, stop in zip(unpadded, padded_starts, padded_stops):
            padded_y = self.padded.y[start:stop]
            padded_u = self.padded.u[start:stop]
            ret.append(PaddedData(base.y, base.u, Data(padded_y, padded_u)))
        return ret


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


def multiseg_init(estimator, data, key):
    """Initialize an estimator for multiple data segments."""
    # Split the PRNG keys
    subkeys = jnp.array(jax.random.split(key, len(data)))

    # Initialize the estimator
    v = [estimator.init(k, d) for k, d in zip(subkeys, data)]

    # Remove duplicate model parameters
    for vseg in v[1:]:
        vseg['params'].pop('model')
    return v


def multiseg_cost(estimator, v, data):
    """Cost function for the optimization of multiple data segments."""
    # Copy the model parameters to all segments
    for vseg in v[1:]:
        vseg['params']['model'] = v[0]['params']['model']

    # Return sum of cost of all segments
    return sum(estimator.apply(vseg, dataseg) for vseg, dataseg in zip(v, data))
