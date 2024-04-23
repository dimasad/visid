"""Base classes for variational inference in state-space models."""

import abc
from typing import Literal

import flax.linen as nn
import jax.flatten_util
import jax.numpy as jnp
import jax.scipy as jsp
import jax_dataclasses as jdc
from numpy.typing import NDArray

from . import common, stats

SampleWeights = NDArray | None
"""Sample weights for computing expectation."""


@jdc.pytree_dataclass
class Data:
    """Data for estimation problems."""

    y: NDArray
    """Measurements."""

    u: NDArray
    """Exogenous inputs."""


@jdc.pytree_dataclass
class StatePathPosterior(abc.ABC):

    @abc.abstractmethod
    def entropy(self, xnext: NDArray, xcurr: NDArray, w: SampleWeights):
        """Entropy of the state posterior."""


class VIBase(nn.Module):
    """Base class for Variational Inference estimators."""

    def elbo(self, data: Data, seed=None):
        """Compute the Evidence Lower Bound."""
        # Obtain the posterior assumed density
        posterior = self.smoother(data)

        # Sample from the assumed density
        xmarg, wmarg = self.sampler.marginals(posterior, seed)
        xpair, wpair = self.sampler.pairs(posterior, seed)

        # Compute the elements of the complete-data log-density
        trans = self.model.avg_path_trans_logpdf(*xpair, data.u, wmarg)
        meas = self.model.avg_path_meas_logpdf(data.y, xmarg, data.u, wpair)

        # Compute the entropy
        entropy = posterior.entropy(*xpair, wpair)

        # Add all terms and return
        return trans + meas + entropy
    
    def __call__(self, data: Data, seed=None):
        """Loss function for minimization."""
        return -self.elbo(data, seed)
