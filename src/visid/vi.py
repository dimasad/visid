"""Base classes for variational inference in state-space models."""

import abc

import flax.linen as nn
import jax.flatten_util
import jax.numpy as jnp
import jax.scipy as jsp
import jax_dataclasses as jdc
import numpy as np
from jax import Array
from scipy import optimize

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


def fixed_model_init(estimator, data, key):
    """Initialize an estimator with fixed model parameters."""
    # Initialize the estimator
    v = estimator.init(key, data)

    # Remove model parameters
    v['params'].pop('model')
    return v

def fixed_model_cost(estimator, v, data, model_params):
    """Cost function of estimator with fixed model parameters."""
    v['params']['model'] = model_params
    return estimator.apply(v, data)


class Optimizer:
    """Deterministic optimization interface for variational inference."""

    def __init__(self, estimator, v, data, multiseg=False, model_params=None):
        dec0, unravel = jax.flatten_util.ravel_pytree(v)
        if multiseg:
            cost = lambda dec: multiseg_cost(estimator, unravel(dec), data)
        elif model_params is not None:
            cost = lambda dec: fixed_model_cost(
                estimator, unravel(dec), data, model_params
            )
        else:
            cost = lambda dec: estimator.apply(unravel(dec), data)
        

        self.dec0 = dec0
        """Initial decision variable vector."""

        self.unravel = unravel
        """Unravel the decision vector into a flax module pytree."""

        self.cost = jax.jit(cost)
        """Cost function."""

        self.grad = jax.jit(jax.grad(cost))
        """Gradient of cost function."""

        self.hess = jax.jit(jax.jacfwd(self.grad))
        """Hessian of cost function."""
        
        hvp = jax.jit(lambda dec, v: jax.jvp(self.grad, (dec,), (v,))[1])
        self.hvp = lambda dec, v: hvp(dec, jnp.asarray(v, float))
        """Product of cost function Hessian with vector."""

    @staticmethod
    def ravel(v):
        return jax.flatten_util.ravel_pytree(v)[0]
    
    def __call__(self, **kwargs):
        sol = optimize.minimize(
            self.cost, self.dec0, jac=self.grad, hessp=self.hvp, **kwargs
        )
        v = self.unravel(sol.x)
        return v, sol
