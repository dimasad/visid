"""Stochastic differential equation (SDE) discretization."""

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import flax.linen as nn
import hedeut as utils


class EulerMaruyamaScheme(nn.Module):
    """Euler-Maruyama SDE discretization scheme."""

    dt: float
    """Discretization time interval."""

    SDE: type
    """SDE model class."""

    SDE_args: tuple = ()
    """Arguments for SDE model class constructor."""

    SDE_kwargs: dict = {}
    """Keyword arguments for SDE model class constructor."""

    nx: int = None
    """Number of states."""

    def setup(self):
        self.sde = self.SDE(*self.SDE_args, **self.SDE_kwargs)
        if self.nx is None:
            self.nx = self.sde.nx

    @utils.jax_vectorize_method(signature='(x),(u)->(x)')
    def f(self, x, u):
        """State transition function."""
        return x + self.sde.f(x, u) * self.dt
