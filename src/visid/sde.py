"""Stochastic differential equation (SDE) discretization and utilities."""

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import flax.linen as nn
import hedeut as utils


class Euler(nn.Module):
    """Euler SDE discretization scheme."""

    dt: float = 1.0
    """Discretization time interval."""

    @utils.jax_vectorize_method(signature='(x),(u)->(x)')
    def f(self, x, u):
        """State transition function."""
        return x + self.fc(x, u) * self.dt
