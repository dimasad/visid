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

    @utils.jax_vectorize_method(signature='(x),(u)->(x)')
    def f(self, x, u):
        """State transition function."""
        return x + self.fc(x, u) * self.dt
