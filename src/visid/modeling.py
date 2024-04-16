"""Classes for representing dynamic system models."""


import hedeut
import jax.numpy as jnp
import jax.scipy as jsp
import flax.linen as nn

from . import common, stats


class GaussianModel(nn.Module):
    """Discrete-time dynamic system model with Gaussian noise."""

    @hedeut.jax_vectorize_method(signature='(x),(x),(u)->()')
    def trans_logpdf(self, xnext, x, u):
        mean = self.f(x, u)
        return stats.mvn_logpdf_logchol(xnext, mean, self.lsQ(x))

    @hedeut.jax_vectorize_method(signature='(y),(x),(u)->()')
    def meas_logpdf(self, y, x, u):
        mean = self.h(x, u)
        return stats.mvn_logpdf_logchol(y, mean, self.lsR(q, x))

    @hedeut.jax_vectorize_method(signature='(x)->()')
    def prior_logpdf(self, x):
        return 0

    def isQ(self, q, x=None):
        """Inverse of the Cholesky factor of the Q matrix."""
        return jsp.linalg.expm(-self.lsQ(q, x))

    def sQ(self, q, x=None):
        """Cholesky factor of the Q matrix."""
        return jsp.linalg.expm(self.lsQ(q, x))

    def Q(self, q, x=None):
        """Process noise covariance matrix."""
        sQ = self.sQ(q, x)
        return sQ @ sQ.T
    
    def isR(self, q, x=None):
        """Inverse of the Cholesky factor of the R matrix."""
        return jsp.linalg.expm(-self.lsR(q, x))

    def sR(self, q, x=None):
        """Cholesky factor of the R matrix."""
        return jsp.linalg.expm(self.lsR(q, x))

    def R(self, q, x=None):
        """Measurement noise covariance matrix."""
        sR = self.sR(q, x)
        return sR @ sR.T


class LinearGaussianModel(GaussianModel):
    """Discrete-time linear dynamic system model with Gaussian noise."""

    def __init__(self, nx, nu, ny, init_packer=True):
        self.nx = nx
        """Number of states."""

        self.nu = nu
        """Number of exogenous inputs."""

        self.ny = ny
        """Number of outputs."""

        self.ntrilx = nx * (nx + 1) // 2
        """Number of elements in lower triangle of nx by nx matrix."""

        self.ntrily = ny * (ny + 1) // 2
        """Number of elements in lower triangle of ny by ny matrix."""

        self.q_packer = hedeut.Packer(
            A=(nx, nx),
            B=(nx, nu),
            C=(ny, nx),
            D=(ny, nu),
            vech_log_sQ=(self.ntrilx,),
            vech_log_sR=(self.ntrily,),
        )

        self.nq = self.q_packer.size
        """Number of classical (deterministic) parameters."""

    def A(self):
        return self.q_packer.unpack(q)['A']

    def B(self):
        return self.q_packer.unpack(q)['B']

    def C(self):
        return self.q_packer.unpack(q)['C']

    def D(self):
        return self.q_packer.unpack(q)['D']
    
    def lsQ(self, x=None):
        vech_log_sQ = self.q_packer.unpack(q)['vech_log_sQ']
        return common.matl(vech_log_sQ)
    
    def lsR(self, x=None):
        vech_log_sR = self.q_packer.unpack(q)['vech_log_sR']
        return common.matl(vech_log_sR)

    def ubias(self):
        """Bias in the input for representing an affine system."""
        return 0

    def ybias(self):
        """Bias in the output for representing an affine system."""
        return 0

    @hedeut.jax_vectorize_method(signature='(x),(u)->(x)')
    def f(self, x, u):
        return self.A(q) @ x + self.B(q) @ (u - self.ubias(q))

    @hedeut.jax_vectorize_method(signature='(x),(u)->(y)')
    def h(self, x, u):
        return self.C(q) @ x + self.D(q) @ (u - self.ubias(q)) + self.ybias(q)

