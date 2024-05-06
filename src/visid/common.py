"""Common functions and utilities."""


import flax.linen as nn
import hedeut
import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt


class ArrayParam(nn.Module):
    """Array parameter for flax with free and given entries."""

    shape: tuple[int, ...]
    """Array shape."""

    free: np.typing.NDArray | bool = True
    """Which entries of the array are free parameters."""

    given: jax.Array | float = 0.0
    """Given values of the array."""

    initializer: nn.initializers.Initializer = nn.initializers.zeros
    """Initializer for the vech_log_chol parameter."""

    def setup(self):
        self._free = np.broadcast_to(self.free, self.shape)
        """Free entries of the array, broadcasted to `self.shape`."""

        self._given = jnp.broadcast_to(self.given, self.shape)
        """Given values of the array, broadcasted to `self.shape`."""
        
        nfree = self._free.sum()
        self.free_values = self.param('free_values', self.initializer, (nfree,))

    def __call__(self):
        return self._given.at[self._free].set(self.free_values)


@hedeut.jax_vectorize(signature='(n,m)->(p)')
def vech(M: npt.ArrayLike) -> npt.NDArray:
    """Pack the lower triangle of a matrix into a vector, columnwise.
    
    Follows the definition of Magnus and Neudecker (2019), Sec. 3.8, 
    DOI: 10.1002/9781119541219
    """
    return M[jnp.triu_indices_from(M.T)[::-1]]


@hedeut.jax_vectorize(signature='(m)->(n,n)')
def matl(v: npt.ArrayLike) -> npt.NDArray:
    """Unpack a vector into a square lower triangular matrix."""
    assert v.ndim == 1
    n = matl_size(len(v))
    M = jnp.zeros((n, n))
    return M.at[jnp.triu_indices_from(M)[::-1]].set(v)


def matl_size(vech_len: int) -> int:
    """Number of rows a square matrix `M` given the length of `vech(M)`."""
    n = int(np.sqrt(2 * vech_len + 0.25) - 0.5)
    assert n * (n + 1) / 2 == vech_len
    return n


def matl_diag(v: npt.ArrayLike) -> npt.NDArray:
    """Diagonal elements of the entries in the lower triangle a matrix."""
    n = matl_size(len(v))
    i, j = jnp.triu_indices(n)[::-1]
    return v[i == j]


def tria_qr(*args) -> npt.NDArray:
    """Array triangularization routine using QR decomposition."""
    M = jnp.concatenate(args, axis=-1)
    Q, R = jnp.linalg.qr(M.T)
    sig = jnp.sign(jnp.diag(R))
    return R.T * sig


def tria_chol(*args) -> npt.NDArray:
    """Array triangularization routine using Cholesky decomposition."""
    M = jnp.concatenate(args, axis=-1)
    MMT = M @ M.T
    return jnp.linalg.cholesky(MMT)


@hedeut.jax_vectorize(signature='(k,m),(k,n)->(k,k)')
def tria2_qr(m1, m2):
    """Triangularization of two matrices using QR decomposition."""
    return tria_qr(m1, m2)


@hedeut.jax_vectorize(signature='(k,m),(k,n)->(k,k)')
def tria2_chol(m1, m2):
    """Triangularization of two matrices using Cholesky decomposition."""
    return tria_chol(m1, m2)


def vconv(sig, kern, mode):
    """Convolution of vectorized signals and kernels."""
    return jax.vmap(lambda s, k: jnp.convolve(s, k, mode=mode))(sig, kern)

def bvconv(sig, kernels, mode):
    """Vectorized convolution of a signal with batched kernels."""
    return jax.vmap(lambda kern: vconv(sig, kern, mode=mode))(kernels)
