"""Estimation problems and algorithms."""


import abc
import functools
import typing
from typing import Literal

import flax.linen as nn
import hedeut
import jax
import jax.flatten_util
import jax.numpy as jnp
import jax.scipy as jsp
import jax_dataclasses as jdc
import numpy as np
import numpy.typing as npt
from scipy import sparse

from . import common, stats


@jdc.pytree_dataclass
class Data:
    """Data for estimation problems."""

    y: npt.NDArray
    """Measurements."""

    u: npt.NDArray
    """Exogenous inputs."""


@jdc.pytree_dataclass
class StatePathPosterior(abc.ABC):

    @property
    @abc.abstractmethod
    def entropy(self):
        """Entropy of the state posterior."""


class VIBase:
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

        # Add all terms and return
        return trans + meas + posterior.entropy
    
    def __call__(self, data: Data, seed=None):
        """Loss function for minimization."""
        return -self.elbo(data, seed)


@jdc.pytree_dataclass
class GaussianStatePathBase(StatePathPosterior):
    """Base class for Gaussian state-path posteriors."""

    mu: npt.NDArray
    """State samples."""

    def sample_marg(self, norm_dev):
        """Sample the states from the marginal posterior at each time index."""
        dev = self.scale_marg_samples(norm_dev)
        if dev.ndim == 2:
            dev = dev[:, None]
        return self.mu + dev

    def sample_xpair(self, norm_dev_next, norm_dev_curr):
        """Sample the states from the marginal posterior at each time index."""
        dev_pair = self.scale_marg_samples(norm_dev_next, norm_dev_curr)
        if dev_pair[0].ndim == 2:
            dev_pair = [dev[:, None] for dev in dev_pair]
        return [self.mu + dev for dev in dev_pair]

    @abc.abstractmethod
    def scale_marg_samples(self, norm_dev):
        """Scale normalized deviations from mean by the state marginals."""

    @abc.abstractmethod
    def scale_xpair_samples(self, norm_dev_next, norm_dev_curr):
        """Scale normalized deviations from mean of consecutive state pairs."""


CovarianceRepr = Literal['log_chol'] | Literal['L_expD']
"""Covariance matrix representation.

- 'L_expD' stands for the `R = L @ expm(D) @ L.T` decomposition of the
  covariance matrix, where L is unitriangular and D is diagonal.
- 'log_chol' stands for the lower-triangular matrix logarithm of the Cholesky 
  factor of the covariance matrix R, that is,
  `R == expm(log_chol) @ expm(log_chol).T`.
"""

Tria = Literal['qr'] | Literal['chol']
"""Matrix triangularization routine."""

@jdc.pytree_dataclass
class GaussianSteadyStatePosteriorBase(GaussianStatePathBase):
    """Gaussian steady-state state-path using representation."""

    mu: npt.NDArray
    """State path mean."""

    cond_scale: tuple[npt.NDArray, npt.NDArray] | npt.NDArray
    """Scaling of the conditional state of a time sample, given the previous."""

    cross: npt.NDArray
    """Normalized cross-correlation between consecutive states."""

    cov_repr: jdc.Static[CovarianceRepr] = 'L_expD'
    """Representation of the covariance matrix."""

    tria = jdc.Static[Tria] = 'qr'
    """Matrix triangularization routine."""

    def scale_marg_samples(self, norm_dev):
        return jnp.inner(norm_dev, self.chol_marg_cov)

    def scale_xpair_samples(self, norm_dev_next, norm_dev_curr):
        return (jnp.inner(norm_dev_curr, self.cross)
                + jnp.inner(norm_dev_next, self.chol_cond_cov))
    
    @property
    def chol_cond_cov(self):
        """Cholesky factor of the conditional covariance."""
        if self.cov_repr == 'L_expD':
            L, d = self.cond_scale
            L = jnp.tril(L, k=-1) + jnp.identity(len(d))
            return L @ jnp.diag(jnp.exp(0.5 * d))
        elif self.cov_repr == 'log_chol':
            return jsp.linalg.expm(jnp.tril(self.cond_scale))
        else:
            raise ValueError("Invalid covariance representation.")

    @property
    def chol_marg_cov(self):
        """Cholesky factor of the marginal covariance."""
        if self.tria == 'qr':
            return common.tria2_qr(self.chol_cond_cov, self.cross)
        elif self.tria == 'chol':
            return common.tria2_chol(self.chol_cond_cov, self.cross)
        else:
            raise ValueError("Invalid triangularization routine.")
        
    @property
    def entropy(self):
        """Entropy of the state-path posterior."""
        N = len(self.mu)
        if self.cov_repr == 'L_expD':
            L, d = self.cond_scale
            return N / 2  * jnp.sum(d)
        elif self.cov_repr == 'log_chol':
            return N * jnp.trace(self.cond_scale)
        else:
            raise ValueError("Invalid covariance representation.")


@jdc.pytree_dataclass
class XCoeff:
    """Coefficients for expectation wrt the state at each time sample."""

    us_dev: npt.NDArray
    """Unscaled deviations of the state."""

    w: npt.NDArray | float
    """Expectation weights."""


@jdc.pytree_dataclass
class XPairCoeff:
    """Coefficients for expectation wrt pairs of consecutive states."""

    curr_us_dev: npt.NDArray
    """Unscaled deviations of the current state."""

    next_us_dev: npt.NDArray
    """Unscaled deviations of the next state."""

    w: npt.NDArray
    """Expectation weights."""


class ExpectationCoeff(typing.NamedTuple):
    """Coefficients for expectation wrt posterior distributions."""
    x: XCoeff
    xpair: XPairCoeff


class GVI:
    """Base for Gaussian Variational Inference estimators."""

    tria2 = staticmethod(common.tria2_qr)
    """Matrix triangularization routine."""

    def __init__(self, model, elbo_multiplier=1):
        self.model = model
        """The underlying dynamical system model."""

        nx = self.model.nx
        self.ntrilx = nx * (nx + 1) // 2
        """Number of elements in lower triangle of nx by nx matrix."""

        self.elbo_multiplier = elbo_multiplier
        """Multiplier for the ELBO (to use with minimizers)."""

    def sample_x(self, v: GVIProblemVariables, coeff: XCoeff):
        """Sample the state from the assumed density."""
        # Obtain scaled deviation from mean
        x_dev = jnp.inner(coeff.us_dev, v.S)

        if x_dev.ndim == 2:
            x_dev = x_dev[:, None]

        # Add mean to deviation and return
        return v.xbar + x_dev

    def sample_xpair(self, v: GVIProblemVariables, coeff: XPairCoeff):
        """Sample the a pair of consecutive states from the assumed density."""
        # Get the square-root correlations with correct shape
        steady_state = v.S_cond.ndim == 2
        S = v.S[None] if steady_state else v.S[:-1]
        S_cond = v.S_cond[None] if steady_state else v.S_cond[1:]
        S_cross = v.S_cross[None] if steady_state else v.S_cross

        # Obtain scaled deviations from mean
        xcurr_dev = jnp.inner(coeff.curr_us_dev, S)
        xnext_dev = (jnp.inner(coeff.curr_us_dev, S_cross)
                     + jnp.inner(coeff.next_us_dev, S_cond))

        # Add mean to deviations and return
        xcurr = v.xbar[:-1] + xcurr_dev
        xnext = v.xbar[1:] + xnext_dev
        return xnext, xcurr

    def elbo(self, dec, data: Data, coeff: ExpectationCoeff):
        # Compute problem variables
        v: GVIProblemVariables = self.problem_variables(dec, data)

        # Sample from the assumed density
        x = self.sample_x(v, coeff.x)
        xnext, xcurr = self.sample_xpair(v, coeff.xpair)

        # Get the data variables
        u = getattr(v, 'u', data.u)
        y = getattr(v, 'y', data.y)

        # Compute elements of the ELBO
        model = self.model
        entropy = self.entropy(v)
        prior_logpdf = model.prior_logpdf(x[:, 0], dec.q)
        meas_logpdf = model.meas_logpdf(y, x, u, dec.q)
        trans_logpdf = model.trans_logpdf(xnext, xcurr, u[:-1], dec.q)

        # Get the average log densities
        avg_prior_logpdf = (prior_logpdf * coeff.x.w).sum(0)
        avg_meas_logpdf = (meas_logpdf.sum(-1) * coeff.x.w).sum(0)
        avg_trans_logpdf = (trans_logpdf.sum(-1) * coeff.xpair.w).sum(0)

        elbo = entropy + avg_prior_logpdf + avg_meas_logpdf + avg_trans_logpdf
        return self.elbo_multiplier * elbo

    @functools.cached_property
    def elbo_grad(self):
        return jax.grad(self.elbo)
    
    def elbo_hvp(self, dec, dec_d, data: Data, coeff: ExpectationCoeff):
        primals = dec, data, coeff
        duals = dec_d, data.zeros_like(), coeff.zeros_like()
        return jax.jvp(self.elbo_grad, primals, duals)[1]
    
    def elbo_packed(self, dvec, data: Data, coeff: ExpectationCoeff, packer):
        dec = self.Decision(**packer.unpack(dvec))
        return self.elbo(dec, data, coeff)

    def elbo_grad_packed(self, dvec, data: Data, coeff: ExpectationCoeff,
                         packer):
        dec = self.Decision(**packer.unpack(dvec))
        grad = self.elbo_grad(dec, data, coeff)
        return packer.pack(*grad)

    def elbo_hvp_packed(self, dvec, dvec_d, data:Data, coeff:ExpectationCoeff,
                        packer):
        dec = self.Decision(**packer.unpack(dvec))
        dec_d = self.Decision(**packer.unpack(dvec_d))
        hvp = self.elbo_hvp(dec, dec_d, data, coeff)
        return packer.pack(*hvp)

    def fix_and_jit(self, fname: str, /, **kwargs):
        f = functools.partial(getattr(self, fname), **kwargs)
        return jax.jit(f)


class SteadyState(GVI):
    class Decision(typing.NamedTuple):
        """Problem decision variables."""
        q: npt.NDArray
        xbar: npt.NDArray
        vech_log_S_cond: npt.NDArray
        S_cross: npt.NDArray

    def packer(self, N):
        return hedeut.Packer(
            q=(self.model.nq,),
            xbar=(N, self.model.nx),
            vech_log_S_cond=(self.ntrilx,),
            S_cross=(self.model.nx, self.model.nx),
        )

    def problem_variables(self, dec: Decision, data: Data): 
        log_S_cond = common.matl(dec.vech_log_S_cond)
        S_cond = jsp.linalg.expm(log_S_cond)
        S = self.tria2(S_cond, dec.S_cross)
        return GVIProblemVariables(
            xbar=dec.xbar,
            log_S_cond=log_S_cond,
            S=S,
            S_cond=S_cond,
            S_cross=dec.S_cross,
        )
    
    def entropy(self, v: GVIProblemVariables):
        N = len(v.xbar)

        # Compute initial state entropy
        entro_x0 = jnp.sum(jnp.log(jnp.abs(jnp.diag(v.S))))

        # Compute entropy of the remaining states
        entro_xrem = (N - 1) * jnp.trace(v.log_S_cond)

        # Return joint entropy
        return entro_x0 + entro_xrem


class Transient(GVI):
    class Decision(typing.NamedTuple):
        """Problem decision variables."""
        q: npt.NDArray
        xbar: npt.NDArray
        vech_log_S_cond: npt.NDArray
        S_cross: npt.NDArray

    def packer(self, N: int):
        return hedeut.Packer(
            q=(self.model.nq,),
            xbar=(N, self.model.nx),
            vech_log_S_cond=(N, self.ntrilx),
            S_cross=(N-1, self.model.nx, self.model.nx),
        )

    def problem_variables(self, dec: Decision, data: Data): 
        nx = self.model.nx

        log_S_cond = common.matl(dec.vech_log_S_cond)
        S_cond = jsp.linalg.expm(log_S_cond)
        S_cross_aug = jnp.concatenate((jnp.zeros((1, nx, nx)), dec.S_cross))
        S = self.tria2(S_cond, S_cross_aug)
        return GVIProblemVariables(
            xbar=dec.xbar,
            log_S_cond=log_S_cond,
            S=S,
            S_cond=S_cond,
            S_cross=dec.S_cross,
        )
    
    def entropy(self, v: GVIProblemVariables):
        return jnp.trace(v.log_S_cond, axis1=1, axis2=2).sum()


class SmootherKernel(SteadyState):
    def __init__(self, model, nwin: int, elbo_multiplier=1):
        super().__init__(model, elbo_multiplier)

        assert nwin % 2 == 1, "Window length must be odd"
        self.nwin = nwin
        """Length of the convolution window."""

    class Decision(typing.NamedTuple):
        """Problem decision variables."""
        q: npt.NDArray
        K: npt.NDArray
        vech_log_S_cond: npt.NDArray
        S_cross: npt.NDArray

    class ProblemVariables(typing.NamedTuple):
        """Problem variables of BatchedSteadyState problem formulation."""
        xbar: npt.NDArray
        log_S_cond: npt.NDArray
        S: npt.NDArray
        S_cond: npt.NDArray
        S_cross: npt.NDArray
        u: npt.NDArray
        y: npt.NDArray

    def packer(self, N=None):
        return hedeut.Packer(
            q=(self.model.nq,),
            K=(self.model.nx, self.model.ny + self.model.nu, self.nwin),
            vech_log_S_cond=(self.ntrilx,),
            S_cross=(self.model.nx, self.model.nx),
        )

    def smooth(self, K, data: Data):
        sig = jnp.c_[data.y, data.u].T
        return common.conv(sig, K).sum(1).T

    def problem_variables(self, dec: Decision, data: Data): 
        xbar = self.smooth(dec.K, data)
        log_S_cond = common.matl(dec.vech_log_S_cond)
        S_cond = jsp.linalg.expm(log_S_cond)
        S = self.tria2(S_cond, dec.S_cross)
        skip = self.nwin // 2
        return self.ProblemVariables(
            xbar=xbar,
            log_S_cond=log_S_cond,
            S=S,
            S_cond=S_cond,
            S_cross=dec.S_cross,
            u=data.u[skip:-skip],
            y=data.y[skip:-skip],
        )


class PEM:
    """Prediction Error Method estimator."""

    class Decision(typing.NamedTuple):
        """Problem decision variables."""
        q: npt.NDArray
        K: npt.NDArray
        vech_log_sR: npt.NDArray
        x0: npt.NDArray

    def __init__(self, model):
        self.model = model
        """The underlying dynamical system model."""

        self.ntrily = model.ny * (model.ny + 1) // 2
        """Number of elements in lower triangle of ny by ny matrix."""

    def predfun(self, x, y, u, dec: Decision):
        ypred = self.model.h(x, u, dec.q)
        e = y - ypred
        xnext = self.model.f(x, u, dec.q) + dec.K @ e
        return xnext, ypred

    def cost(self, dec: Decision, data: Data):
        scanfun = lambda x, datum: self.predfun(x, *datum, dec)
        x0 = dec.x0 if len(dec.x0) > 0 else jnp.zeros(self.model.nx)
        xnext, ypred = jax.lax.scan(scanfun, x0, data)

        log_sR = common.matl(dec.vech_log_sR)
        return -stats.mvn_logpdf_logchol(data.y, ypred, log_sR).sum(0)
    
    def cost_grad(self, dec, data: Data):
        return jax.grad(self.cost)(dec, data)

    def cost_hvp(self, dec, dec_d, data: Data):
        primals = dec, data
        duals = dec_d, data.zeros_like()
        return jax.jvp(self.cost_grad, primals, duals)[1]
