"""Continuous-variable gates as matrices in the truncated Fock basis (JAX, differentiable).

Conventions are PennyLane's (and Strawberry Fields'), hbar = 2:

    Rotation(phi)          R = exp(i phi n)
    Kerr(kappa)            K = exp(i kappa n^2)
    Displacement(a, phi)   D = exp(alpha a^dag - alpha* a),          alpha = a e^{i phi}
    Squeezing(r, phi)      S = exp((z* a^2 - z a^dag^2) / 2),        z = r e^{i phi}
    Beamsplitter(theta, phi) B = exp(theta (e^{i phi} a b^dag - e^{-i phi} a^dag b))

Matrix elements are the EXACT elements of the infinite-dimensional operators between Fock
states below the cutoff (like Strawberry Fields, which takes D, S and B from The Walrus).
That is different from `expm` of the truncated generator (CV-QuForge, and the PyTorch code of
de Lima et al. 2025), which is only exact far from the cutoff. How we get exact elements:

* D and S: disentangled (normal-ordered) forms, e.g.
      D(alpha) = e^{-|alpha|^2/2} e^{alpha a^dag} e^{-alpha* a}
  Every intermediate Fock state in the product lies below min(m, n), so truncating it is exact.
  (QuIara applies the same identities to the state; here we build the matrices.)
* B: the beamsplitter keeps the total photon number s = n_a + n_b. In each block s it is a
  spin-s/2 rotation, computed in full (diagonalized once in float64) and then cut to the cutoff.
* R and K are diagonal.
"""
from __future__ import annotations

from functools import lru_cache

import jax.numpy as jnp
import numpy as np

from .fock import DTYPE


@lru_cache(maxsize=None)
def _logfact(n):
    return np.concatenate([[0.0], np.cumsum(np.log(np.arange(1, n + 1)))])


def _powers(c, n):
    """[1, c, c^2, ..., c^(n-1)] by cumulative product (safe for c = 0 and negative c)."""
    c = jnp.asarray(c, DTYPE)
    return jnp.concatenate([jnp.ones(1, DTYPE), jnp.cumprod(jnp.full(n - 1, c, DTYPE))])


def _exp_ladder(c, cutoff, power, raising):
    """Exact matrix of exp(c a^dag^power) (raising) or exp(c a^power) (lowering), truncated.

    <m| exp(c a^dag^p) |k> = c^j / j! * sqrt(m! / k!)  for m = k + p j, else 0.
    In the (real) Fock basis a = (a^dag)^T, so exp(c a^p) is the transpose of exp(c a^dag^p).
    """
    lf = _logfact(cutoff)
    m, k = np.meshgrid(np.arange(cutoff), np.arange(cutoff), indexing="ij")
    diff = m - k
    valid = (diff >= 0) & (diff % power == 0)
    j = np.where(valid, diff // power, 0)
    mag = np.where(valid, np.exp(0.5 * (lf[m] - lf[k]) - lf[j]), 0.0)
    mat = jnp.where(valid, _powers(c, cutoff)[j] * mag, 0.0).astype(DTYPE)
    return mat if raising else mat.T


def rotation(phi, cutoff):
    """Diagonal of R(phi) = exp(i phi n)."""
    n = jnp.arange(cutoff)
    return jnp.exp(1j * phi * n).astype(DTYPE)


def kerr(kappa, cutoff):
    """Diagonal of K(kappa) = exp(i kappa n^2)."""
    n = jnp.arange(cutoff)
    return jnp.exp(1j * kappa * n ** 2).astype(DTYPE)


def displacement_alpha(alpha, cutoff):
    """D(alpha) for a complex alpha (handy when data are encoded as alpha = x1 + i x2)."""
    alpha = jnp.asarray(alpha, DTYPE)
    up = _exp_ladder(alpha, cutoff, 1, raising=True)
    down = _exp_ladder(-jnp.conj(alpha), cutoff, 1, raising=False)
    return jnp.exp(-0.5 * jnp.abs(alpha) ** 2) * (up @ down)


def displacement(a, phi, cutoff):
    """PennyLane Displacement(a, phi): alpha = a e^{i phi}."""
    return displacement_alpha(a * jnp.exp(1j * phi), cutoff)


def squeezing(r, phi, cutoff):
    """PennyLane Squeezing(r, phi), via
    S = exp(-tau a^dag^2 / 2) (cosh r)^{-(n + 1/2)} exp(tau* a^2 / 2),  tau = e^{i phi} tanh r.
    """
    tau = jnp.exp(1j * phi) * jnp.tanh(r)
    up = _exp_ladder(-0.5 * tau, cutoff, 2, raising=True)
    down = _exp_ladder(0.5 * jnp.conj(tau), cutoff, 2, raising=False)
    n = jnp.arange(cutoff)
    mid = jnp.cosh(r) ** (-(n + 0.5))
    return (up * mid[None, :].astype(DTYPE)) @ down


@lru_cache(maxsize=None)
def _bs_blocks(cutoff):
    """Static bookkeeping for the beamsplitter, per total photon number s (cutoff only).

    In block s (basis |k, s-k>, k = 0..s) the generator is
        theta (e^{i phi} A - e^{-i phi} A^T),   A = a b^dag,  A[k-1, k] = sqrt(k (s-k+1)),
    and equals P [theta (A - A^T)] P^dag with P = diag(e^{-i phi k}). J = A - A^T is real
    antisymmetric (a spin-s/2 rotation generator) with eigenvalues i*mu, mu integer, so
        B_s = P Q diag(e^{i theta mu}) Q^dag P^dag
    with Q, mu computed once here in float64: exact and cheaper than expm.
    """
    blocks = []
    for s in range(2 * cutoff - 1):
        k = np.arange(s + 1)
        a_bdag = np.zeros((s + 1, s + 1))
        a_bdag[k[1:] - 1, k[1:]] = np.sqrt(k[1:] * (s - k[1:] + 1))
        mu, q = np.linalg.eigh(1j * (a_bdag - a_bdag.T))   # i J is Hermitian: J = Q diag(-i mu) Q^dag
        keep = k[(k < cutoff) & (s - k < cutoff)]          # block states inside the cutoff
        blocks.append((s, k, -mu, q, keep))
    return blocks


def beamsplitter(theta, phi, cutoff):
    """PennyLane Beamsplitter(theta, phi) as u4[m, n, k, l] = <m, n|B|k, l> (a = first mode).
    Same index order as The Walrus' `fock_gradients.beamsplitter`."""
    u4 = jnp.zeros((cutoff,) * 4, DTYPE)
    for s, k, mu, q, keep in _bs_blocks(cutoff):
        pq = jnp.exp(-1j * phi * k)[:, None].astype(DTYPE) * jnp.asarray(q, DTYPE)
        ub = (pq * jnp.exp(1j * theta * mu)[None, :].astype(DTYPE)) @ jnp.conj(pq).T
        mi, ki = np.meshgrid(keep, keep, indexing="ij")
        u4 = u4.at[mi, s - mi, ki, s - ki].set(ub[mi, ki])
    return u4
