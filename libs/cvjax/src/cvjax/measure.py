"""Expectation values of a pure Fock-basis state (hbar = 2, as in PennyLane)."""
from __future__ import annotations

import jax.numpy as jnp

HBAR = 2.0


def _sum_except(x, mode):
    axes = tuple(i for i in range(x.ndim) if i != mode)
    return jnp.sum(x, axis=axes) if axes else x


def fock_probs(psi, mode):
    """Marginal photon-number distribution P(n) of one mode."""
    return _sum_except(jnp.abs(psi) ** 2, mode)


def mean_photon(psi, mode):
    """<n> of one mode."""
    p = fock_probs(psi, mode)
    return jnp.sum(jnp.arange(p.shape[0]) * p)


def mean_a(psi, mode):
    """<a> = sum_n sqrt(n+1) conj(psi[.., n, ..]) psi[.., n+1, ..]."""
    d = psi.shape[mode]
    lo = jnp.take(psi, jnp.arange(d - 1), axis=mode)
    hi = jnp.take(psi, jnp.arange(1, d), axis=mode)
    shape = [1] * psi.ndim
    shape[mode] = d - 1
    w = jnp.sqrt(jnp.arange(1, d)).reshape(shape)
    return jnp.sum(jnp.conj(lo) * hi * w)


def quad_x(psi, mode, hbar=HBAR):
    """<x> = sqrt(hbar/2) <a + a^dag> (= <a + a^dag> for hbar = 2, the read-out of de Lima et al.)."""
    return jnp.sqrt(2 * hbar) * jnp.real(mean_a(psi, mode))


def quad_p(psi, mode, hbar=HBAR):
    """<p> = -i sqrt(hbar/2) <a - a^dag>."""
    return jnp.sqrt(2 * hbar) * jnp.imag(mean_a(psi, mode))
