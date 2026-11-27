"""Pure states of M optical modes in a truncated Fock basis, as JAX arrays.

A state is a complex array `psi` with one axis per mode, shape (D, D, ..., D) with D = cutoff:
psi[n_0, n_1, ..., n_{M-1}] = <n_0 n_1 ... n_{M-1} | psi>.

Gates are applied along the axes of the mode(s) they act on, without building the full
D^M x D^M matrix (the same idea as QuIara's per-mode application). Batch over many inputs
with `jax.vmap`.
"""
from __future__ import annotations

import jax.numpy as jnp

DTYPE = jnp.complex64


def vacuum(n_modes: int, cutoff: int, dtype=DTYPE):
    """|0, 0, ..., 0>."""
    psi = jnp.zeros((cutoff,) * n_modes, dtype=dtype)
    return psi.at[(0,) * n_modes].set(1.0)


def fock_state(ns, cutoff: int, dtype=DTYPE):
    """|n_0, n_1, ...> (a basis state)."""
    psi = jnp.zeros((cutoff,) * len(ns), dtype=dtype)
    return psi.at[tuple(ns)].set(1.0)


def apply_one(psi, u, mode: int):
    """Apply a single-mode operator u (D x D, u[m, n] = <m|U|n>) to `mode`."""
    out = jnp.tensordot(u, psi, axes=([1], [mode]))       # new axis 0 = output index
    return jnp.moveaxis(out, 0, mode)


def apply_diag(psi, d, mode: int):
    """Apply a diagonal single-mode operator given by its diagonal d (length D)."""
    shape = [1] * psi.ndim
    shape[mode] = psi.shape[mode]
    return psi * d.reshape(shape)


def apply_two(psi, u4, mode_i: int, mode_j: int):
    """Apply a two-mode operator u4[m, n, k, l] = <m, n|U|k, l> to modes (mode_i, mode_j)."""
    out = jnp.tensordot(u4, psi, axes=([2, 3], [mode_i, mode_j]))   # new axes 0, 1
    return jnp.moveaxis(out, (0, 1), (mode_i, mode_j))


def norm(psi):
    """<psi|psi>. Smaller than 1 when the circuit pushed probability above the cutoff
    (truncation leakage): keep it close to 1 by choosing the cutoff and the gate sizes."""
    return jnp.sum(jnp.abs(psi) ** 2)
