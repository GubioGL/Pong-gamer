"""CV quantum neural network layers (Killoran et al. 2019) on the Fock simulator.

`cv_neural_net_layers` reproduces PennyLane's `qml.CVNeuralNetLayers` template operation by
operation (same 11 parameter arrays, same shapes, same order of gates, rectangular
"pennylane" interferometer mesh). In PennyLane this template needs a Fock device, and the only
one, `strawberryfields.fock` from the `pennylane-sf` plugin, requires pennylane < 0.30. Here
the same circuit runs in JAX (jit, vmap, grad).

`run_pennylane_ops` executes any list of PennyLane CV operations (for instance the output of
`qml.CVNeuralNetLayers.compute_decomposition`) on this simulator; the tests use it to check
that the native layers and PennyLane's template are the same circuit.

One layer, for M modes:
    Interferometer(theta_1, phi_1, varphi_1)   M(M-1)/2 beamsplitters + M rotations
    Squeezing(r_i, phi_r_i)                    on every mode
    Interferometer(theta_2, phi_2, varphi_2)
    Displacement(a_i, phi_a_i)                 on every mode
    Kerr(k_i)                                  on every mode  (the non-Gaussian "activation")
For a single mode the interferometers are rotations, so one layer is
    R(varphi_1) S(r, phi_r) R(varphi_2) D(a, phi_a) K(k)
which is the layer of the regression QNN of de Lima, Farias, Ricardo and Boas (2025).
"""
from __future__ import annotations

from itertools import product

import jax
import jax.numpy as jnp

from . import gates
from .fock import apply_diag, apply_one, apply_two

PARAM_NAMES = ("theta_1", "phi_1", "varphi_1", "r", "phi_r",
               "theta_2", "phi_2", "varphi_2", "a", "phi_a", "k")


def shapes(n_layers, n_modes):
    """Same as qml.CVNeuralNetLayers.shape(n_layers, n_wires)."""
    k = n_modes * (n_modes - 1) // 2
    return [(n_layers, k)] * 2 + [(n_layers, n_modes)] * 3 + [(n_layers, k)] * 2 + [(n_layers, n_modes)] * 4


def init_params(key, n_layers, n_modes, active_sd=0.0001, passive_sd=0.1):
    """Initialization of PennyLane's CV QNN demo (`cv_qnn_layer_weights`): small *active*
    parameters (squeezing, displacement, Kerr) keep the state close to vacuum, so few photons and
    a small cutoff suffice; *passive* ones (beamsplitters, rotations) are normal(0, passive_sd)."""
    passive = {"theta_1", "phi_1", "varphi_1", "theta_2", "phi_2", "varphi_2", "phi_r", "phi_a"}
    keys = jax.random.split(key, len(PARAM_NAMES))
    return {name: jax.random.normal(kk, shp) * (passive_sd if name in passive else active_sd)
            for name, shp, kk in zip(PARAM_NAMES, shapes(n_layers, n_modes), keys)}


def interferometer(psi, theta, phi, varphi, cutoff):
    """qml.Interferometer(theta, phi, varphi, mesh="rectangular", beamsplitter="pennylane")."""
    m = psi.ndim
    if m == 1:
        return apply_diag(psi, gates.rotation(varphi[0], cutoff), 0)
    n = 0
    for layer, k in product(range(m), range(m - 1)):      # Clements rectangular array, depth M
        if (layer + k) % 2 != 1:
            psi = apply_two(psi, gates.beamsplitter(theta[n], phi[n], cutoff), k, k + 1)
            n += 1
    for i in range(m):
        psi = apply_diag(psi, gates.rotation(varphi[i], cutoff), i)
    return psi


def cv_layer(psi, p, cutoff):
    """One CV QNN layer; p holds the 11 arrays of ONE layer (1-D slices)."""
    m = psi.ndim
    psi = interferometer(psi, p["theta_1"], p["phi_1"], p["varphi_1"], cutoff)
    for i in range(m):
        psi = apply_one(psi, gates.squeezing(p["r"][i], p["phi_r"][i], cutoff), i)
    psi = interferometer(psi, p["theta_2"], p["phi_2"], p["varphi_2"], cutoff)
    for i in range(m):
        psi = apply_one(psi, gates.displacement(p["a"][i], p["phi_a"][i], cutoff), i)
    for i in range(m):
        psi = apply_diag(psi, gates.kerr(p["k"][i], cutoff), i)
    return psi


def layer_slice(params, layer):
    return {name: params[name][layer] for name in PARAM_NAMES}


def cv_neural_net_layers(psi, params, cutoff):
    """All layers of qml.CVNeuralNetLayers(*[params[n] for n in PARAM_NAMES], wires=range(M))."""
    for layer in range(params["r"].shape[0]):
        psi = cv_layer(psi, layer_slice(params, layer), cutoff)
    return psi


# ----------------------------------------------------- PennyLane interoperability
def run_pennylane_ops(psi, ops, cutoff):
    """Apply a list of PennyLane CV operations (Rotation, Kerr, Displacement, Squeezing,
    Beamsplitter, Interferometer) to psi. Wires must be the integers 0..M-1."""
    for op in ops:
        name, w, par = op.name, [int(x) for x in op.wires], op.parameters
        if name == "Interferometer":
            psi = run_pennylane_ops(psi, op.decomposition(), cutoff)
        elif name == "Rotation":
            psi = apply_diag(psi, gates.rotation(par[0], cutoff), w[0])
        elif name == "Kerr":
            psi = apply_diag(psi, gates.kerr(par[0], cutoff), w[0])
        elif name == "Displacement":
            psi = apply_one(psi, gates.displacement(par[0], par[1], cutoff), w[0])
        elif name == "Squeezing":
            psi = apply_one(psi, gates.squeezing(par[0], par[1], cutoff), w[0])
        elif name == "Beamsplitter":
            psi = apply_two(psi, gates.beamsplitter(par[0], par[1], cutoff), w[0], w[1])
        else:
            raise NotImplementedError(f"cvjax has no gate {name!r}")
    return psi
