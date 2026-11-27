"""Continuous-variable agent with Kerr non-linearity: CV quantum neural network (Killoran et al.
2019; PennyLane's CVNeuralNetLayers) simulated in the Fock basis by the cvjax library
(libs/cvjax, inspired by PennyLane/Strawberry Fields and by Tiago de S. Farias' and Alexandre
C. Ricardo's CV-QuForge and QuIara).

    observation x (n = 6 Pong, 8 Snake), two numbers per optical mode:
        mode i starts in vacuum and is displaced by alpha_i = s_{2i} x_{2i} + i s_{2i+1} x_{2i+1}
        -> M = n/2 modes (Pong 3, Snake 4) = one mode per action
    L layers of [Interferometer, Squeezing, Interferometer, Displacement, Kerr]
    Q(a) = w_a <x_hat_a> + b_a           (homodyne mean of mode a, hbar = 2)

Fock cutoff D per mode (default 6): the state has D^M amplitudes (Pong 216, Snake 1296).
The encoding is applied once, as D(x)|0> in de Lima et al. (arXiv:2509.00854): re-uploading
displacements would add photons and leak above the cutoff.
"""
from __future__ import annotations

import sys
from pathlib import Path

import jax.numpy as jnp

try:
    from cvjax.models import CVQNN
except ImportError:                     # not pip-installed: use the copy in libs/cvjax
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "libs" / "cvjax" / "src"))
    from cvjax.models import CVQNN


class CVFockQNet:
    def __init__(self, n_obs, n_actions, layers=4, cutoff=6, input_scale=0.5):
        if n_obs != 2 * n_actions:
            raise ValueError("this agent uses two observation numbers per mode and one mode per action")
        self.net = CVQNN(n_inputs=n_obs, n_modes=n_actions, n_layers=layers, cutoff=cutoff,
                         reupload=False, input_scale=input_scale)
        self.n_actions = n_actions
        self.config = {"kind": "cv_fock", "n_modes": n_actions, "n_actions": n_actions, "layers": layers,
                       "cutoff": cutoff, "input_scale": input_scale, "simulator": "cvjax (Fock basis)",
                       "gates": "CVNeuralNetLayers: interferometer, squeezing, interferometer, displacement, Kerr",
                       "readout": "homodyne <x> per mode"}

    def init(self, key):
        p = self.net.init(key)
        p["w"] = jnp.ones(self.n_actions)
        p["b"] = jnp.zeros(self.n_actions)
        return p

    def expvals(self, params, x):
        return self.net.apply(params, x)

    def apply(self, params, x):
        return params["w"] * self.expvals(params, x) + params["b"]

    def labels(self, params):
        return {k: ("out" if k in ("w", "b") else "base") for k in params}

    def leakage(self, params, x):
        return self.net.leakage(params, x)
