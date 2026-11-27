"""Ready-made CV quantum neural networks: encoder -> CV QNN layers -> read-out.

    from cvjax.models import CVQNN
    net = CVQNN(n_inputs=6, n_modes=3, n_layers=4, cutoff=6)
    params = net.init(jax.random.PRNGKey(0))
    y = net.apply(params, x)            # x: (6,) -> y: (3,) = <x_i> of each mode

Encoding: each mode receives two inputs as a displacement of complex amplitude
    alpha_i = s_{2i} x_{2i} + i s_{2i+1} x_{2i+1}          (trainable scales s)
(one input per mode, alpha_i = s_i x_i, when n_inputs == n_modes, as D(x)|0> in de Lima et al.).
By default the encoding is applied once to the vacuum, as D(x)|0> in de Lima et al. and in
Killoran et al.; the non-linearity comes from the Kerr gates. `reupload=True` repeats the
encoding before every layer (data re-uploading, Perez-Salinas et al. 2020), but displacements
add up, the photon number grows and the truncation leaks: check `leakage` before using it.
Read-out: homodyne means <x_i> = <a_i + a_i^dag> (hbar = 2), one number per mode.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from . import gates, layers, measure
from .fock import apply_one, norm, vacuum


class CVQNN:
    def __init__(self, n_inputs, n_modes, n_layers=2, cutoff=6, reupload=False,
                 input_scale=0.5, active_sd=0.0001, passive_sd=0.1):
        if n_inputs not in (n_modes, 2 * n_modes):
            raise ValueError("n_inputs must be n_modes (real displacements) or 2 * n_modes (complex)")
        self.n_inputs, self.n_modes, self.n_layers, self.cutoff = n_inputs, n_modes, n_layers, cutoff
        self.reupload, self.input_scale = reupload, input_scale
        self.active_sd, self.passive_sd = active_sd, passive_sd
        self.config = {"n_inputs": n_inputs, "n_modes": n_modes, "n_layers": n_layers,
                       "cutoff": cutoff, "reupload": reupload, "input_scale": input_scale}

    def init(self, key):
        p = layers.init_params(key, self.n_layers, self.n_modes, self.active_sd, self.passive_sd)
        n_enc = self.n_layers if self.reupload else 1
        p["s"] = jnp.full((n_enc, self.n_inputs), self.input_scale)
        return p

    def _encode(self, psi, x, s):
        z = s * x
        alpha = z if self.n_inputs == self.n_modes else z[0::2] + 1j * z[1::2]
        for i in range(self.n_modes):
            psi = apply_one(psi, gates.displacement_alpha(alpha[i], self.cutoff), i)
        return psi

    def state(self, params, x):
        psi = vacuum(self.n_modes, self.cutoff)
        for layer in range(self.n_layers):
            if layer == 0 or self.reupload:
                psi = self._encode(psi, x, params["s"][layer if self.reupload else 0])
            psi = layers.cv_layer(psi, layers.layer_slice(params, layer), self.cutoff)
        return psi

    def apply(self, params, x):
        psi = self.state(params, x)
        return jnp.stack([measure.quad_x(psi, i) for i in range(self.n_modes)])

    def leakage(self, params, x):
        """1 - <psi|psi>: probability lost above the cutoff (should stay small)."""
        return 1.0 - norm(self.state(params, x))

    def batched(self):
        return jax.vmap(self.apply, in_axes=(None, 0))
