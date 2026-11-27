"""Classical baseline: a multilayer perceptron Q-network (JAX)."""
from __future__ import annotations

import jax
import jax.numpy as jnp


class MLPQNet:
    def __init__(self, n_obs, n_actions, hidden=(64, 64)):
        self.n_obs, self.n_actions, self.hidden = n_obs, n_actions, tuple(hidden)
        self.config = {"kind": "mlp", "n_obs": n_obs, "n_actions": n_actions, "hidden": list(hidden)}

    def init(self, key):
        sizes = (self.n_obs, *self.hidden, self.n_actions)
        params = []
        for k, (d_in, d_out) in zip(jax.random.split(key, len(sizes) - 1), zip(sizes[:-1], sizes[1:])):
            limit = jnp.sqrt(6.0 / d_in) / 2                    # He-uniform-like
            params.append({"w": jax.random.uniform(k, (d_in, d_out), minval=-limit, maxval=limit),
                           "b": jnp.zeros(d_out)})
        return params

    def apply(self, params, x):
        for layer in params[:-1]:
            x = jax.nn.relu(x @ layer["w"] + layer["b"])
        return x @ params[-1]["w"] + params[-1]["b"]

    def labels(self, params):
        return jax.tree_util.tree_map(lambda _: "base", params)


def mlp_parameters(n_obs, n_actions, hidden):
    total, d = 0, n_obs
    for h in hidden:
        total += d * h + h
        d = h
    return total + d * n_actions + n_actions


def matched_mlp(n_obs, n_actions, target_params, depth=1):
    """MLP with `depth` equal hidden layers whose parameter count is closest to target_params.
    Used to compare a quantum model with a classical one of the same size."""
    best = min(range(1, 512), key=lambda h: abs(mlp_parameters(n_obs, n_actions, [h] * depth) - target_params))
    return MLPQNet(n_obs, n_actions, hidden=[best] * depth)
