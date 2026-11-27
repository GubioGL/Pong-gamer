"""Discrete-variable (qubit) agent: variational quantum circuit Q-network in PennyLane.

Architecture after Skolik et al., Quantum 6, 720 (2022) [skolik2022gym], with data re-uploading
[perezsalinas2020reuploading]:

    one qubit per observation feature (Pong: 6, Snake: 8)
    repeat L times:  variational layer  (RY(theta), RZ(theta) on each qubit + ring of CZ)
                     encoding layer     (RX(lambda * x_i) on qubit i, lambda trainable)
    final variational layer
    Q(a) = w_a * <Z_{2a} Z_{2a+1}> + b_a          (one pair of qubits per action)

The encoding angle starts at lambda = pi/2, so x in [-1, 1] maps to [-pi/2, pi/2]
(+-pi would give the same rotation up to a global phase).
Output scale w and bias b are "out" parameters: they get their own learning rate in the DQN.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import pennylane as qml


class QubitQNet:
    def __init__(self, n_obs, n_actions, layers=4, entangle=True):
        if 2 * n_actions > n_obs:
            raise ValueError("needs at least two qubits per action")
        self.n_qubits, self.n_actions, self.layers, self.entangle = n_obs, n_actions, layers, entangle
        self.config = {"kind": "vqc", "n_qubits": n_obs, "n_actions": n_actions, "layers": layers,
                       "entangle": entangle, "device": "default.qubit", "diff_method": "backprop"}
        self.dev = qml.device("default.qubit", wires=n_obs)
        self.qnode = qml.QNode(self._circuit, self.dev, interface="jax", diff_method="backprop")

    # ---------------------------------------------------------------- circuit
    def _variational(self, theta):
        n = self.n_qubits
        for i in range(n):
            qml.RY(theta[i, 0], wires=i)
            qml.RZ(theta[i, 1], wires=i)
        if self.entangle and n > 1:
            for i in range(n):
                qml.CZ(wires=[i, (i + 1) % n])

    def _circuit(self, x, theta, lam):
        for layer in range(self.layers):
            self._variational(theta[layer])
            for i in range(self.n_qubits):
                qml.RX(lam[layer, i] * x[i], wires=i)
        self._variational(theta[self.layers])
        return [qml.expval(qml.Z(2 * a) @ qml.Z(2 * a + 1)) for a in range(self.n_actions)]

    # -------------------------------------------------------------- model API
    def init(self, key):
        n, L = self.n_qubits, self.layers
        return {
            "theta": jax.random.uniform(key, (L + 1, n, 2), minval=0.0, maxval=2 * jnp.pi),
            "lam": jnp.full((L, n), jnp.pi / 2),
            "w": jnp.ones(self.n_actions),
            "b": jnp.zeros(self.n_actions),
        }

    def expvals(self, params, x):
        return jnp.stack(self.qnode(x, params["theta"], params["lam"]))

    def apply(self, params, x):
        return params["w"] * self.expvals(params, x) + params["b"]

    def labels(self, params):
        return {"theta": "base", "lam": "base", "w": "out", "b": "out"}

    def draw(self, params, x=None, **kw):
        """Matplotlib figure of the circuit (for the paper)."""
        x = jnp.zeros(self.n_qubits) if x is None else x
        return qml.draw_mpl(self.qnode, **kw)(x, params["theta"], params["lam"])
