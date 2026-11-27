"""Continuous-variable (photonic) agent: Gaussian CV quantum neural network Q-network.

The circuit is written in PennyLane (`reference_qnode`, device `default.gaussian`). That device
runs one state at a time and differentiates by parameter-shift, which is too slow inside DQN.
So `apply` runs the *same* Gaussian circuit with a small JAX simulator (mean vector and
covariance matrix, hbar = 2 like PennyLane), batched and differentiated by backprop.
tests/test_agentes.py checks that both give the same numbers.

    one optical mode per observation feature (Pong: 6, Snake: 8), starting in vacuum
    repeat L times:  Squeezing(r_i, phi_i) and Displacement(d_i, psi_i) on each mode  (trainable)
                     Rotation(lambda_i * x_i) on mode i                                (data)
                     chain of Beamsplitter(theta_j, varphi_j) between modes j, j+1      (trainable)
    Q(a) = w_a * (<n_{2a}> - <n_{2a+1}>) + b_a       (photon-number difference of a pair of modes)

Data enter as phase rotations (re-uploaded every layer), so the output is a nonlinear,
Fourier-like function of the observation even though every gate is Gaussian. Non-Gaussian gates
(Kerr, cubic phase: the CV-QNN of Killoran et al. 2019) would need a Fock-space simulator: listed
as an extension in docs/PLANO.md.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import pennylane as qml

HBAR = 2.0


# ---------------------------------------------------------- Gaussian simulator
# Quadrature order: (x_0, ..., x_{N-1}, p_0, ..., p_{N-1}).
def _single_mode(n, i, s2):
    """Embed a 2x2 symplectic acting on mode i into a 2N x 2N identity."""
    idx = jnp.array([i, n + i])
    return jnp.eye(2 * n).at[idx[:, None], idx[None, :]].set(s2)


def rotation(n, i, phi):
    c, s = jnp.cos(phi), jnp.sin(phi)
    return _single_mode(n, i, jnp.array([[c, -s], [s, c]]))


def squeezing(n, i, r, phi):
    ch, sh, c, s = jnp.cosh(r), jnp.sinh(r), jnp.cos(phi), jnp.sin(phi)
    return _single_mode(n, i, jnp.array([[ch - sh * c, -sh * s], [-sh * s, ch + sh * c]]))


def beamsplitter(n, i, j, theta, phi):
    ct, st, cp, sp = jnp.cos(theta), jnp.sin(theta), jnp.cos(phi), jnp.sin(phi)
    b = jnp.array([[ct, -cp * st, 0.0, -sp * st],
                   [cp * st, ct, -sp * st, 0.0],
                   [0.0, sp * st, ct, -cp * st],
                   [sp * st, 0.0, cp * st, ct]])        # order (x_i, x_j, p_i, p_j)
    idx = jnp.array([i, j, n + i, n + j])
    return jnp.eye(2 * n).at[idx[:, None], idx[None, :]].set(b)


class GaussianState:
    def __init__(self, n):
        self.n = n
        self.mu = jnp.zeros(2 * n)
        self.cov = jnp.eye(2 * n) * HBAR / 2           # vacuum

    def apply(self, s):
        self.mu = s @ self.mu
        self.cov = s @ self.cov @ s.T

    def displace(self, i, a, phi):
        shift = jnp.sqrt(2 * HBAR) * a
        self.mu = self.mu.at[i].add(shift * jnp.cos(phi)).at[self.n + i].add(shift * jnp.sin(phi))

    def mean_photon(self, i):
        n = self.n
        var = self.cov[i, i] + self.cov[n + i, n + i]
        m2 = self.mu[i] ** 2 + self.mu[n + i] ** 2
        return (var + m2) / (2 * HBAR) - 0.5


# ------------------------------------------------------------------- model
class CVQNet:
    def __init__(self, n_obs, n_actions, layers=4):
        if 2 * n_actions > n_obs:
            raise ValueError("needs at least two modes per action")
        self.n_modes, self.n_actions, self.layers = n_obs, n_actions, layers
        self.config = {"kind": "cv", "n_modes": n_obs, "n_actions": n_actions, "layers": layers,
                       "gates": "Gaussian (squeezing, displacement, rotation, beamsplitter)",
                       "readout": "photon-number difference", "hbar": HBAR}

    def init(self, key):
        n, L = self.n_modes, self.layers
        k1, k2, k3, k4 = jax.random.split(key, 4)
        return {
            "squeeze": jnp.stack([jax.random.uniform(k1, (L, n), maxval=0.1),
                                  jax.random.uniform(k2, (L, n), maxval=2 * jnp.pi)], -1),
            "disp": jnp.stack([jnp.full((L, n), 0.5),
                               jax.random.uniform(k3, (L, n), maxval=2 * jnp.pi)], -1),
            "bs": jax.random.uniform(k4, (L, max(n - 1, 0), 2), maxval=2 * jnp.pi),
            "lam": jnp.full((L, n), jnp.pi / 2),
            "w": jnp.ones(self.n_actions),
            "b": jnp.zeros(self.n_actions),
        }

    def photon_numbers(self, params, x):
        n = self.n_modes
        st = GaussianState(n)
        for layer in range(self.layers):
            for i in range(n):
                r, phi = params["squeeze"][layer, i]
                st.apply(squeezing(n, i, r, phi))
                a, psi = params["disp"][layer, i]
                st.displace(i, a, psi)
            for i in range(n):
                st.apply(rotation(n, i, params["lam"][layer, i] * x[i]))
            for j in range(n - 1):
                theta, varphi = params["bs"][layer, j]
                st.apply(beamsplitter(n, j, j + 1, theta, varphi))
        return jnp.stack([st.mean_photon(i) for i in range(n)])

    def expvals(self, params, x):
        nbar = self.photon_numbers(params, x)
        return jnp.stack([nbar[2 * a] - nbar[2 * a + 1] for a in range(self.n_actions)])

    def apply(self, params, x):
        return params["w"] * self.expvals(params, x) + params["b"]

    def labels(self, params):
        return {"squeeze": "base", "disp": "base", "bs": "base", "lam": "base", "w": "out", "b": "out"}

    # ------------------------------------------- PennyLane definition (reference)
    def reference_qnode(self):
        """The same circuit on PennyLane's default.gaussian device (slow, exact reference).
        Returns <n> of one mode per call: the device allows a single measurement."""
        n, L = self.n_modes, self.layers
        dev = qml.device("default.gaussian", wires=n, hbar=HBAR)

        @qml.qnode(dev)
        def circuit(x, squeeze, disp, bs, lam, mode=0):
            for layer in range(L):
                for i in range(n):
                    qml.Squeezing(squeeze[layer, i, 0], squeeze[layer, i, 1], wires=i)
                    qml.Displacement(disp[layer, i, 0], disp[layer, i, 1], wires=i)
                for i in range(n):
                    qml.Rotation(lam[layer, i] * x[i], wires=i)
                for j in range(n - 1):
                    qml.Beamsplitter(bs[layer, j, 0], bs[layer, j, 1], wires=[j, j + 1])
            return qml.expval(qml.NumberOperator(mode))   # device allows one measurement

        return circuit

    def reference_photon_numbers(self, params, x):
        import numpy as np
        circuit = self.reference_qnode()
        p = [np.asarray(params[k]) for k in ("squeeze", "disp", "bs", "lam")]
        return np.array([circuit(np.asarray(x), *p, mode=i) for i in range(self.n_modes)])

    def draw(self, params, x=None, **kw):
        import numpy as np
        x = np.zeros(self.n_modes) if x is None else x
        p = {k: np.asarray(v) for k, v in params.items()}
        return qml.draw_mpl(self.reference_qnode(), **kw)(x, p["squeeze"], p["disp"], p["bs"], p["lam"])
