"""Single source of truth for the experiments: games, agents and DQN settings.

Changing anything here invalidates results already saved in data/experimentos/.
"""
from __future__ import annotations

from agentes.cvfock import CVFockQNet
from agentes.cvqnn import CVQNet
from agentes.mlp import MLPQNet, matched_mlp
from agentes.vqc import QubitQNet
from games.pong import PongEnv
from games.snake import SnakeEnv
from rl.dqn import DQNConfig

LAYERS = 4
CUTOFF = 6        # Fock levels per mode of the CV agent (leakage <= ~1e-3 at initialization)

# Pong: lr 3e-4 chosen with the MLP on 3 seeds (data/ajuste, Supplementary S5); Snake kept at 1e-3.
# The quantum models' learning rates were NOT tuned (see PLANO.md).
GAMES = {
    # Pong episodes are short matches (3 points) so that agents see many episodes.
    "pong": {"env": lambda seed: PongEnv(seed=seed, points_to_win=3, max_steps=5_000),
             "n_obs": PongEnv.OBS_SIZE, "n_actions": PongEnv.N_ACTIONS,
             "dqn": DQNConfig(total_steps=100_000, eps_decay_steps=40_000, lr=3e-4, lr_out=1e-2,
                     eval_every=5_000, eval_episodes=5)},
    "snake": {"env": lambda seed: SnakeEnv(seed=seed),
              "n_obs": SnakeEnv.OBS_SIZE, "n_actions": SnakeEnv.N_ACTIONS,
              "dqn": DQNConfig(total_steps=100_000, eps_decay_steps=40_000, lr=1e-3, lr_out=1e-2,
                     eval_every=5_000, eval_episodes=5)},
}


def make_agent(name, game):
    g = GAMES[game]
    n, a = g["n_obs"], g["n_actions"]
    if name == "vqc":
        return QubitQNet(n, a, layers=LAYERS)
    if name == "cv":                    # CV QNN with Kerr, Fock basis (cvjax)
        return CVFockQNet(n, a, layers=LAYERS, cutoff=CUTOFF)
    if name == "cv_gauss":              # Gaussian-only CV network (ablation: no Kerr)
        return CVQNet(n, a, layers=LAYERS)
    if name == "mlp_igual":            # same number of parameters as the qubit VQC
        import jax
        from rl.dqn import count_parameters
        target = count_parameters(QubitQNet(n, a, layers=LAYERS).init(jax.random.PRNGKey(0)))
        return matched_mlp(n, a, target)
    if name == "mlp":
        return MLPQNet(n, a, hidden=(64, 64))
    raise ValueError(f"unknown agent {name!r}")


AGENTS = ("mlp", "mlp_igual", "vqc", "cv", "cv_gauss")
