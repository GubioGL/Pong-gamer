import numpy as np
import pytest

jax = pytest.importorskip("jax")
pytest.importorskip("pennylane")
import jax.numpy as jnp  # noqa: E402

from agentes.cvfock import CVFockQNet  # noqa: E402
from agentes.cvqnn import CVQNet  # noqa: E402
from agentes.mlp import MLPQNet, matched_mlp, mlp_parameters  # noqa: E402
from agentes.vqc import QubitQNet  # noqa: E402
from rl.dqn import DQNConfig, count_parameters, train  # noqa: E402
from games.pong import PongEnv  # noqa: E402


@pytest.mark.parametrize("model", [MLPQNet(6, 3), QubitQNet(6, 3, layers=1), CVQNet(6, 3, layers=1),
                                   CVFockQNet(6, 3, layers=1, cutoff=4)])
def test_output_shape(model):
    p = model.init(jax.random.PRNGKey(0))
    q = jax.vmap(model.apply, (None, 0))(p, jnp.zeros((5, 6)))
    assert q.shape == (5, 3) and bool(jnp.all(jnp.isfinite(q)))


def test_qubit_expvals_bounded_and_depend_on_input():
    m = QubitQNet(6, 3, layers=2)
    p = m.init(jax.random.PRNGKey(1))
    e0, e1 = m.expvals(p, jnp.zeros(6)), m.expvals(p, jnp.ones(6) * 0.7)
    assert bool(jnp.all(jnp.abs(e0) <= 1.0 + 1e-5))
    assert float(jnp.abs(e0 - e1).max()) > 1e-3


def test_cv_jax_simulator_matches_pennylane_default_gaussian():
    m = CVQNet(4, 2, layers=2)
    p = m.init(jax.random.PRNGKey(3))
    x = np.random.default_rng(0).uniform(-1, 1, 4)
    mine = np.asarray(m.photon_numbers(p, jnp.asarray(x)))
    ref = m.reference_photon_numbers(p, x)
    np.testing.assert_allclose(mine, ref, atol=1e-5)


def test_parameter_counts_and_matched_mlp():
    assert count_parameters(QubitQNet(6, 3, layers=4).init(jax.random.PRNGKey(0))) == 5 * 6 * 2 + 4 * 6 + 3 + 3
    mlp = matched_mlp(6, 3, 90)
    assert abs(mlp_parameters(6, 3, mlp.hidden) - 90) <= 10


def test_dqn_mlp_smoke(tmp_path):
    cfg = DQNConfig(total_steps=600, warmup=100, eps_decay_steps=400, target_update=50)
    params, hist = train(lambda s: PongEnv(seed=s, points_to_win=1, max_steps=300), MLPQNet(6, 3), cfg,
                         seed=0, log_path=tmp_path / "seed_0.csv")
    assert len(hist) >= 1 and np.isfinite(hist[-1]["loss"])
    assert (tmp_path / "seed_0.csv").exists() and (tmp_path / "seed_0.json").exists()


def test_cv_fock_agent_kerr_matters_and_leakage_small():
    m = CVFockQNet(6, 3, layers=2, cutoff=6)
    p = m.init(jax.random.PRNGKey(0))
    x = jnp.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0])
    assert float(m.leakage(p, x)) < 5e-3
    p2 = dict(p, k=p["k"] + 0.3)
    assert float(jnp.abs(m.apply(p2, x) - m.apply(p, x)).max()) > 1e-4
    assert count_parameters(p) == 2 * 11 * 3 + 6 + 6     # 11 arrays x 3 modes per layer, scales, w, b
