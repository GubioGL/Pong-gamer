import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402

from cvjax import gates, layers, measure  # noqa: E402
from cvjax.fock import apply_one, fock_state, norm, vacuum  # noqa: E402
from cvjax.models import CVQNN  # noqa: E402

D = 8


def _a(n):
    return np.diag(np.sqrt(np.arange(1, n)), 1)


# ------------------------------------------------------------ single gates
@pytest.mark.parametrize("r,phi", [(0.3, 0.2), (-0.7, 1.1)])
def test_displacement_and_squeezing_exact_elements(r, phi):
    from scipy.linalg import expm
    big = 60
    a = _a(big)
    al, z = r * np.exp(1j * phi), r * np.exp(1j * phi)
    d_ref = expm(al * a.T - np.conj(al) * a)[:D, :D]
    s_ref = expm(0.5 * (np.conj(z) * a @ a - z * a.T @ a.T))[:D, :D]
    np.testing.assert_allclose(gates.displacement(r, phi, D), d_ref, atol=2e-6)
    np.testing.assert_allclose(gates.squeezing(r, phi, D), s_ref, atol=2e-6)


def test_coherent_state_amplitudes():
    from math import factorial
    alpha = 0.6 - 0.3j
    psi = apply_one(vacuum(1, D), gates.displacement_alpha(alpha, D), 0)
    ref = [np.exp(-abs(alpha) ** 2 / 2) * alpha ** n / np.sqrt(factorial(n)) for n in range(D)]
    np.testing.assert_allclose(psi, ref, atol=1e-6)


def test_beamsplitter_matches_the_walrus():
    tw = pytest.importorskip("thewalrus.fock_gradients")
    for theta, phi in [(0.4, 0.1), (1.2, -0.7), (2.5, 0.3)]:
        np.testing.assert_allclose(gates.beamsplitter(theta, phi, D), tw.beamsplitter(theta, phi, D), atol=2e-6)


def test_beamsplitter_conserves_photons_and_norm():
    psi = fock_state([2, 1], D)
    from cvjax.fock import apply_two
    out = apply_two(psi, gates.beamsplitter(0.7, 0.3, D), 0, 1)
    assert abs(float(norm(out)) - 1.0) < 1e-5
    total = float(measure.mean_photon(out, 0) + measure.mean_photon(out, 1))
    assert abs(total - 3.0) < 1e-5


def test_kerr_and_rotation_are_phases():
    k = np.asarray(gates.kerr(0.3, D))
    np.testing.assert_allclose(k, np.exp(1j * 0.3 * np.arange(D) ** 2), atol=1e-6)
    np.testing.assert_allclose(np.abs(gates.rotation(1.1, D)), 1.0, atol=1e-6)


# --------------------------------------------- against PennyLane (the reference)
def _random_params(seed, n_layers, n_modes, scale=0.3):
    rng = np.random.default_rng(seed)
    return {n: rng.normal(0, scale, shp) for n, shp in zip(layers.PARAM_NAMES, layers.shapes(n_layers, n_modes))}


def test_layers_match_pennylane_template_decomposition():
    qml = pytest.importorskip("pennylane")
    p = _random_params(0, 2, 3)
    ops = qml.CVNeuralNetLayers.compute_decomposition(*[p[n] for n in layers.PARAM_NAMES], wires=[0, 1, 2])
    mine = layers.cv_neural_net_layers(vacuum(3, 5), p, 5)
    via_pl = layers.run_pennylane_ops(vacuum(3, 5), ops, 5)
    np.testing.assert_allclose(mine, via_pl, atol=1e-5)
    assert qml.CVNeuralNetLayers.shape(2, 3) == layers.shapes(2, 3)


def test_gaussian_part_matches_pennylane_default_gaussian():
    """With Kerr = 0 the circuit is Gaussian: compare <n_i> and <x_i> with PennyLane's exact
    Gaussian simulator, using a cutoff large enough for the truncation to be negligible."""
    qml = pytest.importorskip("pennylane")
    p = _random_params(1, 2, 2, scale=0.25)
    p["k"] = np.zeros_like(p["k"])
    cutoff = 25
    psi = layers.cv_neural_net_layers(vacuum(2, cutoff), p, cutoff)
    assert float(norm(psi)) > 0.9999
    dev = qml.device("default.gaussian", wires=2, hbar=2)
    for mode in range(2):
        for obs, mine in [(qml.NumberOperator, measure.mean_photon(psi, mode)),
                          (qml.QuadX, measure.quad_x(psi, mode))]:
            ops = qml.CVNeuralNetLayers.compute_decomposition(*[p[n] for n in layers.PARAM_NAMES], wires=[0, 1])

            @qml.qnode(dev)
            def circ():
                for op in ops:
                    if op.name != "Kerr":          # default.gaussian refuses Kerr, even kappa = 0
                        qml.apply(op)
                return qml.expval(obs(mode))
            assert abs(float(mine) - float(circ())) < 2e-4, (mode, obs)


# ----------------------------------------------------------------- the model
def test_model_jit_grad_vmap_and_low_leakage():
    net = CVQNN(n_inputs=4, n_modes=2, n_layers=2, cutoff=10)
    params = net.init(jax.random.PRNGKey(0))
    x = jnp.linspace(-1, 1, 12).reshape(3, 4)
    y = jax.jit(net.batched())(params, x)
    assert y.shape == (3, 2) and bool(jnp.all(jnp.isfinite(y)))
    g = jax.grad(lambda p: net.batched()(p, x).sum())(params)
    assert all(bool(jnp.all(jnp.isfinite(v))) for v in jax.tree_util.tree_leaves(g))
    assert float(net.leakage(params, x[0])) < 1e-3
    # Kerr really enters the output
    params2 = dict(params, k=params["k"] + 0.5)
    assert float(jnp.abs(net.apply(params2, x[0]) - net.apply(params, x[0])).max()) > 1e-4
