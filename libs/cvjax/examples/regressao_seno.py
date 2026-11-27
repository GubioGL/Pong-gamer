"""Single-mode CV QNN regression, in the setting of de Lima, Farias, Ricardo and Boas,
"Assessing the Advantages and Limitations of Quantum Neural Networks in Regression Tasks"
(arXiv:2509.00854; Quantum Machine Intelligence 8, 20 (2026)):

    input  x           -> D(x)|0>                 (displacement of the vacuum)
    layers             -> [R S R D K] x L         (one-mode CVNeuralNetLayers)
    output             -> <x_hat> = <a + a^dag>   (homodyne mean, hbar = 2)
    loss               -> mean squared error

The original code is PyTorch with matrix_exp of truncated generators and a trapped-ion model of
the Kerr gate; this example uses cvjax's exact matrix elements and the ideal Kerr exp(i k n^2).
The target below is only a demonstration, not the paper's benchmark.

    python examples/regressao_seno.py
"""
import time

import jax
import jax.numpy as jnp
import optax

from cvjax.models import CVQNN

net = CVQNN(n_inputs=1, n_modes=1, n_layers=3, cutoff=12, input_scale=1.0,
            active_sd=0.1, passive_sd=0.5)
x = jnp.linspace(-1.0, 1.0, 40)[:, None]
y = 0.5 * jnp.sin(jnp.pi * x[:, 0])

params = net.init(jax.random.PRNGKey(1))
opt = optax.adam(0.02)
state = opt.init(params)
f = net.batched()


@jax.jit
def step(params, state):
    loss, g = jax.value_and_grad(lambda p: jnp.mean((f(p, x)[:, 0] - y) ** 2))(params)
    upd, state = opt.update(g, state, params)
    return optax.apply_updates(params, upd), state, loss


t = time.perf_counter()
for it in range(1, 801):
    params, state, loss = step(params, state)
    if it in (1, 100, 200, 400, 800):
        print(f"step {it:4d}  MSE {float(loss):.2e}")
leak = max(float(net.leakage(params, xi)) for xi in x)
print(f"max leakage above the cutoff: {leak:.1e}   ({time.perf_counter() - t:.1f}s)")
