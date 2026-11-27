# cvjax

**Redes neurais quânticas de variáveis contínuas (CV-QNN) na base de Fock, em JAX.**

Biblioteca pequena (≈ 420 linhas) para simular e treinar circuitos fotônicos com portas gaussianas **e Kerr** (não gaussiana), com `jax.jit`, `jax.vmap` e `jax.grad`. Ela foi feita para o projeto [Pong-gamer](../../README.md) (agentes quânticos em Pong e Snake), mas é independente: só depende de `jax` e `numpy`.

## Por que existe

No PennyLane, a camada `qml.CVNeuralNetLayers` (Killoran et al. 2019) precisa de um dispositivo de Fock. O único, `strawberryfields.fock` (plugin `pennylane-sf`), exige `pennylane<0.30`, ou seja, não funciona com o PennyLane atual (0.45). O `default.gaussian` do PennyLane recusa a porta Kerr. O cvjax roda **o mesmo circuito do template do PennyLane**, porta por porta, em JAX.

## O que tem

| Módulo | Conteúdo |
|---|---|
| `cvjax.fock` | estado puro de M modos como tensor `(D,)*M`; aplicação de portas de 1 e 2 modos por eixo (sem a matriz `D^M × D^M`) |
| `cvjax.gates` | `rotation`, `kerr`, `displacement`, `displacement_alpha`, `squeezing`, `beamsplitter`, com convenções do PennyLane (ħ = 2) e **elementos de matriz exatos** abaixo do cutoff |
| `cvjax.measure` | `mean_photon`, `quad_x`, `quad_p`, `fock_probs` |
| `cvjax.layers` | `interferometer` (malha retangular, igual ao `qml.Interferometer`), `cv_layer`, `cv_neural_net_layers` (igual ao `qml.CVNeuralNetLayers`, mesmos 11 parâmetros e formatos), `init_params` (inicialização da demo do PennyLane) e `run_pennylane_ops` (executa uma lista de operações CV do PennyLane) |
| `cvjax.models.CVQNN` | modelo pronto: codificação por deslocamento → camadas → leitura homódina ⟨x̂⟩ por modo; `leakage` mede a probabilidade perdida acima do cutoff |

```python
import jax, jax.numpy as jnp
from cvjax import CVQNN

net = CVQNN(n_inputs=6, n_modes=3, n_layers=4, cutoff=6)   # 2 entradas por modo: alpha = x1 + i x2
params = net.init(jax.random.PRNGKey(0))
y = jax.jit(net.batched())(params, jnp.zeros((32, 6)))      # (32, 3): <x> de cada modo
```

## Como é exato

- **D e S** usam as formas ordenadas normalmente, por exemplo D(α) = e^{−|α|²/2} e^{α a†} e^{−α* a}. Todo estado intermediário do produto fica abaixo de min(m, n), então truncar não muda nenhum elemento. É a mesma identidade que o QuIara aplica ao estado; aqui ela monta a matriz.
- **BS** conserva o número total de fótons s. Em cada bloco s, a porta é uma rotação de spin s/2, calculada inteira (diagonalizada uma vez em float64) e só depois cortada no cutoff.
- **R e K** são diagonais.

Os testes (`tests/test_cvjax.py`) conferem:
- D e S contra `expm` com cutoff 60;
- o BS contra a The Walrus;
- a amplitude do estado coerente;
- a conservação de fótons;
- as camadas contra a decomposição do `qml.CVNeuralNetLayers`;
- a parte gaussiana (Kerr = 0) contra o `default.gaussian` do PennyLane (⟨n⟩ e ⟨x⟩).

Tudo bate dentro de ~1e-6 (precisão float32).

> Diferença para o `matrix_exp` do gerador truncado (CV-QuForge e o código PyTorch de de Lima et al.): esse método só é exato longe do cutoff. Para estados com poucos fótons, os dois coincidem.

## Cuidado: truncamento

O espaço de Fock é cortado em D níveis por modo. Se o circuito cria muitos fótons, a probabilidade "vaza" acima do cutoff, e `net.leakage(params, x)` mostra quanto. Por isso:
- a codificação é aplicada **uma vez** (D(x)|0⟩), como no artigo de regressão e em Killoran et al.;
- os parâmetros ativos (r, a, k) começam pequenos (`active_sd = 1e-4`, como na demo do PennyLane).

Com 3 modos, 4 camadas, entradas em [−1, 1] e escala 0,5, o vazamento é ≤ 1,4e−3 com D = 6. Reenviar a entrada a cada camada (`reupload=True`) soma os deslocamentos e pode vazar ~99%.

## Exemplo: regressão com 1 modo

`examples/regressao_seno.py` usa a configuração do artigo de regressão: D(x)|0⟩ → [R S R D K] × 3 → ⟨x̂⟩, perda MSE, Adam. Ele chega a MSE ≈ 1,5e−5 em 800 passos (~6 s em CPU).

```bash
pip install -e ".[test,examples]"
```
```bash
python examples/regressao_seno.py
```
```bash
python -m pytest tests
```

## Créditos e referências

Escrito do zero em JAX, mas inspirado e motivado por estes trabalhos. Nenhum arquivo foi copiado; as fórmulas e convenções seguem as fontes abaixo.

- **Tiago de S. Farias**, principalmente, e **Alexandre C. Ricardo**: as implementações **CV-QuForge** (Tiago, GitHub `tiago939`, licença MIT) e **QuIara** (Tiago e Alexandre, licença Apache-2.0), ambas simuladores CV em PyTorch. Usamos as cópias enviadas ao autor deste repositório; não achamos versão pública para linkar. Delas vêm o conjunto de portas e a parametrização do CV-QuForge, e do QuIara a ideia de aplicar portas por eixo do tensor de estado e as identidades ordenadas para D e S. Obrigado, Tiago e Alexandre!
- **G. G. de Lima, T. de S. Farias, A. C. Ricardo, C. J. Villas-Bôas.** *Assessing the Advantages and Limitations of Quantum Neural Networks in Regression Tasks.* arXiv:2509.00854; Quantum Machine Intelligence 8, 20 (2026). Código: [GubioGL/Quantum-Neural-Networks-in-Regression-Tasks](https://github.com/GubioGL/Quantum-Neural-Networks-in-Regression-Tasks) (camada R S R D K, codificação D(x)|0⟩, leitura ⟨a + a†⟩).
- **PennyLane** (Xanadu, Apache-2.0): o template [`CVNeuralNetLayers`](https://docs.pennylane.ai/en/stable/code/api/pennylane.CVNeuralNetLayers.html), o `Interferometer` e a demo [Function fitting with a photonic quantum neural network](https://pennylane.ai/demos/quantum_neural_net) (inicialização dos pesos). Convenções de todas as portas.
- **Strawberry Fields** (Xanadu, Apache-2.0), backend `fock`, e **The Walrus** (Xanadu, Apache-2.0), `fock_gradients`: elementos de matriz exatos de D, S e BS. A The Walrus é a referência numérica dos testes.
- N. Killoran et al., *Continuous-variable quantum neural networks*, Phys. Rev. Research 1, 033063 (2019).

> Confira a grafia dos nomes e os links antes de publicar este repositório separadamente.

## Licença

MIT (veja `LICENSE`). As obras citadas mantêm suas licenças (MIT para o CV-QuForge; Apache-2.0 para QuIara, PennyLane, Strawberry Fields e The Walrus).
