---
name: agente-quantico-variacional
description: Receita dos dois agentes quânticos do projeto em PennyLane + JAX, o de qubits (VQC, src/agentes/vqc.py) e o de variáveis contínuas com Kerr (CV-QNN, src/agentes/cvfock.py + libs/cvjax) e a ablação CV gaussiana (src/agentes/cvqnn.py), como Q-networks do DQN para jogos com observação pequena e 3-4 ações. Baseada em Skolik et al. 2022 e Killoran et al. 2019. Use ao implementar ou alterar um agente quântico.
---
# Agentes quânticos (qubits e variáveis contínuas)

Referências: `skolik2022gym`, `jerbi2021pqp`, `perezsalinas2020reuploading`, `killoran2019cvqnn`, `delima2026regression` (ver `referencias/ARTIGOS.md`).
Interface comum (lida pelo `src/rl/dqn.py`): `init(key)`, `apply(params, obs) -> Q (n_actions,)`, `labels(params)` ("base"/"out"), `config`.

## Qubits: `QubitQNet` (`src/agentes/vqc.py`)
- 1 qubit por número da observação (Pong 6, Snake 8).
- L vezes: camada variacional (RY, RZ por qubit + anel de CZ) → codificação RX(λ·x_i), λ treinável começando em π/2 (±π daria a mesma rotação a menos de fase global).
- Camada variacional final. Saída Q(a) = w_a ⟨Z_{2a} Z_{2a+1}⟩ + b_a.
- `default.qubit`, `interface="jax"`, `diff_method="backprop"`; `jax.vmap` no lote e `jax.jit`.

## Variáveis contínuas com Kerr: `CVFockQNet` (`src/agentes/cvfock.py`, agente `cv`)
- Camadas `CVNeuralNetLayers` (interferômetro, squeezing, interferômetro, deslocamento, Kerr), simuladas na base de Fock pela biblioteca `libs/cvjax` (README e testes lá).
- Entrada: 2 números por modo, D(α) com α = s·x₁ + i·s·x₂, **uma vez** no vácuo (como D(x)|0⟩ do artigo de regressão). Reenviar deslocamentos soma fótons e vaza o cutoff.
- Saída: Q(a) = w_a ⟨x̂_a⟩ + b_a, um modo por ação (Pong 3, Snake 4).
- Cutoff 6 por modo; meça `leakage` (deve ficar ≲ 1e-3). Mudou portas? Rode `python -m pytest libs/cvjax/tests`.

## Ablação gaussiana: `CVQNet` (`src/agentes/cvqnn.py`, agente `cv_gauss`)
- 1 modo óptico por número da observação, começando no vácuo.
- L vezes: Squeezing(r, φ) e Displacement(d, ψ) treináveis → Rotation(λ·x_i) (dados) → cadeia de Beamsplitter(θ, φ).
- Saída Q(a) = w_a (⟨n_{2a}⟩ − ⟨n_{2a+1}⟩) + b_a.
- O circuito é definido em PennyLane (`reference_qnode`, `default.gaussian`). O treino usa um simulador gaussiano equivalente em JAX (média + covariância, ħ = 2), porque o `default.gaussian` roda um estado por vez e só faz parameter-shift. **Mudou o circuito? Mude os dois e rode `tests/test_agentes.py`** (compara com o PennyLane).
- Só portas gaussianas: o dado entra como fase, então a saída é não linear (tipo Fourier) em x. Serve para medir quanto o Kerr acrescenta.

## Treino
- DQN idêntico para todos (`src/rl/dqn.py`, configurações em `src/experimentos.py`).
- `lr` para os parâmetros do circuito; `lr_out` (maior, ~10×) para w e b (Skolik et al. usam taxas separadas: conferir os valores no artigo).
- Conte parâmetros com `rl.dqn.count_parameters(params)`; o `mlp_igual` é ajustado ao nº de parâmetros do VQC.
- A primeira chamada compila (~15–25 s). Custo medido por passo: `data/custo/custo_por_agente.csv` (`run/medir_custo.py`).

## Cuidados
- Barren plateaus (`mcclean2018barren`): poucas camadas, inicialização pequena.
- Sem afirmar vantagem quântica: compare pelo protocolo da skill `comparar-quantico-classico`.
