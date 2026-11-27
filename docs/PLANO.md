# Plano de pesquisa

**Pergunta:** em jogos simples com poucos comandos, um agente de aprendizado por reforço com **circuito variacional quântico (VQC / rede neural quântica)** aprende tão bem quanto uma **rede neural clássica**? Com quantos parâmetros, em quanto tempo e com que estabilidade?

**Por que é original** (ver "Lacunas" em [`referencias/ARTIGOS.md`](../referencias/ARTIGOS.md)): os trabalhos de Pong usam pixels do Atari, e lá a parte clássica faz quase todo o trabalho (`lockwood2021atari`, `freinberger2024atari`, `wang2026entanglement`). Ninguém testou Snake com agente quântico. Aqui a observação é pequena, então o agente pode ser quase todo quântico e a comparação fica limpa.

## Ambientes

| Jogo | Ações | Observação | Recompensa | Status |
|---|---|---|---|---|
| Pong (`games/pong`) | 3: parado, cima, baixo | 6: bola x, y, vx, vy; raquete do jogador; raquete da CPU (todos em [-1, 1]) | +1 ponto feito, −1 ponto sofrido, +0.1 por rebatida | ✅ pronto |
| Snake (`games/snake`) | 4: cima, baixo, esquerda, direita | 8: direção da comida (dx, dy), perigo nas 4 direções, direção atual (dx, dy) | +1 comida, −1 morte, −0.01 por passo; corte após 200 passos sem comer | ✅ pronto |

Uma observação de tamanho n vira n qubits (agente de qubits, 1 número por qubit), n/2 modos ópticos (agente CV com Kerr, 2 números por modo como deslocamento complexo) ou n modos (ablação CV gaussiana).

## Agentes (todos treinados pelo mesmo DQN, `src/rl/dqn.py`)

| Agente | Arquivo | Parâmetros (Pong / Snake, L = 4) |
|---|---|---|
| `vqc`: circuito variacional de **qubits** (PennyLane `default.qubit`) | `src/agentes/vqc.py` | 90 / 120 |
| `cv`: rede neural quântica de **variáveis contínuas** com Kerr (`CVNeuralNetLayers` de Killoran et al., base de Fock com cutoff 6, biblioteca `libs/cvjax`) | `src/agentes/cvfock.py` | 144 / 224 |
| `cv_gauss`: rede CV **só gaussiana** (ablação sem Kerr; `default.gaussian` + simulador JAX equivalente) | `src/agentes/cvqnn.py` | 166 / 224 |
| `mlp_igual`: rede clássica com ≈ o mesmo nº de parâmetros do `vqc` | `src/agentes/mlp.py` | 93 / 121 |
| `mlp`: rede clássica padrão 2×64 | `src/agentes/mlp.py` | 4 803 / 4 996 |

Treinar: `python run/treinar.py --jogo pong --agente vqc cv cv_gauss mlp mlp_igual --sementes 0 1 2 3 4`.

## Fases

- [x] **0. Organização:** estrutura do paper-template, referências, equipe de agentes/skills, Pong terminado e testado.
- [x] **1. Snake:** `games/snake/env.py` + `play.py` + testes; visual retrô igual ao Pong.
- [x] **2. DQN + rede clássica:** `src/rl/dqn.py` (JAX + optax: replay, target network, Double DQN, ε-greedy) e `src/agentes/mlp.py`. Teste de fumaça OK. *Falta:* validar que o MLP aprende Pong/Snake num treino completo (engenheiro-de-rl).
- [x] **3. Agentes quânticos:** `vqc.py` (qubits), `cvfock.py` (CV com Kerr via `libs/cvjax`, conferida contra PennyLane e The Walrus) e `cvqnn.py` (CV gaussiana, ablação). *Falta:* revisão do `fisico-revisor` antes de treinar longo; acelerar o agente CV no Snake (110 ms/passo, ~3 h por semente) com o `otimizador-de-desempenho`, por exemplo `lax.scan` nas camadas ou cutoff 5.
- [ ] **4. Experimentos:** protocolo da skill `comparar-quantico-classico` (VQC, MLP com o mesmo nº de parâmetros, MLP padrão; ≥ 5 sementes; mesmo orçamento). Dados em `data/experimentos/`, figuras em `paper/figures/`. Agente: `cientista-de-experimentos`.
- [ ] **5. Extras (se der tempo):** policy gradient (`jerbi2021pqp`); efeito do emaranhamento (anel de CZ vs. sem CZ, `entangle=False`; como em `wang2026entanglement`); cutoff de Fock 5–8 e penalidade de vazamento no agente CV; versão de íons aprisionados da porta Kerr (como em `delima2026regression`); ruído de hardware.
- [ ] **6. Artigo:** ✅ rascunho em `paper/main.tex` (estrutura, métodos, tabelas reais de parâmetros e custo, figuras reais dos jogos e circuitos, e figuras **ilustrativas** de resultados marcadas como sintéticas). Falta: resultados reais (`run/treinar.py` → figuras), `fisico-revisor`, `revisor-cientifico`, conferir entradas `note = {check}` do `.bib`, Overleaf.

## Métricas

Retorno por episódio (média ± desvio entre sementes) · episódios/passos até um limiar · avaliação greedy final · nº de parâmetros treináveis · tempo de parede por passo.

## Riscos

- **Tempo de simulação** (medido, `data/custo/custo_por_agente.csv`, CPU, lote 32): por passo de gradiente, VQC 5,9 ms (Pong) e 26,6 ms (Snake); CV 12,5 ms e 16,1 ms; MLP ~0,3 ms. Uma semente de 100 mil passos leva ~10–45 min para os quânticos, mais ~20 s de compilação JIT.
- **Instabilidade entre sementes** (`franz2022instab`): por isso várias sementes e desvio-padrão nas figuras.
- **Barren plateaus** (`mcclean2018barren`): poucas camadas, inicialização pequena.
- **Episódios longos no Pong:** agentes que não erram fazem ralis enormes; nos treinos usar `points_to_win` baixo e `max_steps` curto.

## Decisões

- ✅ Biblioteca: **PennyLane**, com sistemas **discretos (qubits)** e **contínuos (modos ópticos)** (decisão do dono, 2026-11-27). Backend numérico **JAX** em vez de PyTorch: o servidor do PyTorch para CPU estava bloqueado no ambiente de desenvolvimento, e o JAX roda em CPU (Windows, Linux, Mac) a partir do PyPI, com `jit`/`vmap` rápidos no PennyLane.
- ✅ Algoritmo: **DQN** (decisão do dono).
- ⬜ Conectar `paper/` ao Overleaf (passos no README do paper-template: variável `OVERLEAF_PROJECT` e segredo `OVERLEAF_TOKEN`).
