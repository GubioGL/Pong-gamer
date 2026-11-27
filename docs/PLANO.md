# Plano de pesquisa

**Pergunta:** em jogos simples com poucos comandos, um agente de aprendizado por reforço com **circuito variacional quântico (VQC / rede neural quântica)** aprende tão bem quanto uma **rede neural clássica**? Com quantos parâmetros, em quanto tempo e com que estabilidade?

**Por que é original** (ver "Lacunas" em [`referencias/ARTIGOS.md`](../referencias/ARTIGOS.md)): os trabalhos de Pong usam pixels do Atari, e lá a parte clássica faz quase todo o trabalho (`lockwood2021atari`, `freinberger2024atari`, `wang2026entanglement`). Ninguém testou Snake com agente quântico. Aqui a observação é pequena, então o agente pode ser quase todo quântico e a comparação fica limpa.

## Ambientes

| Jogo | Ações | Observação | Recompensa | Status |
|---|---|---|---|---|
| Pong (`games/pong`) | 3: parado, cima, baixo | 6: bola x, y, vx, vy; raquete do jogador; raquete da CPU (todos em [-1, 1]) | +1 ponto feito, −1 ponto sofrido, +0.1 por rebatida | ✅ pronto |
| Snake (`games/snake`) | 4: cima, baixo, esquerda, direita | 8 (proposta): direção da comida (dx, dy), perigo nas 4 direções, direção atual (dx, dy) | +1 comida, −1 morte, −0.01 por passo | ⬜ fazer |

Uma observação de tamanho n vira n qubits (1 número por qubit, como ângulo).

## Fases

- [x] **0. Organização:** estrutura do paper-template, referências, equipe de agentes/skills, Pong terminado e testado.
- [ ] **1. Snake:** `games/snake/env.py` + `play.py` + testes (skill `criar-jogo-pygame`). Agente: `engenheiro-de-jogos`.
- [ ] **2. DQN + rede clássica:** `src/rl/dqn.py` (replay, target network, ε-greedy) e `src/agentes/mlp.py`. Validar no Pong curto (`points_to_win=1` ou `max_steps` pequeno) contra o agente `seguidor`. Agente: `engenheiro-quantico`.
- [ ] **3. Agente quântico:** `src/agentes/vqc.py` (PennyLane + PyTorch, re-uploading, Q(a) = w·⟨Z_a⟩), skill `agente-quantico-variacional`. Revisão do `fisico-revisor` antes de treinar longo.
- [ ] **4. Experimentos:** protocolo da skill `comparar-quantico-classico` (VQC, MLP com o mesmo nº de parâmetros, MLP padrão; ≥ 5 sementes; mesmo orçamento). Dados em `data/experimentos/`, figuras em `paper/figures/`. Agente: `cientista-de-experimentos`.
- [ ] **5. Extras (se der tempo):** policy gradient (`jerbi2021pqp`); efeito do emaranhamento (anel de CZ vs. sem CZ, como em `wang2026entanglement`); ruído de hardware.
- [ ] **6. Artigo:** `redator-do-artigo` → `fisico-revisor` → conferir entradas `note = {check}` do `.bib` → Overleaf.

## Métricas

Retorno por episódio (média ± desvio entre sementes) · episódios/passos até um limiar · avaliação greedy final · nº de parâmetros treináveis · tempo de parede por passo.

## Riscos

- **Tempo de simulação:** cada passo do VQC custa mais que um MLP. Para reduzir: poucos qubits (≤ 8), `default.qubit` com backprop e lotes pequenos; medir o tempo antes de rodar as sementes.
- **Instabilidade entre sementes** (`franz2022instab`): por isso várias sementes e desvio-padrão nas figuras.
- **Barren plateaus** (`mcclean2018barren`): poucas camadas, inicialização pequena.
- **Episódios longos no Pong:** agentes que não erram fazem ralis enormes; nos treinos usar `points_to_win` baixo e `max_steps` curto.

## Decisões para o dono

1. Biblioteca quântica: **PennyLane + PyTorch** (sugerida; tutorial e integração simples) ou Qiskit / TensorFlow Quantum?
2. Algoritmo principal: **DQN** (sugerido; `skolik2022gym`) ou policy gradient?
3. Conectar `paper/` ao Overleaf (passos no README do paper-template: variável `OVERLEAF_PROJECT` e segredo `OVERLEAF_TOKEN`).
