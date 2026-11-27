# Pong-gamer: agentes quânticos x redes neurais clássicas em jogos simples

Jogos simples em Python, **Pong** e **cobrinha (Snake)**, com visual retrô de fliperama, que servem para duas coisas:

1. **Jogar:** você contra o computador.
2. **Pesquisar:** treinar agentes de IA quânticos, de **qubits** (circuito variacional) e de **variáveis contínuas** (rede neural quântica fotônica), e comparar com **redes neurais clássicas**, nos mesmos jogos e com comandos simples (cima, baixo, esquerda, direita). Tudo em PennyLane, treinado pelo mesmo DQN.

O rascunho do artigo está em [`paper/main.tex`](paper/main.tex).

O plano de pesquisa está em [`docs/PLANO.md`](docs/PLANO.md), os artigos de referência em [`referencias/ARTIGOS.md`](referencias/ARTIGOS.md), e o que mudou em cada commit em [`ATUALIZACOES.md`](ATUALIZACOES.md).

## Jogar Pong

```bash
pip install -r run/requirements.txt
```
```bash
python pongamer.py
```

| Tecla | Ação |
|---|---|
| ↑ / W, ↓ / S | mover a sua raquete (direita) |
| Espaço | começar / jogar de novo |
| P | pausar |
| R | reiniciar |
| M | liga/desliga o som |
| Esc | sair |

Opções: `--nivel facil|normal|dificil`, `--pontos 3` (partida mais curta), `--agente seguidor` (assistir um agente simples jogar), `--tema campo` (fundo verde), `--sem-som`.

## Jogar a cobrinha

```bash
python cobrinha.py
```

Setas ou WASD movem; Espaço começa; P pausa; R reinicia; M som; Esc sai.
Opções: `--grade 12x12` (a grade dos experimentos), `--nivel facil|normal|dificil`, `--agente guloso`.

## Pong como ambiente de IA

```python
from games.pong import PongEnv

env = PongEnv(seed=0)
obs = env.reset()                         # 6 números em [-1, 1]
obs, reward, done, info = env.step(1)     # 0 parado, 1 cima, 2 baixo
```

| | |
|---|---|
| Ações | 0 parado · 1 cima · 2 baixo |
| Observação | bola x, y, vx, vy · raquete do jogador · raquete da CPU |
| Recompensa | +1 ponto feito · −1 ponto sofrido · +0.1 por rebatida |
| Fim | alguém chega a `points_to_win` (7) ou `max_steps` |

Sem pygame e determinístico por `seed`: roda rápido e sem janela para treinar. A cobrinha é igual: `from games.snake import SnakeEnv`, 4 ações (cima, baixo, esquerda, direita) e 8 números de observação.

## Agentes e treino

| Agente | O que é | Arquivo |
|---|---|---|
| `vqc` | circuito variacional de **qubits** (1 qubit por número da observação) | `src/agentes/vqc.py` |
| `cv` | rede neural quântica de **variáveis contínuas** com Kerr (`CVNeuralNetLayers`, base de Fock; 2 números por modo óptico) | `src/agentes/cvfock.py` + [`libs/cvjax`](libs/cvjax/README.md) |
| `cv_gauss` | rede CV só gaussiana (ablação sem Kerr; 1 modo por número) | `src/agentes/cvqnn.py` |
| `mlp_igual` | rede clássica com o mesmo nº de parâmetros do `vqc` | `src/agentes/mlp.py` |
| `mlp` | rede clássica padrão 2×64 | `src/agentes/mlp.py` |

```bash
python run/treinar.py --jogo pong --agente vqc cv cv_gauss mlp mlp_igual --sementes 0 1 2 3 4
```
```bash
python run/figuras_artigo.py
```

As curvas vão para `data/experimentos/<jogo>/<agente>/seed_<n>.csv` e as figuras para `paper/figures/`. O backend numérico é JAX (`pennylane==0.45.1` exige `jax==0.7.1`; já fixado em `run/requirements.txt`).

## Estrutura (modelo [paper-template](https://github.com/GubioGL/paper-template))

```text
├── games/        pong/ e snake/: env.py (regras) + play.py (janela); retro.py (visual e sons)
├── src/          agentes/ (vqc, cvfock, cvqnn, mlp), rl/dqn.py, experimentos.py, paths.py
├── libs/cvjax/   biblioteca CV-QNN na base de Fock em JAX (independente, pode virar repositório próprio)
├── run/          scripts de treino e figuras + requirements.txt
├── data/         resultados dos experimentos (CSV)
├── paper/        artigo LaTeX (sincroniza com Overleaf) + references.bib
├── referencias/  ARTIGOS.md: lista anotada de artigos
├── docs/         PLANO.md, log dos agentes
├── tests/        python -m pytest
└── .claude/      agentes, skills e /maestro para o Claude Code
```

## Trabalhando com o Claude Code

O projeto tem uma equipe de 12 subagentes e 12 skills (receita do projeto q-learning + skills e agentes externos do GitHub, créditos em [`docs/agentes/FONTES-EXTERNAS.md`](docs/agentes/FONTES-EXTERNAS.md)). Cada agente usa o modelo mais barato que dá conta do trabalho dele, e o Opus fica só para a revisão crítica. Detalhes em [`CLAUDE.md`](CLAUDE.md).

- `/maestro <pedido>`: divide o trabalho entre os agentes certos.
- Skills do projeto: `criar-jogo-pygame`, `agente-quantico-variacional`, `comparar-quantico-classico`, `adicionar-referencia`, `registrar-atualizacao`, `checklist-de-commit`.
- Skills externas: `pennylane`, `statistical-analysis`, `scientific-visualization`, `scientific-writing`, `peer-review`, `python-performance-optimization`.

## Testes

```bash
python -m pytest
```
