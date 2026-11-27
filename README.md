# Pong-gamer: agentes quânticos x redes neurais clássicas em jogos simples

Jogos simples em Python (Pong, e em breve Snake) que servem para duas coisas:

1. **Jogar:** você contra o computador.
2. **Pesquisar:** treinar agentes de IA com **circuitos variacionais quânticos / redes neurais quânticas** e comparar com **redes neurais clássicas**, nos mesmos jogos e com comandos simples (cima, baixo, esquerda, direita).

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
| ↑ / W, ↓ / S | mover a raquete azul (direita) |
| Espaço | começar / jogar de novo |
| P | pausar |
| R | reiniciar |
| Esc | sair |

Opções: `--nivel facil|normal|dificil`, `--pontos 3` (partida mais curta), `--agente seguidor` (assistir um agente simples jogar).

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

Sem pygame e determinístico por `seed`: roda rápido e sem janela para treinar.

## Estrutura (modelo [paper-template](https://github.com/GubioGL/paper-template))

```text
├── games/        jogos: env.py (regras) + play.py (janela) + assets/
├── src/          código reutilizável (agentes, DQN) + paths.py (save_figure)
├── run/          scripts de treino e figuras + requirements.txt
├── data/         resultados dos experimentos (CSV)
├── paper/        artigo LaTeX (sincroniza com Overleaf) + references.bib
├── referencias/  ARTIGOS.md: lista anotada de artigos
├── docs/         PLANO.md, log dos agentes
├── tests/        python -m pytest
└── .claude/      agentes, skills e /maestro para o Claude Code
```

## Trabalhando com o Claude Code

O projeto tem uma equipe de subagentes e skills (receita do projeto q-learning). Cada agente usa o modelo mais barato que dá conta do trabalho dele, e o Opus fica só para a revisão crítica. Detalhes em [`CLAUDE.md`](CLAUDE.md).

- `/maestro <pedido>`: divide o trabalho entre os agentes certos.
- Skills: `criar-jogo-pygame`, `agente-quantico-variacional`, `comparar-quantico-classico`, `adicionar-referencia`, `registrar-atualizacao`, `checklist-de-commit`.

## Testes

```bash
python -m pytest
```
