# Relatório de atualizações

O que mudou em cada commit, em linguagem simples, para quem acompanha o projeto.
Mais recente em cima. Formato e regras: skill `registrar-atualizacao` (o agente `escriba` mantém este arquivo).

## 2026-11-27 (2ª rodada): cobrinha, agentes quânticos e rascunho do artigo

| Commit | O que mudou | Arquivos principais |
|---|---|---|
| (este commit) | README e este relatório atualizados. | `README.md`, `ATUALIZACOES.md` |
| `8661220` | **Rascunho do artigo** com estrutura completa e equações. Figuras reais: os dois jogos, os dois circuitos e o custo medido por passo. Figuras de resultados **ilustrativas** (dados sintéticos, com marca d'água), mostrando a comparação que queremos fazer. | `paper/main.tex`, `run/figuras_artigo.py`, `run/medir_custo.py` |
| `e62db01` | **Agentes de IA:** DQN único para todos, circuito variacional de **qubits**, rede quântica de **variáveis contínuas** (gaussiana) e duas redes clássicas. Tudo em PennyLane + JAX; o simulador CV foi conferido contra o PennyLane. Script de treino. | `src/rl/dqn.py`, `src/agentes/`, `run/treinar.py` |
| `4408f0a` | **Jogo da cobrinha** jogável e como ambiente de IA (4 ações, 8 números de observação), com o mesmo visual retrô. | `games/snake/`, `cobrinha.py` |
| `eb24548` | **Pong com visual retrô** de fliperama: fundo preto, peças brancas, placar em blocos, bipes. O fundo verde continua com `--tema campo`. | `games/retro.py`, `games/pong/play.py` |
| `70eac76` | Pesquisa no GitHub de agentes e skills: 6 skills copiadas (PennyLane, estatística, figuras, escrita, revisão por pares, desempenho) e 4 agentes novos (RL, desempenho, estatística, revisão). | `.claude/`, `docs/agentes/FONTES-EXTERNAS.md` |

**Por quê:** ter os dois jogos e os três tipos de agente prontos para treinar, e o artigo estruturado antes dos experimentos.
**Mudança de plano:** o backend numérico passou de PyTorch para **JAX**. O servidor do PyTorch para CPU estava bloqueado no ambiente de desenvolvimento, e o JAX instala em CPU pelo PyPI e acelera os circuitos no PennyLane.
**Próximos passos:** revisão do `fisico-revisor` nos dois circuitos; validar que o MLP aprende num treino completo; rodar 5–10 sementes por agente; trocar as figuras ilustrativas pelas reais.

## 2026-11-27: Organização do projeto e Pong terminado

| Commit | O que mudou | Arquivos principais |
|---|---|---|
| (este commit) | Plano de pesquisa, README novo e este relatório. | `docs/PLANO.md`, `README.md`, `ATUALIZACOES.md` |
| `e5a02e3` | **Pong terminado.** Regras separadas da janela, para servir de ambiente de IA (3 ações, observação de 6 números). Corrigidos: caminhos de imagem do Windows, raquete que pulava, bola que grudava, raquete da CPU saindo da tela. Novidades: ângulo da bola, aceleração, níveis da CPU, telas de início/pausa/fim, partida até 7, fundo do campo, agente "seguidor", 9 testes. | `games/pong/env.py`, `games/pong/play.py`, `tests/` |
| `b0e05b0` | Equipe de 8 subagentes do Claude Code, 6 skills e o comando `/maestro`. Cada agente usa o modelo mais barato que resolve a tarefa (Haiku/Sonnet/Opus). | `.claude/`, `CLAUDE.md`, `.mcp.json` |
| `e4abc7d` | Lista anotada de 32 artigos sobre agentes quânticos em jogos, comparações com redes clássicas, fundamentos e baselines, com o `.bib` para o artigo. | `referencias/ARTIGOS.md`, `paper/references.bib` |
| `f9247ed` | Estrutura do `paper-template`: `src/`, `run/`, `data/`, `paper/` (LaTeX com sincronização Overleaf). | `paper/main.tex`, `src/paths.py`, `.github/` |

**Por quê:** preparar o repositório para comparar agentes quânticos variacionais e redes neurais clássicas em Pong e Snake, com as fontes reunidas e o fluxo de trabalho organizado.
**Próximos passos:** jogo Snake (fase 1); DQN + rede clássica (fase 2); agente quântico (fase 3). Ver `docs/PLANO.md`.
