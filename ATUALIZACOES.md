# Relatório de atualizações

O que mudou em cada commit, em linguagem simples, para quem acompanha o projeto.
Mais recente em cima. Formato e regras: skill `registrar-atualizacao` (o agente `escriba` mantém este arquivo).

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
