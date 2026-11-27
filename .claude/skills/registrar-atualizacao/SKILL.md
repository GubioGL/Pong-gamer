---
name: registrar-atualizacao
description: Formato do relatório de atualizações (ATUALIZACOES.md) e do log dos agentes, para que outras pessoas saibam o que mudou em cada commit. Use ao fim de cada tarefa, antes do push.
---
# Registrar atualização

## `ATUALIZACOES.md` (raiz), mais recente em cima
```
## AAAA-MM-DD: <título curto da rodada>
| Commit | O que mudou | Arquivos principais |
|---|---|---|
| `abc1234` | frase curta, para quem não programa | `games/pong/env.py` |

**Por quê:** 1-2 frases. **Próximos passos:** 1-3 itens.
```
- Pegue os commits com `git log --oneline <base>..HEAD` e arquivos com `git show --stat --format= <hash>`.
- O commit que adiciona a própria entrada não sabe o próprio hash: escreva `(este commit)`.
- Uma linha por commit; não cole diff.

## `docs/agentes/log.md`
Uma linha por rodada: `AAAA-MM-DD | objetivo | agentes acionados (modelo) | resultado | pendências`.

## Mensagem de commit
Título em português, ≤ 70 caracteres, verbo no presente ("Termina o Pong..."). Corpo: o que e por quê em 2-5 linhas.
