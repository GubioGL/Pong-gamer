---
description: Orquestra a equipe de subagentes: classifica o pedido, escolhe a fase e aciona só quem for necessário, no modelo mais barato que resolve
argument-hint: <o que você quer fazer>
---
Atue como o **maestro** do projeto (veja `CLAUDE.md`, seções Equipe, Modelos e Roteamento).

Pedido: $ARGUMENTS

1. Classifique: pesquisa · jogo · agente quântico/clássico · experimento · artigo · qualidade · tarefa pequena.
2. Tarefa pequena (typo, ajuste de 1 arquivo, pergunta)? Faça direto, sem subagente.
3. Senão, diga em 3–5 linhas o plano: quais agentes, ordem, o que roda em paralelo. Faltou decisão do dono? Pergunte antes.
4. Acione os subagentes com prompt completo (contexto, arquivos, critério de pronto, tamanho máximo do relatório). Rode em paralelo os independentes. Só passe `model: "opus"` para tarefa difícil de verdade.
5. Se `fisico-revisor` ou `qa-testador` acharem problema, devolva ao autor e repita.
6. Feche com `escriba` (ATUALIZACOES.md + log) e resuma: feito, falta, próxima fase.
