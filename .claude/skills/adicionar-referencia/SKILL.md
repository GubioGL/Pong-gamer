---
name: adicionar-referencia
description: Como registrar um artigo novo nas referências do projeto (referencias/ARTIGOS.md + paper/references.bib) sem duplicar nem inventar dados. Use sempre que uma fonte nova for encontrada.
---
# Adicionar referência

1. `grep -i "<arXiv id ou palavra do título>" referencias/ARTIGOS.md paper/references.bib`: se já existe, só complemente.
2. Chave BibTeX: `<sobrenome do 1º autor><ano><palavra>` (ex.: `skolik2022gym`).
3. Em `ARTIGOS.md`, na seção certa (A jogos · B QRL · C comparações · D fundamentos · E baselines clássicos · F jogos quânticos · G ferramentas), uma linha: ⭐ | chave | referência com link | o que interessa ao projeto (1 frase) | status.
4. Status ✔ só se título/autores/local foram confirmados em página aberta ou busca; senão ◐ e `checknote = {...}` no `.bib`.
5. Em `paper/references.bib`, na mesma seção. Nunca invente DOI, volume ou página.
6. Se a fonte muda o plano (ex.: novo resultado em Pong/Snake), acrescente uma linha em "Lacunas" ou avise o maestro.
