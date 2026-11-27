---
name: checklist-de-commit
description: Verificações antes de commit/push neste repositório: testes, jogo sem tela, scripts de run/, figuras e referências. Use antes de cada commit e no qa-testador.
---
# Checklist de commit

Rode e marque; relate ❌ com a saída do erro. Nunca marque ✅ sem ter executado.

- [ ] `pip install -r run/requirements.txt` (se faltar pacote)
- [ ] `python -m pytest -q`
- [ ] `SDL_VIDEODRIVER=dummy python -m games.pong.play --frames 120` (e de cada jogo em `games/`)
- [ ] Scripts de `run/` tocados rodam em modo rápido (poucos episódios)
- [ ] Figuras novas em `paper/figures/*.pdf` (nenhum PNG commitado fora de `games/*/assets/`)
- [ ] Citações novas em `paper/main.tex` existem em `paper/references.bib`
- [ ] `ATUALIZACOES.md` atualizado (escriba)
