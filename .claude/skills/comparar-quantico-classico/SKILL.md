---
name: comparar-quantico-classico
description: Protocolo de comparação justa entre agente quântico (VQC) e rede neural clássica (MLP) nos jogos: mesmo algoritmo, sementes, métricas, formato dos dados e figuras. Use ao planejar, rodar ou relatar experimentos.
---
# Protocolo de comparação quântico x clássico

Motivação: resultados de QRL variam muito entre sementes e baselines (`franz2022instab`, `skolik2022gym`).

1. **Mesmo algoritmo e orçamento:** DQN igual (`src/rl/dqn.py`), mesmo nº de passos de ambiente, mesmo ε-greedy, mesmo replay, mesmo γ.
2. **Agentes:** `vqc` (L camadas) · `mlp_pequeno` (≈ mesmo nº de parâmetros do VQC) · `mlp_padrao` (ex.: 2×64). Registre nº de parâmetros de cada um.
3. **Sementes:** ≥ 5 por agente (10 para o artigo). Semente controla ambiente, rede e replay.
4. **Métricas:** retorno por episódio, pontos ganhos/perdidos (Pong) ou comprimento (Snake), episódios até atingir o limiar, tempo de parede, avaliação final greedy (ε = 0) em 20 episódios.
5. **Dados:** `data/experimentos/<jogo>/<agente>/seed_<n>.csv` com colunas `episode,return,steps,epsilon,wall_time`; `config.json` ao lado com todos os hiperparâmetros.
6. **Figuras:** média ± desvio entre sementes, eixo x = passos de ambiente; salvar com `paths.save_figure(fig, "<jogo>_curvas")`.
7. **Relato:** tabela agente | parâmetros | retorno final (média ± desvio) | passos até o limiar | tempo. Diferença pequena com desvio grande = "comparável", não "melhor".
