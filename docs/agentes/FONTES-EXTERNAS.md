# Agentes e skills externos (pesquisa no GitHub, 2026-11-27)

Busca por coleções de subagentes/skills do Claude Code que ajudem a organizar o projeto, otimizar o código e, principalmente, implementar os algoritmos (RL, PennyLane, estatística, artigo).

## Adotados (copiados para `.claude/skills/`, sem modificação)

| Skill | Origem | Licença | Usada por | Para quê |
|---|---|---|---|---|
| `pennylane` | [K-Dense-AI/scientific-agent-skills](https://github.com/K-Dense-AI/scientific-agent-skills) | Apache-2.0 (indicado na própria skill) | engenheiro-quantico, fisico-revisor, otimizador | API atual do PennyLane 0.45: circuitos, modelos híbridos com PyTorch, dispositivos, otimização |
| `statistical-analysis` | K-Dense-AI/scientific-agent-skills | MIT | estatistico, cientista-de-experimentos | testes, IC, tamanho de efeito entre sementes |
| `scientific-visualization` | K-Dense-AI/scientific-agent-skills | MIT | cientista-de-experimentos, redator | figuras de publicação (estilos matplotlib, paletas acessíveis) |
| `scientific-writing` | K-Dense-AI/scientific-agent-skills | MIT | redator-do-artigo | estrutura IMRaD, matriz afirmação-evidência |
| `peer-review` | K-Dense-AI/scientific-agent-skills | MIT | revisor-cientifico | revisão por pares simulada |
| `python-performance-optimization` | [wshobson/agents](https://github.com/wshobson/agents) | MIT | otimizador-de-desempenho | perfil (cProfile), gargalos, vetorização |

Licenças completas em [`licencas/`](licencas/). Para atualizar uma skill: copie de novo a pasta do repositório de origem.

## Agentes novos inspirados em agentes externos (reescritos, curtos)

Os agentes externos são listas longas e genéricas: cada chamada carregaria centenas de linhas. Reescrevemos só o que o projeto usa, com modelo barato:

| Agente nosso | Inspirado em | Modelo |
|---|---|---|
| `engenheiro-de-rl` | `reinforcement-learning-engineer` ([VoltAgent/awesome-claude-code-subagents](https://github.com/VoltAgent/awesome-claude-code-subagents), MIT) | Sonnet · medium |
| `otimizador-de-desempenho` | `performance-engineer` (wshobson/agents, MIT) | Sonnet · low |
| `estatistico` | skill `statistical-analysis` (K-Dense) | Sonnet · low |
| `revisor-cientifico` | skill `peer-review` (K-Dense) | Sonnet · medium |

## Avaliados e não adotados (por enquanto)

| Repositório | Por quê não |
|---|---|
| [hesreallyhim/awesome-claude-code](https://github.com/hesreallyhim/awesome-claude-code) | Lista de links; útil para consulta, nada para copiar. |
| [VoltAgent/awesome-agent-skills](https://github.com/VoltAgent/awesome-agent-skills) | Catálogo grande; as skills científicas relevantes já vieram da K-Dense. |
| K-Dense `stable-baselines3` | Opcional: serve para conferir nosso DQN clássico contra uma implementação de referência. Instalar se surgir dúvida no baseline. |
| K-Dense `qiskit`, `cirq` | Decidimos usar PennyLane. |
| K-Dense `literature-review`, `citation-management` | Já temos `pesquisador-de-artigos` + `adicionar-referencia`; avaliar quando o artigo crescer. |
| wshobson `game-development` | Focado em Unity/Godot, não pygame. |
| wshobson `machine-learning-ops` (`ml-engineer`, `data-scientist`) | Voltado a produção/MLOps; nosso caso é pesquisa pequena. |
