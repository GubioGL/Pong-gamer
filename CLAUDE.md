# Pong-gamer: agentes quânticos x clássicos em jogos simples (guia do Claude Code)

Projeto de pesquisa: comparar **redes neurais quânticas / circuitos variacionais (VQC)** com **redes neurais clássicas (MLP)** como agentes de aprendizado por reforço em jogos simples com poucos comandos (Pong: cima/baixo; Snake: cima/baixo/esquerda/direita).
Estrutura do `GubioGL/paper-template`. O dono escreve em português; código e artigo em inglês.
Leia `README.md`, `docs/PLANO.md` e `referencias/ARTIGOS.md`. Relatório de mudanças: `ATUALIZACOES.md`.

Comandos: `python -m pytest -q` · `python -m games.pong.play` · `SDL_VIDEODRIVER=dummy python -m games.pong.play --frames 120`.

## Layout (regras do template)
- `games/<jogo>/`: jogos (`env.py` lógica pura sem pygame; `play.py` janela). Ver skill `criar-jogo-pygame`.
- `src/`: código importável (agentes, DQN, `paths.py`). `run/`: scripts executáveis, sempre com o bloco de cabeçalho do template (copie de um script existente em `run/`).
- Figuras: sempre PDF em `paper/figures/` via `paths.save_figure(fig, nome)`. Nunca commitar PNG (exceto `games/*/assets/`).
- Dados das figuras em `data/`. Antes de escrever um número no artigo, confira em `data/`.
- `paper/` sincroniza com Overleaf (`.github/workflows/overleaf-sync.yml`) quando `OVERLEAF_PROJECT` estiver configurado; até lá o job não roda.

## Papel do maestro (sessão principal)
Entende o pedido, escolhe a fase, aciona só os subagentes necessários (em paralelo quando independentes), junta os resultados e decide o próximo passo. Subagentes não chamam subagentes. `/maestro <pedido>` força esse fluxo.

## Equipe (`.claude/agents/`) e modelos (economia de tokens)

| Agente | Faz | Modelo · esforço |
|---|---|---|
| `pesquisador-de-artigos` | busca artigos, atualiza `referencias/` e `.bib` | Haiku · medium |
| `escriba` | `ATUALIZACOES.md`, log, mensagens de commit | Haiku · low |
| `qa-testador` | pytest, jogo sem tela, scripts | Haiku · low |
| `cientista-de-experimentos` | roda sementes, salva `data/`, figuras | Sonnet · low |
| `engenheiro-de-jogos` | jogos pygame + ambiente `reset/step` | Sonnet · medium |
| `engenheiro-quantico` | agentes quânticos (qubits; variáveis contínuas com Kerr via `libs/cvjax`; CV gaussiano) e MLP | Sonnet · medium |
| `redator-do-artigo` | `paper/main.tex` | Sonnet · medium |
| `engenheiro-de-rl` | DQN compartilhado, treino que não converge | Sonnet · medium |
| `otimizador-de-desempenho` | perfil e aceleração do treino/simulação | Sonnet · low |
| `estatistico` | IC, testes e tamanho de efeito entre sementes | Sonnet · low |
| `revisor-cientifico` | revisão por pares simulada do artigo (só leitura) | Sonnet · medium |
| `fisico-revisor` | revisa circuito, comparação justa e afirmações (só leitura) | Opus · medium |

- Maestro: Sonnet, esforço médio (`.claude/settings.json`). Para arquitetura difícil ou física nova: `/model opus`.
- Tarefas mecânicas (registrar, testar, buscar e listar) vão para Haiku. Escrever código e texto vai para Sonnet. O Opus fica só para a revisão crítica, onde um erro custa caro.
- Exceção: tarefa realmente difícil (bug que resiste, ansatz novo, análise estatística delicada): o maestro passa `model: "opus"` só naquela chamada.
- Para gastar menos: tarefa pequena sem subagente; `/clear` entre tarefas sem relação; `/compact` em conversa longa; relatórios curtos (máx. ~10 itens); revisores salvam relatório em arquivo e devolvem só o resumo; não reler arquivos grandes inteiros (use Grep/offset).

## Skills do projeto (`.claude/skills/`)
Do projeto: `criar-jogo-pygame` · `agente-quantico-variacional` · `comparar-quantico-classico` · `adicionar-referencia` · `registrar-atualizacao` · `checklist-de-commit`.
Externas (créditos em `docs/agentes/FONTES-EXTERNAS.md`): `pennylane` · `statistical-analysis` · `scientific-visualization` · `scientific-writing` · `peer-review` · `python-performance-optimization`.
MCP em `.mcp.json`: `context7` (documentação atual de PennyLane/JAX/optax/pygame) e `parallel-search` (busca web grátis).

## Roteamento
- **Jogo novo ou bug de jogo:** engenheiro-de-jogos → qa-testador → escriba.
- **Agente quântico/clássico:** engenheiro-quantico → fisico-revisor → qa-testador → escriba. Treino não converge: engenheiro-de-rl. Lento: otimizador-de-desempenho.
- **Experimento:** (fisico-revisor confere o protocolo) → otimizador-de-desempenho (tempo por passo) → cientista-de-experimentos → estatistico → redator-do-artigo → revisor-cientifico → escriba.
- **Referências:** pesquisador-de-artigos → escriba.
- **Tarefa pequena:** direto, roda `pytest`.

## Regras para todos
1. Comparação justa: mesmo algoritmo, mesmo orçamento, várias sementes, nº de parâmetros reportado (skill `comparar-quantico-classico`).
2. Nada de "vantagem quântica" sem evidência aprovada pelo `fisico-revisor`.
3. Referências: nunca inventar DOI/volume/página; ◐ e `checknote = {...}` (campo que o BibTeX ignora; não aparece no PDF) quando não confirmado.
4. Mudou observação/ações/recompensa de um jogo? Resultados antigos em `data/` ficam inválidos: avisar.
5. Todo commit entra em `ATUALIZACOES.md` (escriba, skill `registrar-atualizacao`).
