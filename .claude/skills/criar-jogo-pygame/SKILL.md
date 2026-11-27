---
name: criar-jogo-pygame
description: Receita para criar ou terminar um jogo simples em pygame (Pong, Snake) que serve ao mesmo tempo para humano jogar e como ambiente de aprendizado por reforço (reset/step). Use ao criar jogo novo em games/, corrigir física/controles ou mudar observação, ações e recompensa.
---
# Criar jogo pygame (jogável + ambiente de RL)

## Estrutura de `games/<jogo>/`
```
games/<jogo>/
├── env.py      lógica pura: sem pygame, determinística com seed
├── play.py     janela pygame: humano x CPU (ou agente), desenho, sons opcionais
└── assets/     imagens do jogo (sempre caminho relativo ao arquivo)
tests/test_<jogo>_env.py
```

## Regras do `env.py`
1. Classe `<Jogo>Env` com `reset(seed=None) -> obs` e `step(action) -> (obs, reward, done, info)` (padrão Gymnasium simplificado).
2. Ações discretas pequenas e documentadas (Pong: 0 parado, 1 cima, 2 baixo; Snake: 0 cima, 1 baixo, 2 esquerda, 3 direita).
3. Observação = vetor curto normalizado em [-1, 1] (bom para codificar em ângulos de qubits). Documente cada posição.
4. Recompensa simples e documentada (+1 ponto feito, -1 ponto sofrido; extras pequenos, ex.: +0.1 ao rebater).
5. `random.Random(seed)` próprio: nada de `random` global. Mesmo seed + mesmas ações = mesmo jogo.
6. Física por passo fixo (1 passo = 1 frame a 60 FPS). Evite colisão "grudada": ao rebater, reposicione fora da raquete e use `abs()` na velocidade.
7. `max_steps` para o episódio sempre terminar.

## Regras do `play.py`
- Teclas: setas **e** W/S (ou WASD), `P` pausa, `R` reinicia, `Esc` sai. Mostre os controles na tela.
- Movimento contínuo com `pygame.key.get_pressed()` (não "pulos" em KEYDOWN).
- Desenha a partir do estado do `env` (o `play.py` não tem regra de jogo própria).
- `--agente <nome>` opcional para ver um agente treinado jogar.
- Funciona sem tela para teste: `SDL_VIDEODRIVER=dummy python -m games.<jogo>.play --frames 120`.

## Pronto quando
- `python -m pytest -q` verde (testes: reset, ações mexem a raquete/cobra, colisão com parede, ponto marca placar, determinismo por seed, episódio termina).
- Teste sem tela roda sem erro.
- README com como jogar e a tabela observação/ações/recompensa.
