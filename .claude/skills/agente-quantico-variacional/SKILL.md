---
name: agente-quantico-variacional
description: Receita de Q-network com circuito variacional quântico (PennyLane + PyTorch) para jogos com observação pequena e 3-4 ações, baseada em Skolik et al. 2022 e Jerbi et al. 2021. Use ao implementar ou alterar o agente quântico.
---
# Agente quântico variacional (VQC-DQN)

Referências: `skolik2022gym`, `jerbi2021pqp`, `chen2020vqc`, `perezsalinas2020reuploading` (ver `referencias/ARTIGOS.md`).

## Arquitetura padrão
- **Qubits:** n = tamanho da observação (Pong: 6; se for grande, reduza com features, não com rede clássica grande).
- **Codificação:** `RX(λ_i · x_i)` por qubit, com λ treinável (escala da entrada).
- **Camada variacional:** `RY(θ)`, `RZ(θ)` por qubit + anel de `CZ` (emaranhamento).
- **Re-uploading:** repete [codificação → camada variacional] L vezes (começar com L = 3–5).
- **Saída:** Q(a) = w_a · ⟨Z_a⟩ (+ bias), um observável por ação, w treinável (o valor esperado fica em [-1, 1] e o retorno não).
- Simulador: `default.qubit` com `diff_method="backprop"` (rápido em CPU para ≤ 10 qubits); `interface="torch"`.

## Treino
- DQN com replay buffer e target network, **idêntico** ao do MLP (mesmo `src/rl/dqn.py`).
- Learning rates separados: pesos de saída w maior (~0.1); escalas de entrada λ e ângulos θ menores (~0.001). Valores do tutorial TFQ de Skolik et al.: conferir.
- Conte parâmetros com `sum(p.numel() for p in model.parameters())` e registre.

## Cuidados
- Barren plateaus (`mcclean2018barren`): poucas camadas, inicialização pequena.
- Sem afirmar vantagem quântica: compare pelo protocolo da skill `comparar-quantico-classico`.
