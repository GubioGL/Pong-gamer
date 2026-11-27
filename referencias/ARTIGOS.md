# Referências: algoritmos quânticos aplicados a jogos

Lista de fontes para o projeto: agentes quânticos variacionais, de **qubits** e de **variáveis contínuas**, comparados com redes neurais clássicas em Pong e Snake.
Cada item tem a chave BibTeX usada em [`paper/references.bib`](../paper/references.bib).

**Status:** ✔ = conferido em busca web em 2026-11-27 (título, autores, local); ◐ = da memória, conferir antes de citar.
**Prioridade de leitura:** ⭐⭐⭐ ler primeiro · ⭐⭐ importante · ⭐ contexto.

Como adicionar uma fonte: use a skill `adicionar-referencia` (`.claude/skills/adicionar-referencia/`).

---

## A. Agentes quânticos em jogos (Pong / Atari): o núcleo do projeto

| ⭐ | Chave | Referência | O que interessa para nós | Status |
|---|---|---|---|---|
| ⭐⭐⭐ | `lockwood2021atari` | O. Lockwood, M. Si. *Playing Atari with Hybrid Quantum-Classical Reinforcement Learning*. arXiv:2107.04114 (2021). [arXiv](https://arxiv.org/abs/2107.04114) | Rede clássica codifica os pixels e alimenta um circuito variacional dentro de Double DQN. **Não** conseguiu aprender Pong e Breakout. Mostra o problema de entrada de alta dimensão. | ✔ |
| ⭐⭐⭐ | `freinberger2024atari` | D. Freinberger, J. Lemmel, R. Grosu, S. Jerbi. *A quantum-classical reinforcement learning model to play Atari games*. arXiv:2412.08725 (2024). [arXiv](https://arxiv.org/abs/2412.08725) | Modelo híbrido (codificação clássica + PQC + pós-processamento) **resolve Pong** e chega a ~84% do clássico em Breakout. Sem vantagem quântica. Referência direta para o nosso Pong. | ✔ |
| ⭐⭐⭐ | `wang2026entanglement` | P. Wang, K. Hymas, J. Quach. *Quantum entanglement provides a competitive advantage in adversarial games*. arXiv:2603.10289 (2026). [arXiv](https://arxiv.org/abs/2603.10289) | Pong competitivo; PQC de 8 qubits como extrator de features; compara circuito separável, CZ fixo, IsingZZ treinável e MLP clássico (10 sementes). Emaranhamento ajuda contra separável com o mesmo nº de parâmetros. Preprint, sem revisão por pares. | ✔ |

## B. Aprendizado por reforço quântico variacional em ambientes simples

| ⭐ | Chave | Referência | O que interessa | Status |
|---|---|---|---|---|
| ⭐⭐⭐ | `chen2020vqc` | S. Y.-C. Chen, C.-H. H. Yang, J. Qi, P.-Y. Chen, X. Ma, H.-S. Goan. *Variational Quantum Circuits for Deep Reinforcement Learning*. IEEE Access 8, 141007–141024 (2020). doi:[10.1109/ACCESS.2020.3010470](https://doi.org/10.1109/ACCESS.2020.3010470) · arXiv:1907.00397 · [código](https://github.com/ycchen1989/var-quantumcircuits-deeprl) | Primeiro DQN com circuito variacional (FrozenLake, CognitiveRadio). Experience replay + target network com VQC. | ✔ |
| ⭐⭐⭐ | `skolik2022gym` | A. Skolik, S. Jerbi, V. Dunjko. *Quantum agents in the Gym: a variational quantum algorithm for deep Q-learning*. Quantum 6, 720 (2022). doi:[10.22331/q-2022-05-24-720](https://doi.org/10.22331/q-2022-05-24-720) · arXiv:2103.15084 | Q-learning com PQC em CartPole e FrozenLake; importância de re-uploading e de pesos treináveis na entrada e na saída. Comparável ao clássico com menos parâmetros, mas hiperparâmetros importam mais que nº de parâmetros. **Receita base para nosso agente.** | ✔ |
| ⭐⭐⭐ | `jerbi2021pqp` | S. Jerbi, C. Gyurik, S. C. Marshall, H. J. Briegel, V. Dunjko. *Parametrized Quantum Policies for Reinforcement Learning*. NeurIPS 34 (2021). arXiv:2103.05577 | Política (REINFORCE) com PQC e poucos qubits em benchmarks padrão; prova de separação quântico/clássico (log discreto). Alternativa policy-gradient ao DQN. | ✔ |
| ⭐⭐ | `lockwood2020rlqvc` | O. Lockwood, M. Si. *Reinforcement Learning with Quantum Variational Circuits*. Proc. AIIDE 16(1), 245–251 (2020). arXiv:2008.07524 | CartPole e Blackjack; compara com redes de 58, 226 e 1282 pesos. Mostra como escolher baselines clássicos por nº de parâmetros. | ✔ |
| ⭐⭐ | `sequeira2023pg` | A. Sequeira, L. P. Santos, L. S. Barbosa. *Policy Gradients using Variational Quantum Circuits*. arXiv:2203.10591 (2022). | Policy gradient com VQC; análise de gradiente/treinabilidade. | ✔ (título/arXiv) |
| ⭐⭐ | `chen2023qrnn` | S. Y.-C. Chen. *Quantum deep recurrent reinforcement learning*. arXiv:2210.14876 (2022). | QLSTM para ambientes parcialmente observáveis; útil se a observação do Snake for parcial. | ✔ (título/arXiv) |
| ⭐⭐ | `chen2022evo` | S. Y.-C. Chen et al. *Variational quantum reinforcement learning via evolutionary optimization*. arXiv:2109.00540 (2021). | MiniGrid (mundo em grade): o mais parecido estruturalmente com Snake. Otimização sem gradiente. | ✔ (arXiv) |
| ⭐⭐ | `maze2023hqnn` | *Deep-Q Learning with Hybrid Quantum Neural Network on Solving Maze Problems*. arXiv:2304.10159 (2023). | Labirinto em grade com DQN híbrido; referência para Snake. | ✔ (título/arXiv) |
| ⭐ | `luo2025hdqnn` | Luo et al. *Training Hybrid Deep Quantum Neural Network for Efficient Reinforcement Learning*. arXiv:2503.09119 (2025). | Redes quânticas profundas híbridas; controle contínuo (não jogos). | ✔ (arXiv) |
| ⭐⭐⭐ | `nagy2021photonic` | D. Nagy et al. *Photonic Quantum Policy Learning in OpenAI Gym*. arXiv:2108.12926 (2021). | Agente de **variáveis contínuas (fotônico)** com PPO em CartPole: comparável ao clássico com o mesmo nº de parâmetros, convergindo mais rápido. Referência direta para o nosso agente CV. | ✔ (título/arXiv; autores ◐) |
| ⭐⭐ | `nagy2024latent` | *Hybrid Quantum-Classical Reinforcement Learning in Latent Observation Spaces*. arXiv:2410.18284 (2024). | Autoencoder clássico + agente quântico, com versões **fotônica (CV)** e de **qubits**: o mesmo par de algoritmos que comparamos. | ✔ (título/arXiv; autores ◐) |
| ⭐ | `kwak2021intro` | Y. Kwak et al. *Introduction to Quantum Reinforcement Learning: Theory and PennyLane-based Implementation*. ICTC 2021. arXiv:2108.06849 | Tutorial com código PennyLane: bom ponto de partida de implementação. | ◐ |

## C. Comparações quântico vs. clássico (como fazer uma comparação justa)

| ⭐ | Chave | Referência | O que interessa | Status |
|---|---|---|---|---|
| ⭐⭐⭐ | `franz2022instab` | M. Franz et al. *Uncovering Instabilities in Variational-Quantum Deep Q-Networks*. arXiv:2202.05195 (2022). | Instabilidade do VQ-DQN entre sementes; baseline clássico limitado a 58 parâmetros. **Mostra por que precisamos de várias sementes.** | ✔ |
| ⭐⭐ | `oslomet2023compare` | *A Comparative Study of Classical and Quantum Deep Q-Learning Agents in OpenAI Gym*. OsloMet (2023). [link](https://oda.oslomet.no/oda-xmlui/handle/11250/3101183) | Desempenho comparável com bem menos parâmetros treináveis. | ✔ (link) |
| ⭐⭐⭐ | `meyer2022survey` | N. Meyer, C. Ufrecht, M. Periyasamy, D. D. Scherer, A. Plinge, C. Mutschler. *A Survey on Quantum Reinforcement Learning*. arXiv:2211.03464 (2022, v2 2024). | Revisão do campo; conclui que ainda não há vantagem quântica robusta em NISQ. Mapa de toda a literatura. | ✔ |

## D. Fundamentos: circuitos variacionais / redes neurais quânticas

| ⭐ | Chave | Referência | O que interessa | Status |
|---|---|---|---|---|
| ⭐⭐ | `cerezo2021vqa` | M. Cerezo et al. *Variational quantum algorithms*. Nat. Rev. Phys. 3, 625–644 (2021). doi:10.1038/s42254-021-00348-9 | Revisão de VQAs (ansatz, otimizadores, treinabilidade). | ◐ |
| ⭐⭐⭐ | `killoran2019cvqnn` | N. Killoran, T. R. Bromley, J. M. Arrazola, M. Schuld, N. Quesada, S. Lloyd. *Continuous-variable quantum neural networks*. Phys. Rev. Research 1, 033063 (2019). arXiv:1806.06871 | Rede neural quântica de variáveis contínuas: portas gaussianas = parte afim, porta não gaussiana (Kerr) = ativação. Base do nosso `CVQNet` (versão gaussiana). | ✔ |
| ⭐⭐⭐ | `delima2026regression` | G. G. de Lima, T. de S. Farias, A. C. Ricardo, C. J. Villas-Bôas. *Assessing the Advantages and Limitations of Quantum Neural Networks in Regression Tasks*. Quantum Machine Intelligence 8, 20 (2026). arXiv:2509.00854. [código](https://github.com/GubioGL/Quantum-Neural-Networks-in-Regression-Tasks) | **Trabalho do autor.** QNN CV de 1 modo (camada R S R D K, entrada D(x)\|0⟩, saída ⟨a + a†⟩): base do nosso agente `cv` e da `libs/cvjax`. | ✔ (busca; nº do artigo ◐) |
| ⭐⭐ | `weedbrook2012gaussian` | C. Weedbrook et al. *Gaussian quantum information*. Rev. Mod. Phys. 84, 621 (2012). | Formalismo de média e covariância usado no nosso simulador gaussiano. | ◐ |
| ⭐⭐ | `mcclean2018barren` | J. R. McClean et al. *Barren plateaus in quantum neural network training landscapes*. Nat. Commun. 9, 4812 (2018). doi:10.1038/s41467-018-07090-4 | Gradientes que somem: limita o nº de qubits/camadas. | ◐ |
| ⭐⭐⭐ | `perezsalinas2020reuploading` | A. Pérez-Salinas et al. *Data re-uploading for a universal quantum classifier*. Quantum 4, 226 (2020). doi:10.22331/q-2020-02-06-226 | Re-uploading: como codificar o estado do jogo em poucos qubits. | ◐ |
| ⭐⭐ | `schuld2021encoding` | M. Schuld, R. Sweke, J. J. Meyer. *Effect of data encoding on the expressive power of variational quantum machine-learning models*. Phys. Rev. A 103, 032430 (2021). doi:10.1103/PhysRevA.103.032430 | Por que escalar a entrada / repetir a codificação. | ◐ |
| ⭐ | `dong2008qrl` | D. Dong, C. Chen, H. Li, T.-J. Tarn. *Quantum Reinforcement Learning*. IEEE Trans. SMC-B 38(5), 1207–1220 (2008). | QRL "clássico" (pré-variacional), contexto histórico. | ◐ |
| ⭐ | `dunjko2016qeml` | V. Dunjko, J. M. Taylor, H. J. Briegel. *Quantum-Enhanced Machine Learning*. Phys. Rev. Lett. 117, 130501 (2016). | Vantagem quântica em agentes (teórico). | ◐ |
| ⭐ | `saggio2021speedup` | V. Saggio et al. *Experimental quantum speed-up in reinforcement learning agents*. Nature 591, 229–233 (2021). | Experimento fotônico com aceleração no aprendizado de agentes. | ◐ |

## E. Baselines clássicos (redes neurais em jogos)

| ⭐ | Chave | Referência | O que interessa | Status |
|---|---|---|---|---|
| ⭐⭐⭐ | `mnih2015dqn` | V. Mnih et al. *Human-level control through deep reinforcement learning*. Nature 518, 529–533 (2015). doi:10.1038/nature14236 | DQN (replay + target network) em Atari, incluindo Pong. Algoritmo base para os dois agentes. | ◐ |
| ⭐⭐ | `mnih2013atari` | V. Mnih et al. *Playing Atari with Deep Reinforcement Learning*. arXiv:1312.5602 (2013). | Versão original do DQN. | ◐ |
| ⭐⭐⭐ | `wei2018snake` | Z. Wei, D. Wang, M. Zhang, A.-H. Tan, C. Miao, Y. Zhou. *Autonomous Agents in Snake Game via Deep Reinforcement Learning*. IEEE ICA 2018, pp. 20–25. [código](https://github.com/weizhepei/DRL-SnakeGame) | Baseline clássico de Snake (CNN + Q-learning, grade 12×12). | ✔ |
| ⭐ | `towers2024gymnasium` | M. Towers et al. *Gymnasium: A Standard Interface for Reinforcement Learning Environments*. arXiv:2407.17032 (2024). | Nossa interface `reset()/step()` segue o padrão Gymnasium. | ◐ |

## F. "Jogos quânticos" (contexto; tema diferente do nosso)

Estes tratam de jogos *sobre* física quântica ou de teoria dos jogos quântica, não de agentes quânticos que jogam. Úteis na introdução para separar os conceitos.

| ⭐ | Chave | Referência | Status |
|---|---|---|---|
| ⭐ | `piispanen2023history` | L. Piispanen, E. Morrell, S. Park, M. Pfaffhauser, A. Kultima. *The History of Quantum Games*. IEEE CoG 2023. arXiv:2309.01525 (mais de 260 jogos catalogados). | ✔ |
| ⭐ | `piispanen2022defining` | L. Piispanen, M. Pfaffhauser, J. Wootton, A. Kultima. *Defining Quantum Games*. arXiv:2206.00089. | ✔ |
| ⭐ | `meyer1999strategies` | D. A. Meyer. *Quantum Strategies*. Phys. Rev. Lett. 82, 1052 (1999). | ◐ |
| ⭐ | `eisert1999games` | J. Eisert, M. Wilkens, M. Lewenstein. *Quantum Games and Quantum Strategies*. Phys. Rev. Lett. 83, 3077 (1999). | ◐ |

## G. Ferramentas e tutoriais

- `bergholm2018pennylane`: V. Bergholm et al. *PennyLane: Automatic differentiation of hybrid quantum-classical computations*. arXiv:1811.04968. Biblioteca sugerida (integra com PyTorch). ◐
- TensorFlow Quantum, tutorial *Parametrized quantum circuits for reinforcement learning* (reproduz Jerbi 2021 e Skolik 2022): <https://www.tensorflow.org/quantum/tutorials/quantum_reinforcement_learning> ◐
- Código de Chen et al. 2020: <https://github.com/ycchen1989/var-quantumcircuits-deeprl> ✔
- `killoran2019sf`: N. Killoran et al. *Strawberry Fields: A Software Platform for Photonic Quantum Computing*. Quantum 3, 129 (2019). Backend `fock` (inspiração da `libs/cvjax`). ◐
- `gupt2019walrus`: B. Gupt, J. Izaac, N. Quesada. *The Walrus*. JOSS 4(44), 1705 (2019). Elementos de matriz de D, S e BS; referência numérica dos testes da `cvjax`. ◐
- `pennylane_cvqnn_demo`: demo do PennyLane *Function fitting with a photonic quantum neural network*: <https://pennylane.ai/demos/quantum_neural_net>; template [`CVNeuralNetLayers`](https://docs.pennylane.ai/en/stable/code/api/pennylane.CVNeuralNetLayers.html). ✔ (links do autor)
- CV-QuForge (Tiago de S. Farias, `tiago939`, MIT) e QuIara (Tiago de S. Farias e Alexandre C. Ricardo, Apache-2.0): simuladores CV em PyTorch que inspiraram a `cvjax` (cópias enviadas pelo autor; sem link público).
- `jax2018github`: JAX. `vanhasselt2016double`: H. van Hasselt, A. Guez, D. Silver. *Deep Reinforcement Learning with Double Q-Learning*. AAAI 2016 (o DQN que usamos). ◐

---

## Lacunas encontradas (oportunidade para o artigo)

1. **Snake:** não achei nenhum trabalho com agente quântico variacional no Snake. O mais próximo é mundo em grade (MiniGrid, labirinto). Seria uma contribuição original.
2. **Pong com observação de baixa dimensão:** os trabalhos de Pong usam pixels (Atari), e a parte clássica faz a maior parte do trabalho. Um Pong com estado pequeno (posição/velocidade da bola e raquetes, 3 ações) permite um agente *majoritariamente quântico* e uma comparação limpa com MLP de mesmo nº de parâmetros.
3. **Qubits x variáveis contínuas no mesmo jogo:** só achei a comparação em CartPole/espaço latente (`nagy2024latent`); não encontrei trabalho comparando os dois tipos de agente quântico e uma rede clássica em Pong e Snake.
4. **Comparação justa:** os resultados são mistos e dependem muito do baseline e das sementes (`franz2022instab`, `skolik2022gym`). Precisamos de várias sementes, mesmo algoritmo (DQN) e de reportar parâmetros, tempo de treino e variância.
