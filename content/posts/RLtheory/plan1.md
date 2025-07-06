---
date: "2025-07-06"
title: "(higher level) Tentative Plan for RL Theory Series"
summary: "(higher level) Tentative Plan for RL Theory Series"
category: Plan
series: ["RL Theory"]
author: "Author: Bryan Chan"
hero: /assets/images/hero3.png
image: /assets/images/card3.png
---

**Part I: Foundations**

* **Chapter 1: The Reinforcement Learning Problem: MDPs & Dynamic Programming**
    * **1.1.** Formalism: MDPs, Policies, Value Functions, and Trajectories
    * **1.2.** Optimality: Bellman Operators, Contraction Properties, and the Fundamental Theorem
    * **1.3.** Exact Solutions: Value Iteration, Policy Iteration, and their Convergence Analysis
    * **1.4.** The Linear Programming Formulation of MDPs

* **Chapter 2: Core Tools: Function Approximation & Statistical Analysis**
    * **2.1.** Function Approximation Architectures (Linear, Kernel, Neural)
    * **2.2.** The "Deadly Triad" and Divergence Pathologies
    * **2.3.** Statistical Toolkit: Concentration Inequalities, Uniform Convergence, and Martingales
    * **2.4.** Error Decomposition: Approximation, Estimation, and Propagation

**Part II: Core Algorithmic Paradigms**

* **Chapter 3: Model-Free Learning: Prediction and Control**
    * **3.1.** On-Policy Value Prediction (MC, TD($\lambda$), LSTD)
    * **3.2.** On-Policy Control: Sarsa, Expected Sarsa, and GLIE
    * **3.3.** Off-Policy Value Prediction & Control (Q-Learning, Double-Q, Retrace, V-trace)
    * **3.4.** Policy Gradient Methods (REINFORCE, Actor-Critic, GAE)
    * **3.5.** Advanced Policy Gradients (TRPO, PPO, SAC, DDPG)
    * **3.6.** Distributional and Risk-Sensitive RL
    * **3.7.** Exploration Strategies (Count-based, Optimism, Intrinsic Motivation)

* **Chapter 4: Model-Based Learning: Planning, Imagination, and Control**
    * **4.1.** Planning with a Known Model (Online Planning, MCTS)
    * **4.2.** Learning World Models (Parametric, Ensembles, Latent-State Models)
    * **4.3.** Planning in Learned Models (Dreamer, MuZero)
    * **4.4.** Model-Based Exploration and Uncertainty Quantification
    * **4.5.** Hybrid Model-Based/Model-Free Architectures

* **Chapter 5: Offline (Batch) Reinforcement Learning**
    * **5.1.** Problem Formulation: The Challenge of Distribution Shift
    * **5.2.** Offline Policy Evaluation (OPE): IS, DR, FQE, MAGIC
    * **5.3.** Offline Control via Policy Constraint (CQL, IQL, AWR)
    * **5.4.** Offline Control via Model-Based Pessimism (MOPO, COMBO)
    * **5.5.** Theoretical Guarantees and Concentrability
    * **5.6.** Benchmarks and Evaluation Protocols (D4RL)

**Part III: Advanced Topics**

* **Chapter 6: Decision-Making Under Uncertainty: POMDPs & Bayesian RL**
    * **6.1.** The POMDP Framework and Belief-State MDPs
    * **6.2.** Planning in Belief Space (Point-Based Methods, POMCP)
    * **6.3.** Model-Free RL with Memory (Recurrent and Transformer Agents)
    * **6.4.** Bayesian RL: The Bayes-Adaptive MDP
    * **6.5.** Posterior Sampling for Exploration (PSRL)
    * **6.6.** Approximate Inference for Deep BRL (Ensembles, Variational Methods)

* **Chapter 7: Imitation, Inverse, and Preference-Based Learning**
    * **7.1.** Behavioural Cloning and Its Limitations (DAgger)
    * **7.2.** Inverse Reinforcement Learning (MaxEnt, Adversarial)
    * **7.3.** Learning from Preferences: The RLHF Pipeline
    * **7.4.** Offline Imitation Learning (IQ-Learn)
    * **7.5.** Foundation Models for Control (Decision/Diffusion Transformers)

* **Chapter 8: Abstraction: Hierarchy, State, and Temporal Structure**
    * **8.1.** State Abstraction and Bisimulation
    * **8.2.** Temporal Abstraction: The Options Framework
    * **8.3.** Hierarchical Architectures and Skill Discovery (Option-Critic, DIAYN)
    * **8.4.** Theoretical Benefits of Abstraction
    * **8.5.** HRL with Learned World Models

* **Chapter 9: Multi-Agent Reinforcement Learning**
    * **9.1.** Foundations: Stochastic Games and Solution Concepts
    * **9.2.** Centralised Training with Decentralised Execution (QMIX, MADDPG)
    * **9.3.** Communication and Coordination
    * **9.4.** Offline and Hierarchical MARL
    * **9.5.** Population-Based Methods and Game Theory Links (PSRO, CFR)

**Part IV: Cross-Cutting Challenges & Future Directions**

* **Chapter 10: Learning to Learn: Transfer, Meta, and Continual RL**
    * **10.1.** Quantifying Task Similarity and Transfer
    * **10.2.** Meta-RL: Learning Fast Adaptation (MAML, PEARL, RL²)
    * **10.3.** Continual RL: Overcoming Catastrophic Forgetting (EWC, Replay, PackNet)
    * **10.4.** Open-World and Lifelong Learning
    * **10.5.** Theoretical Guarantees for Generalization

* **Chapter 11: Safe and Robust Reinforcement Learning**
    * **11.1.** Formalisms for Safety and Robustness (Constrained MDPs, Robust MDPs)
    * **11.2.** Safe Exploration (Shielding, Lagrangian Methods, Conservative Baselines)
    * **11.3.** Robust Planning and Control (Distributionally Robust DP, Tube MPC)
    * **11.4.** Verification and Runtime Assurance
    * **11.5.** Safe Offline and Imitation Learning

* **Chapter 12: Synthesis and Open Problems**
    * **12.1.** A Unified View of RL Paradigms
    * **12.2.** Grand Challenges: Sample Complexity, Generalization, and Scalability
    * **12.3.** Computational vs. Statistical Limits in Deep RL
    * **12.4.** The Future of Foundation Models in Decision-Making
