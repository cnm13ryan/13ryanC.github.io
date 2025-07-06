---
date: "2025-07-06"
title: "(Lower-level) Tentative Plan for RL Theory Series"
summary: "(Lower-level) Tentative Plan for RL Theory Series"
category: Plan
series: ["RL Theory"]
author: "Author: Bryan Chan"
hero: /assets/images/hero3.png
image: /assets/images/card3.png
---

### **Comprehensive Reinforcement Learning: A Unified View**

---

### **Part I: Foundations**

**Chapter 1: The Reinforcement Learning Problem: MDPs & Dynamic Programming**
* **1.1. Formalism: MDPs, Policies, Value Functions, and Trajectories**
    * 1.1.1. Mathematical Foundations: Measure-theoretic set-up
    * 1.1.2. Formal Definition of an MDP: $(S, A, P, r, \gamma)$
    * 1.1.3. Policies and the Agent–Environment Loop
    * 1.1.4. Trajectories, Returns, and Value Functions
    * 1.1.5. The Discount Factor $\gamma$ & Effective Horizon
* **1.2. Optimality: Bellman Operators, Contraction Properties, and the Fundamental Theorem**
    * 1.2.1. Optimality Notions and Objective Functions
    * 1.2.2. Bellman Operators: $T^\pi, T^*$
    * 1.2.3. Key Analytical Properties (Monotonicity, Contraction, Error Bounds)
    * 1.2.4. Greedy Policies and the Fundamental Theorem of Dynamic Programming
* **1.3. Exact Solutions: Value Iteration, Policy Iteration, and their Convergence Analysis**
    * 1.3.1. Value Iteration (VI): Algorithm, Convergence, and $\varepsilon$-Stopping
    * 1.3.2. Policy Iteration (PI): Algorithm, Performance-Difference Identity, and Geometric Convergence
    * 1.3.3. Comparison and Runtime Bounds for VI and PI
* **1.4. The Linear Programming Formulation of MDPs**
    * 1.4.1. Primal LP Formulation (Minimising Value)
    * 1.4.2. Dual LP Formulation (Occupancy Measures)
    * 1.4.3. Complementary Slackness and Connections to Policy Iteration

**Chapter 2: Core Tools: Function Approximation & Statistical Analysis**
* **2.1. Function Approximation Architectures (Linear, Kernel, Neural)**
    * 2.1.1. Function Classes: Tabular, Linear, Kernel/RKHS, Neural Networks
    * 2.1.2. Realizability and the Limits of Approximation
* **2.2. The "Deadly Triad" and Divergence Pathologies**
    * 2.2.1. Understanding Divergence: Off-policy learning, function approximation, and bootstrapping
    * 2.2.2. Baird's Star Counterexample and Analysis
* **2.3. Statistical Toolkit: Concentration Inequalities, Uniform Convergence, and Martingales**
    * 2.3.1. Concentration of Measure: Hoeffding, Bernstein, Azuma–Hoeffding (Martingales)
    * 2.3.2. Self-Normalised Bounds for Adaptive Data (Elliptical Potential Lemma)
    * 2.3.3. Uniform Convergence: Covering Numbers, Rademacher Complexity, and Chaining
* **2.4. Error Decomposition: Structural, Data‑Driven, Algorithmic, and Propagated Errors**
    * 2.4.1. The Core Decomposition: Bias-Variance and Approximation-Estimation Trade-offs
    * 2.4.2  Algorithmic & Numerical Errors: Optimisation Limits and Hardware Precision
    * 2.4.3  Bellman Error Propagation and Stability Analysis

---

### **Part II: Core Algorithmic Paradigms**

**Chapter 3: Model-Free Learning: Prediction and Control**
* **3.1. On-Policy Value Prediction (MC, TD($\lambda$), LSTD)**
    * 3.1.1. Monte-Carlo (MC) Prediction: First-visit, Every-visit, and Incremental Updates
    * 3.1.2. Temporal-Difference (TD) Learning: TD(0), Bias-Variance Trade-offs
    * 3.1.3. Multi-Step TD and the $\lambda$-Return: TD($\lambda$) and Eligibility Traces (GAE)
    * 3.1.4. Least-Squares Temporal Difference (LSTD) and Batch Methods
* **3.2. On-Policy Control: Sarsa, Expected Sarsa, and GLIE**
    * 3.2.1. Sarsa and Expected Sarsa
    * 3.2.2. Sarsa($\lambda$) and True-Online Sarsa($\lambda$)
    * 3.2.3. GLIE (Greedy in the Limit with Infinite Exploration) Convergence
* **3.3. Off-Policy Value Prediction & Control (Q-Learning, Double-Q, Retrace, V-trace)**
    * 3.3.1. Q-Learning: Maximisation Operator and Convergence
    * 3.3.2. Double Q-Learning: Mitigating Maximisation Bias
    * 3.3.3. Importance Sampling for Off-Policy Correction
    * 3.3.4. Safe and Efficient Off-Policy TD: Retrace($\lambda$), V-trace, and Emphatic TD (ETD)
    * 3.3.5. Gradient-TD Methods for Stability (GTD, TDC)
* **3.4. Policy Gradient Methods (REINFORCE, Actor-Critic)**
    * 3.4.1. The Policy Gradient Theorem
    * 3.4.2. REINFORCE: Monte-Carlo Policy Gradient
    * 3.4.3. Actor-Critic (A2C/A3C): Bootstrapping and Variance Reduction
    * 3.4.4. Generalised Advantage Estimation (GAE)
* **3.5. Advanced Policy Gradients (TRPO, PPO, SAC, DDPG)**
    * 3.5.1. Natural Policy Gradient and Trust Region Policy Optimisation (TRPO)
    * 3.5.2. Proximal Policy Optimisation (PPO)
    * 3.5.3. Off-Policy Deterministic Policy Gradient (DPG, DDPG, TD3)
    * 3.5.4. Maximum Entropy RL and Soft Actor-Critic (SAC)
* **3.6. Distributional and Risk-Sensitive RL**
    * 3.6.1. Distributional Bellman Operator (C51, QR-DQN)
    * 3.6.2. Risk-Sensitive Objectives (CVaR, Coherent Risk Measures)
* **3.7. Exploration Strategies (Count-based, Optimism, Intrinsic Motivation)**
    * 3.7.1. Optimism in the Face of Uncertainty (UCB)
    * 3.7.2. Count-Based and Pseudo-Count Exploration
    * 3.7.3. Intrinsic Motivation: Curiosity (ICM), Random Network Distillation (RND)

**Chapter 4: Model-Based Learning: Planning, Imagination, and Control**
* **4.1. Planning with a Known Model (Online Planning, MCTS)**
    * 4.1.1. The Online Planning Problem and Simulator Access
    * 4.1.2. Sparse Sampling and Upper/Lower Runtime Bounds
    * 4.1.3. Monte-Carlo Tree Search (MCTS)
* **4.2. Learning World Models (Parametric, Ensembles, Latent-State Models)**
    * 4.2.1. Model-Bias Bounds and Ensemble Methods
    * 4.2.2. Latent State-Space Models (RSSM, VRNN)
* **4.3. Planning in Learned Models (Dreamer, MuZero)**
    * 4.3.1. Trajectory Optimisation and Model-Predictive Control (MPC)
    * 4.3.2. Planning in Latent Space (Dreamer Family)
    * 4.3.3. Integrating Planning and Learning (MuZero)
* **4.4. Model-Based Exploration and Uncertainty Quantification**
    * 4.4.1. Optimism Under Model Uncertainty
    * 4.4.2. Thompson Sampling with Learned Models
* **4.5. Hybrid Model-Based/Model-Free Architectures**
    * 4.5.1. Dyna: Integrating Planning, Acting, and Learning
    * 4.5.2. Imagination-Augmented Agents

**Chapter 5: Offline (Batch) Reinforcement Learning**
* **5.1. Problem Formulation: The Challenge of Distribution Shift**
    * 5.1.1. The Offline Setting: Static Datasets and Behaviour Policies
    * 5.1.2. Extrapolation Error and the Perils of Distribution Mismatch
* **5.2. Offline Policy Evaluation (OPE): IS, DR, FQE, MAGIC**
    * 5.2.1. Importance Sampling (IS) Methods (WIS, PDIS)
    * 5.2.2. Doubly Robust (DR) Estimators
    * 5.2.3. Model-Based Evaluation: Fitted Q-Evaluation (FQE)
    * 5.2.4. Advanced Estimators (MAGIC, OPERA)
* **5.3. Offline Control via Policy Constraint (CQL, IQL, AWR)**
    * 5.3.1. Behaviour Regularisation and Cloning (BC)
    * 5.3.2. Conservative Q-Learning (CQL)
    * 5.3.3. Implicit Q-Learning (IQL)
* **5.4. Offline Control via Model-Based Pessimism (MOPO, COMBO)**
    * 5.4.1. Pessimistic Value Functions with Model Uncertainty
    * 5.4.2. Algorithms: MOPO, COMBO
* **5.5. Theoretical Guarantees and Concentrability**
    * 5.5.1. Concentrability Coefficients and Coverage Assumptions
    * 5.5.2. Minimax Rates and Computational Barriers
* **5.6. Benchmarks and Evaluation Protocols (D4RL)**
    * 5.6.1. The D4RL Benchmark Suite
    * 5.6.2. Practical Considerations and Pitfalls

---

### **Part III: Advanced Topics**

**Chapter 6: Decision-Making Under Uncertainty: POMDPs & Bayesian RL**
* **6.1. The POMDP Framework and Belief-State MDPs**
    * 6.1.1. Formal Definition of a POMDP
    * 6.1.2. Information States and the Belief-MDP Transformation
    * 6.1.3. Value Functions and Computational Complexity
* **6.2. Planning in Belief Space (Point-Based Methods, POMCP)**
    * 6.2.1. Exact Planning: Value Iteration with Alpha-Vectors
    * 6.2.2. Offline Point-Based Methods (PBVI, SARSOP)
    * 6.2.3. Online Tree Search (POMCP, DESPOT)
* **6.3. Model-Free RL with Memory (Recurrent and Transformer Agents)**
    * 6.3.1. Recurrent Architectures for Partial Observability (DRQN, R2D2)
    * 6.3.2. Transformer-Based Agents (GTrXL)
* **6.4. Bayesian RL: The Bayes-Adaptive MDP**
    * 6.4.1. Priors over MDPs and Posterior Updates
    * 6.4.2. The Bayes-Adaptive MDP (BAMDP) Formulation
* **6.5. Posterior Sampling for Exploration (PSRL)**
    * 6.5.1. Thompson Sampling for RL (PSRL)
    * 6.5.2. Bayesian Regret Bounds
* **6.6. Approximate Inference for Deep BRL (Ensembles, Variational Methods)**
    * 6.6.1. Ensembles and Bootstrapped Methods
    * 6.6.2. Variational Inference for BRL

**Chapter 7: Imitation, Inverse, and Preference-Based Learning**
* **7.1. Behavioural Cloning and Its Limitations (DAgger)**
    * 7.1.1. Supervised Learning for Control and Compounding Errors
    * 7.1.2. Interactive Imitation: Dataset Aggregation (DAgger)
* **7.2. Inverse Reinforcement Learning (MaxEnt, Adversarial)**
    * 7.2.1. The Ill-Posed Nature of Reward Inference
    * 7.2.2. Maximum-Entropy and Adversarial IRL
* **7.3. Learning from Preferences: The RLHF Pipeline**
    * 7.3.1. The Reinforcement Learning from Human Feedback (RLHF) Loop
    * 7.3.2. Reward Modelling and KL-Regularised Fine-Tuning
* **7.4. Offline Imitation Learning (IQ-Learn)**
    * 7.4.1. Challenges of Imitation from Static Datasets
    * 7.4.2. Inverse Q-Learning (IQ-Learn) and Adversarial Methods
* **7.5. Foundation Models for Control (Decision/Diffusion Transformers)**
    * 7.5.1. Decision Transformer: IL as Sequence Modelling
    * 7.5.2. Diffusion Policies

**Chapter 8: Abstraction: Hierarchy, State, and Temporal Structure**
* **8.1. State Abstraction and Bisimulation**
    * 8.1.1. Exact and Approximate Abstractions
    * 8.1.2. Bisimulation Metrics and Value Loss Bounds
* **8.2. Temporal Abstraction: The Options Framework**
    * 8.2.1. Semi-Markov Decision Processes (SMDPs)
    * 8.2.2. The Options Framework: Policies, Initiation Sets, Termination Conditions
* **8.3. Hierarchical Architectures and Skill Discovery (Option-Critic, DIAYN)**
    * 8.3.1. Policy-over-Options and Hierarchical Policy Gradients
    * 8.3.2. End-to-End Learning: The Option-Critic Architecture
    * 8.3.3. Unsupervised Skill Discovery via Mutual Information (DIAYN)
* **8.4. Theoretical Benefits of Abstraction**
    * 8.4.1. Sample-Complexity Speed-ups and Regret Bounds
* **8.5. HRL with Learned World Models**
    * 8.5.1. Factored World Models and Latent Goal Spaces

**Chapter 9: Multi-Agent Reinforcement Learning**
* **9.1. Foundations: Stochastic Games and Solution Concepts**
    * 9.1.1. The Stochastic Game Formalism
    * 9.1.2. Solution Concepts: Nash Equilibrium, Correlated Equilibrium
* **9.2. Centralised Training with Decentralised Execution (QMIX, MADDPG)**
    * 9.2.1. Independent Learners and Their Pitfalls
    * 9.2.2. Value Decomposition Networks (VDN, QMIX)
    * 9.2.3. Multi-Agent Deep Deterministic Policy Gradient (MADDPG)
* **9.3. Communication and Coordination**
    * 9.3.1. Differentiable Communication Protocols
    * 9.3.2. Graph-Based Coordination with GNNs
* **9.4. Offline and Hierarchical MARL**
    * 9.4.1. Offline MARL with Static Datasets
    * 9.4.2. Hierarchical Team Architectures
* **9.5. Population-Based Methods and Game Theory Links (PSRO, CFR)**
    * 9.5.1. Fictitious Play and Policy-Space Response Oracles (PSRO)
    * 9.5.2. Counterfactual Regret Minimisation (CFR)

---

### **Part IV: Cross-Cutting Challenges & Future Directions**

**Chapter 10: Learning to Learn: Transfer, Meta, and Continual RL**
* **10.1. Quantifying Task Similarity and Transfer**
    * 10.1.1. Task Distributions and MDP Families
    * 10.1.2. Metrics for Task Similarity (Distributional, Dynamics-based)
* **10.2. Meta-RL: Learning Fast Adaptation (MAML, PEARL, RL²)**
    * 10.2.1. The Bilevel Meta-RL Objective
    * 10.2.2. Gradient-Based Meta-Learning (MAML)
    * 10.2.3. Recurrent/In-Context Meta-Learning (RL²)
    * 10.2.4. Latent-Variable and Posterior Sampling Methods (PEARL)
* **10.3. Continual RL: Overcoming Catastrophic Forgetting (EWC, Replay, PackNet)**
    * 10.3.1. Non-Stationary Environments and Drift Taxonomy
    * 10.3.2. Regularisation-Based Methods (EWC, SI)
    * 10.3.3. Replay and Generative Replay
    * 10.3.4. Dynamic Architectures (Progressive Nets, PackNet)
* **10.4. Open-World and Lifelong Learning**
    * 10.4.1. Autonomous Task Discovery
    * 10.4.2. Safety and Stability in Lifelong Learning
* **10.5. Theoretical Guarantees for Generalization**
    * 10.5.1. Meta-PAC-Bayes and Regret Bounds
    * 10.5.2. Regret and Sample Complexity under Non-Stationarity

**Chapter 11: Safe and Robust Reinforcement Learning**
* **11.1. Formalisms for Safety and Robustness (Constrained MDPs, Robust MDPs)**
    * 11.1.1. Constrained MDPs (CMDPs)
    * 11.1.2. Robust MDPs with Uncertainty Sets (Worst-case, DRO)
* **11.2. Safe Exploration (Shielding, Lagrangian Methods, Conservative Baselines)**
    * 11.2.1. Primal-Dual and Lagrangian Methods for CMDPs
    * 11.2.2. Shielding and Runtime Enforcement with Barrier Certificates
* **11.3. Robust Planning and Control (Distributionally Robust DP, Tube MPC)**
    * 11.3.1. Robust Dynamic Programming
    * 11.3.2. Model-Predictive Control with Robust Tubes
* **11.4. Verification and Runtime Assurance**
    * 11.4.1. Formal Specification and Reachability Analysis
    * 11.4.2. Online Falsification and Counterexample Search
* **11.5. Safe Offline and Imitation Learning**
    * 11.5.1. Offline Constrained RL
    * 11.5.2. Safe Policy Improvement from Demonstrations

**Chapter 12: Synthesis and Open Problems**
* **12.1. A Unified View of RL Paradigms**
    * 12.1.1. A Unified Bayesian Perspective
    * 12.1.2. The Interplay of Model-Free and Model-Based Approaches
* **12.2. Grand Challenges: Sample Complexity, Generalization, and Scalability**
    * 12.2.1. Long-Horizon and Sparse-Reward Credit Assignment
    * 12.2.2. Robustness and Generalisation to Out-of-Distribution Shifts
* **12.3. Computational vs. Statistical Limits in Deep RL**
    * 12.3.1. Theoretical Gaps in Non-Linear Function Approximation
    * 12.3.2. Representation-Aware Exploration and Planning
* **12.4. The Future of Foundation Models in Decision-Making**
    * 12.4.1. Integration with Language, Vision, and Action
    * 12.4.2. Alignment, Governance, and Societal Implications
