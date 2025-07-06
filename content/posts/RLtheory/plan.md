---
date: "2025-07-03"
title: "Tentative Plan for RL Theory Series"
summary: "Tentative Plan for RL Theory Series"
category: Plan
series: ["RL Theory"]
author: "Author: Bryan Chan"
hero: /assets/images/hero3.png
image: /assets/images/card3.png
---

# **Comprehensive Reinforcement Learning: Theory and Practice**

## **1. MDP Foundations & Optimality**

### **1.1. Mathematical Foundations**
- **1.1.1.** Measure-theoretic set-up
- **1.1.2.** Formal definition of an MDP: $(S, A, P, r, \gamma)$
- **1.1.3.** Discount factor $\gamma$ & effective horizon $H_{\gamma, \varepsilon}$

### **1.2. Policy Framework**
- **1.2.1.** Policies and the agent–environment loop
- **1.2.2.** Probability of trajectories (Ionescu–Tulcea theorem)
- **1.2.3.** Return, value functions, and optimality notions
- **1.2.4.** Memoryless vs. general policies; fundamental optimality theorem

### **1.3. Optimality Theory**
- **1.3.1.** Objective functions and modelling choices
- **1.3.2.** Discounted occupancy measures
- **1.3.3.** Bellman operators: $T^\pi, T^*$
- **1.3.4.** Key analytical properties (contraction, error bounds)
- **1.3.5.** Greedy policies and the Fundamental Theorem

### **1.4. Concentration Inequalities**
- **1.4.1.** Self-normalised concentration inequalities (Elliptical Potential, Bernstein)

---

## **2. Exact Dynamic Programming**

### **2.1. Value Iteration Theory**
- **2.1.1.** Fundamental Theorem of Dynamic Programming (finite MDPs)
- **2.1.2.** Effective horizon and Bellman update: $v_{k+1} = T^*v_k$
- **2.1.3.** Policy-error (greedy) bound
- **2.1.4.** Fixed-point iteration via Banach's lemma
- **2.1.5.** Finite-horizon interpretation
- **2.1.6.** Algorithmic description, convergence, and $\varepsilon$-stopping rule
- **2.1.7.** Geometry of value functions (Dadashi et al.)
- **2.1.8.** Banach's Fixed-Point Theorem (background)
- **2.1.9.** Linear programming view (primal/dual)
- **2.1.10.** Value iteration as *approximate* planning
- **2.1.11.** Runtime of $\varepsilon$-optimal planning with Value Iteration
- **2.1.12.** Computational complexity of exact planning
- **2.1.13.** $\delta$-$\varepsilon$ error-control summary

### **2.2. Policy Iteration Analysis**
- **2.2.1.** Definition of the Policy Iteration (PI) algorithm
- **2.2.2.** Advantage function and the Performance-Difference Identity
- **2.2.3.** Geometric-Progress Lemma
- **2.2.4.** Geometric convergence of value error
- **2.2.5.** Strict-Progress Lemma (sub-optimal action elimination)
- **2.2.6.** Overall runtime bound (Scherrer)
- **2.2.7.** Value Iteration vs. Policy Iteration comparison
- **2.2.8.** Proof that PI is generally faster than VI
- **2.2.9.** Mixing rates and span-seminorm contraction
- **2.2.10.** Upper and lower runtime bounds (Ye; Feinberg-Huang-Scherrer)
- **2.2.11.** Measure-theoretic view of occupancy-measure projection

### **2.3. Learned-Model Dynamic Programming**
- **2.3.1.** Model-bias bounds for Bellman operators
- **2.3.2.** Ensemble variance as a proxy for model error $\varepsilon_P$
- **2.3.3.** Optimism under model uncertainty and Thompson sampling

---

## **3. Online Planning in Discounted MDPs**

### **3.1. Problem Formulation**
- **3.1.1.** Motivation (curse of dimensionality; local planning)
- **3.1.2.** Access modes: global → local → online
- **3.1.3.** Environment model: black-box simulator
- **3.1.4.** Formal statement of the online planning problem

### **3.2. Algorithmic Framework**
- **3.2.1.** Optimisation language and oracle types
- **3.2.2.** Agent–environment interaction loop
- **3.2.3.** Online planner and $\delta$-soundness definition
- **3.2.4.** Cost metrics (queries and arithmetic)

### **3.3. Analysis & Bounds**
- **3.3.1.** Baseline algorithm (recursive value evaluation)
- **3.3.2.** Upper runtime bounds (deterministic and sparse-sampling)
- **3.3.3.** Matching lower bounds: $\Omega(A^H)$
- **3.3.4.** Local vs. online access trade-offs
- **3.3.5.** Sampling and averaging fundamentals
- **3.3.6.** Policy-error analysis ($\varepsilon$ and "almost-$\varepsilon$" cases)
- **3.3.7.** Parameter selection for $\delta$-soundness: choosing $m, H, \zeta$

### **3.4. Extensions & Applications**
- **3.4.1.** Open questions and extensions (e.g., links to MCTS)
- **3.4.2.** Model-Predictive Control with Learned Simulators
    - **3.4.2.1.** Cross-Entropy Method (CEM) and trajectory optimisation
    - **3.4.2.2.** TD-MPC algorithm and stability analysis
    - **3.4.2.3.** Comparative runtime vs. MCTS

---

## **4. Value-Function Approximation & Approximate Policy Iteration**

### **4.1. Value-Function Approximation**
- **4.1.1.** Approximate Universal Value-Function Realizability
- **4.1.2.** Approximate Policy Evaluation: Monte Carlo rollout + weighted Least Squares
- **4.1.3.** Extrapolation-error control in weighted Least Squares
- **4.1.4.** Kiefer–Wolfowitz Theorem (G-optimal design)
- **4.1.5.** Corollary: Extrapolation control via optimal design
- **4.1.6.** LSPE-G high-probability error bound

### **4.2. Approximate Policy Iteration (API)**
- **4.2.1.** Geometric Progress Lemma with approximate improvement
- **4.2.2.** API theorem: $\frac{\gamma^k}{1-\gamma} + \varepsilon$ bounds
- **4.2.3.** API with approximate action-value functions (corollary)
- **4.2.4.** Least-Squares Policy Iteration (LSPI) algorithm and guarantees

### **4.3. State Abstractions**
- **4.3.1.** Motivation (sample sharing, dimensionality reduction)
- **4.3.2.** Statistical trade-off: sample size vs. approximation error
- **4.3.3.** Exact abstraction hierarchy ($\pi^*$, $Q^*$, model-irrelevance)
- **4.3.4.** Key theorems on the hierarchy and error bounds
- **4.3.5.** Improvements and variants (homomorphisms, utile distinctions)
- **4.3.6.** Approximate abstractions and bisimulation metrics ($\varepsilon$-$\pi^*$, $\varepsilon$-$Q^*$, $\varepsilon_R$, $\varepsilon_P$ bisimulation)
    - **4.3.6.1.** **Action-bisimulation** (Rudolph et al., 2024)
    - **4.3.6.2.** **Effect-equivalent abstraction** (Mavor-Parker et al., 2025)
- **4.3.7.** Bounding value loss for approximate abstractions
- **4.3.8.** Finite-sample analysis ($n_\phi(D)$, Hoeffding bound)
    - **4.3.8.1.** **Encoder bias term**: Bounding bias from a learned encoder $\hat{\phi}$ with $\|\hat{\phi} - \phi^*\|_\infty \le \varepsilon_{\text{enc}}$.
- **4.3.9.** Bridging symbolic and learned abstractions
    - **4.3.9.1.** Motivation and open problem statement
    - **4.3.9.2.** **Self-supervised Markov latent-state discovery** (Sobal et al., 2025)
    - **4.3.9.3.** Iterative *refine–plan* pipeline
    - **4.3.9.4.** Conjectured regret bound: $\tilde{O}\bigl(H\sqrt{dT} + \frac{\varepsilon_{\text{enc}}}{1-\gamma}\bigr)$

### **4.4. Offline / Batch Reinforcement Learning**
- **4.4.1.** Motivation (logged data, safety, distribution shift, sample-efficiency)
- **4.4.2.** Formal framework and assumptions (static dataset $\mathcal{D}$, behaviour policy $\beta$, realizability, Bellman-completeness, coverage $C$)
- **4.4.3.** Core algorithm: **Fitted Q-Iteration (FQI)**
- **4.4.4.** Error-analysis pipeline (uniform deviation, one-step error, propagation)
- **4.4.5.** Pessimistic / Conservative algorithms (**CQL**, **OPAL**, **IQL**)
- **4.4.6.** Robustness to corrupted or sub-optimal logs
- **4.4.7.** Representation learning for Bellman completeness
- **4.4.8.** Empirical benchmarks and practical guidelines (**D4RL-2025**, **MineRL-Offline**)

### **4.5. Latent-World-Model Learning**
- **4.5.1.** Recurrent State-Space Models (**RSSM**) and the ELBO
- **4.5.2.** **Dreamer** variants (V2, V3) and KL balancing
- **4.5.3.** Representation capacity and empirical scaling laws

---

## **5. Sampling and Computational Complexity**

### **5.1. Foundations & State-Representation Learning (SRL)**
- **5.1.1.** Markov Decision Processes (recap)
- **5.1.2.** Feature maps $\phi$: from hand-crafted bases to learned encoders $\phi_\psi$
- **5.1.3.** Six SRL families (metric, contrastive, augmentation, world-model, reconstruction, auxiliary-task)
- **5.1.4.** Evaluation protocols (DMControl-100k, Atari-100k, ProcGen)
- **5.1.5.** Norms and error metrics ($\|\cdot\|_\infty$, $\|\cdot\|_2$, bisimulation-metric)
- **5.1.6.** Pre-computed core sets and LSPI recap

### **5.2. Probabilistic Tools for Sampling Analysis**
- **5.2.1.** Hoeffding's inequality
- **5.2.2.** Azuma–Hoeffding for martingales
- **5.2.3.** Union-bound for simultaneous guarantees
- **5.2.4.** Worked example: uniform-sampling best-arm identification

### **5.3. Covering Numbers & Uniform Convergence**
- **5.3.1.** $\ell_\infty$ covers and growth with dimension $d$
- **5.3.2.** Lipschitz compositions and loss-class covering
- **5.3.3.** Sample-complexity bound: $\tilde{O}\left(\sqrt{\frac{\log N_\varepsilon}{n}}\right)$

### **5.4. Limits of Query-Efficient Planning**
- **5.4.1.** Definition of $(\delta, \varepsilon)$-sound online planners
- **5.4.2.** Large-$A$ lower bound
    - **5.4.2.1.** Johnson–Lindenstrauss packing lemma
    - **5.4.2.2.** High-probability "needle" lemma
    - **5.4.2.3.** Exponential query cost $e^{\Omega(d)}$ (Weisz et al., 2021)
- **5.4.3.** Fixed-horizon, small-$A$ lower bound
    - **5.4.3.1.** Horizon-dependent fundamental theorem
    - **5.4.3.2.** Query complexity $\tilde{\Omega}(A^{H/H})$ vs "large-$H$" regime
    - **5.4.3.3.** Tightened few-actions lower bound (Weisz et al., 2022)
- **5.4.4.** Open question: computational gap when $A$ is fixed but $H\to\infty$

### **5.5. Planning under Realizability**
- **5.5.1.** $q^*$-realizability: linear assumption, global planner variant
- **5.5.2.** $v^*$-realizability (**TensorPlan**)
    - **5.5.2.1.** Interaction protocol and local simulator calls
    - **5.5.2.2.** Ridge regression hypothesis set $\Theta$
    - **5.5.2.3.** Optimism via square-root bonus ($\beta$)
    - **5.5.2.4.** Covering-number analysis (size $\tilde{O}(d^2)$)
    - **5.5.2.5.** Open question: polynomial-time implementation

### **5.6. Exploration in Linear MDPs**
- **5.6.1.** Problem set-up and boundedness assumptions
- **5.6.2.** **LSVI-UCB** algorithm: step-wise ridge with UCB bonus
- **5.6.3.** Martingale concentration (Azuma) for adaptive data
- **5.6.4.** Elliptical-potential lemma and regret: $\tilde{O}(H^2 d \sqrt{T})$
- **5.6.5.** Bernstein-bonus refinement (**LSVI-UCB+**) and near-minimax regret: $\tilde{O}(H d \sqrt{T})$
- **5.6.6.** Variance-aware bonuses and practical tuning
- **5.6.7.** Robustness to encoder error: regret bound becomes $\tilde{O}\bigl(H^2\sqrt{dT} + H\sqrt{T}\varepsilon_{\text{enc}}\bigr)$

### **5.7. Linear-Programming View of MDPs**
- **5.7.1.** Primal LP: minimise $d_0^\top V$ s.t. $V \geq T^*V$
- **5.7.2.** Dual LP: occupancy-measure constraints, $d^\pi$ cone
- **5.7.3.** Monotonicity of $T$ and convergence to $V^*$

### **5.8. Sample-Complexity of Model Learning**
- **5.8.1.** PAC bounds for $\varepsilon_P$-accurate models
- **5.8.2.** Lower bounds under model misspecification
- **5.8.3.** Sample-efficient exploration via information gain

### **5.9. Synthesis & Open Questions**
- **5.9.1.** Comparison table: lower vs. upper bounds ($d, A, H, \gamma$)
- **5.9.2.** Gap: factor-$H$ mismatch in discounted long-horizon setting
- **5.9.3.** Representation-learning for exploration
- **5.9.4.** Computational vs. query complexity: can **TensorPlan** be made polynomial-time?

---

## **6. Robust & Safe Model-Based Reinforcement Learning**

### **6.1. Distributionally Robust MDPs**
- **6.1.1.** KL/TVD ambiguity sets and robust Bellman operators
- **6.1.2.** Contraction and value-error guarantees under robustness

### **6.2. $\mathcal{L}_1$-Adaptive Model-Based Control**
- **6.2.1.** Online parameter estimator and adaptation law
- **6.2.2.** Bounded tracking-error theorem

### **6.3. Ensemble Uncertainty & Risk Metrics**
- **6.3.1.** Epistemic–aleatoric decomposition
- **6.3.2.** CVaR and other coherent risk measures

### **6.4. Safe Model-Predictive Control**
- **6.4.1.** Terminal set and tube MPC with learned dynamics
- **6.4.2.** Constraint-tightening for probabilistic safety

### **6.5. Runtime Monitoring & Verification**
- **6.5.1.** Temporal-logic specifications
- **6.5.2.** Online falsification and fallback policies

---

## **7. Model-Free Prediction**

### **7.1. Problem Formulation**
- **7.1.1.** MDP refresher
- **7.1.2.** Episodic vs. continuing tasks
- **7.1.3.** Return $G_t$ and value-function target $v^\pi$
- **7.1.4.** Mean-squared-error objective

### **7.2. Monte-Carlo (MC) Prediction**
- **7.2.1.** Full-return estimate and the variance problem
- **7.2.2.** Incremental MC update
- **7.2.3.** Bias–variance trade-off

### **7.3. TD(0): One-Step Bootstrapping**
- **7.3.1.** TD target and TD-error $\delta_t$
- **7.3.2.** On-line incremental update
- **7.3.3.** Geometric view (DP vs. MC)
- **7.3.4.** Tabular convergence proof sketch

### **7.4. $n$-Step TD & $\lambda$-Return (Forward View)**
- **7.4.1.** Derivation of the $n$-step return $G_t^{(n)}$
- **7.4.2.** Continuum from TD(0) to MC
- **7.4.3.** Weighted mixture: the $\lambda$-return $G_t^\lambda$
- **7.4.4.** Analytical bias–variance curve

### **7.5. Eligibility Traces (Backward View)**
- **7.5.1.** Accumulating vs. replacing traces
- **7.5.2.** Proof of forward $\leftrightarrow$ backward equivalence (tabular)
- **7.5.3.** **TD($\lambda$)**, **Sarsa($\lambda$)**, and Watkins **Q($\lambda$)** update rules
- **7.5.4.** **True-online TD($\lambda$)**

### **7.6. Analysis & Guarantees**
- **7.6.1.** Detailed bias–variance trade-off across $n$ and $\lambda$
- **7.6.2.** Robbins-Monro conditions; convergence with linear function approximation
- **7.6.3.** Divergence counter-example (off-policy + function approx.)
- **7.6.4.** Practical heuristics (step-size schedules, trace resets, $\lambda$-sweeps)

---

## **8. Model-Free Control (On- & Off-Policy)**

### **8.1. Problem Formulation**
- **8.1.1.** Control objective: $\max_\pi \mathbb{E}\left[\sum_{t=0}^\infty \gamma^{t} r_t\right]$
- **8.1.2.** "Deadly triad" recap: bootstrapping + function approximation + off-policy

### **8.2. Sarsa & On-Policy TD Control**
- **8.2.1.** Tabular **Sarsa** update and convergence
- **8.2.2.** **Expected Sarsa** and variance comparison
- **8.2.3.** $\lambda$-extension with eligibility traces

### **8.3. Q-Learning**
- **8.3.1.** Watkins **Q-learning** update rule
- **8.3.2.** Non-asymptotic sample-complexity: $\tilde{O}\bigl(\frac{SA}{(1-\gamma)^3\varepsilon^2}\bigr)$
- **8.3.3.** **Double Q-learning** and bias correction

### **8.4. Variance-Reduced & Regularised Variants**
- **8.4.1.** **Cascade Q-learning**; **RegQ** (provably convergent with linear FA)
- **8.4.2.** Stability analysis under function approximation

### **8.5. Deep Q-Networks & Rainbow**
- **8.5.1.** Replay buffer and target network heuristics
- **8.5.2.** Distributional Q-learning, prioritized replay, noisy nets, etc.

### **8.6. Actor-Critic Methods**
- **8.6.1.** Policy-gradient theorem and importance-sampling ratios
- **8.6.2.** **DDPG**, **TD3**, **SAC**; entropy regularisation

### **8.7. Convergence with Function Approximation**
- **8.7.1.** Baird's counter-example revisited
- **8.7.2.** Gradient-TD view; projected Bellman operator contraction

### **8.8. Finite-Sample Regret Bounds**
- **8.8.1.** Q-learning + UCB exploration upper bounds
- **8.8.2.** Matching lower bounds and optimality gaps

### **8.9. Open Questions**
- **8.9.1.** Non-linear function approximation theory
- **8.9.2.** Sample-efficient exploration strategies

---

## **9. Off-Policy Learning: Prediction & Control**

### **9.1. Taxonomy**
- **9.1.1.** Distinguishing OPE, prediction, and control
- **9.1.2.** Importance-sampling vs. density-ratio methods

### **9.2. Importance-Sampling (IS) Fundamentals**
- **9.2.1.** Ordinary vs. weighted IS; variance properties
- **9.2.2.** Capping, clipping, and relative IS

### **9.3. Multi-Step Off-Policy TD**
- **9.3.1.** **Tree-Backup($\lambda$)**, **ABTD($\zeta$)**, **V-trace($\lambda$)**
- **9.3.2.** Bias–variance trade-offs

### **9.4. Off-Policy Actor-Critic**
- **9.4.1.** Deterministic policy gradient with IS
- **9.4.2.** Twin-critic and clipped-weighting tricks

### **9.5. Batch / Offline RL**
- **9.5.1.** Cross-reference: Fitted Q-Iteration (§4.4)
- **9.5.2.** **Conservative Q-Learning (CQL)**; pessimistic bootstrapping

### **9.6. High-Dimensional Action Spaces**
- **9.6.1.** State-value-only critics (e.g., **V-learn**)

### **9.7. Theoretical Guarantees**
- **9.7.1.** PAC bounds for linear function approximation
- **9.7.2.** Variance lower bound: $O\bigl(\frac{1}{(1-\gamma)^4}\bigr)$

### **9.8. Practical Heuristics**
- **9.8.1.** Behaviour-policy regularisation, trust regions, replay prioritisation

### **9.9. Summary & Open Problems**
- **9.9.1.** Safe policy improvement
- **9.9.2.** Data-quality diagnostics and benchmarks

---

## **10. Policy Search & Policy-Gradient Methods**

### **10.1. Foundations**
- **10.1.1.** Objective functions and the likelihood-ratio gradient
- **10.1.2.** Monte-Carlo Policy Gradient (**REINFORCE**)
- **10.1.3.** Baselines and variance-reduction theory

### **10.2. Actor–Critic Architecture**
- **10.2.1.** Actor–Critic architecture and **GAE**
- **10.2.2.** Natural Policy Gradient and compatible function approximation

### **10.3. Trust Region Methods**
- **10.3.1.** Trust-Region Policy Gradient (**TRPO**)
- **10.3.2.** **Proximal Policy Optimization (PPO)**: clip vs. KL penalty

### **10.4. Deterministic Policy Gradients**
- **10.4.1.** Deterministic PG, **DDPG**, **TD3**
- **10.4.2.** Maximum-Entropy RL and **Soft Actor-Critic (SAC)**

### **10.5. Advanced Topics**
- **10.5.1.** Exploration, entropy regularisation, and KL constraints
- **10.5.2.** Off-policy corrections and importance sampling (**IS**, **V-trace**, **Q-Prop**)
- **10.5.3.** Gradient-free policy search (**ES**, **CEM**, **GPS**)

### **10.6. Theoretical Analysis**
- **10.6.1.** Sample-complexity bounds, bias-variance analysis, lower bounds
- **10.6.2.** Safe and constrained PG (**CPO**, Lagrangian, Lyapunov)

### **10.7. Implementation & Applications**
- **10.7.1.** Implementation pragmatics (normalisation, clipping, LR schedules)
- **10.7.2.** Case studies (MuJoCo, Atari, robotics)

### **10.8. Research Frontiers**
- **10.8.1.** Credit assignment, large-action spaces

---

## **11. Partially Observable Reinforcement Learning**

### **11.1. POMDP Foundations**
- **11.1.1.** Formal definition of a POMDP: $(S, A, O, T, \Omega, R, \gamma)$
- **11.1.2.** Bayes filter and belief state: $b_{t+1} = \tau(b_t, a_t, o_{t+1})$
- **11.1.3.** Optimality of belief-stationary policies

### **11.2. Exact Planning in Belief Space**
- **11.2.1.** $\alpha$-vector value iteration; piece-wise-linear-convex value function
- **11.2.2.** Complexity (PSPACE-complete) of exact POMDP planning

### **11.3. Approximate Planning**
- **11.3.1.** **Point-Based Value Iteration (PBVI)** and variants (**HSVI**, **SARSOP**)
- **11.3.2.** Anytime guarantees and error bounds

### **11.4. Representation Learning under Partial Observability**
- **11.4.1.** Predictive State Representations (PSRs)
- **11.4.2.** Recurrent memory architectures and finite-memory controllers
- **11.4.3.** Sample-complexity bounds with windowed histories

### **11.5. PAC & Regret Guarantees**
- **11.5.1.** PAC-RL for POMDPs with privileged simulators
- **11.5.2.** Regret lower and upper bounds in latent-state environments

### **11.6. Software & Benchmarks**
- **11.6.1.** **SARSOP** bindings and `pomdp` R package
- **11.6.2.** Small-scale benchmarks (Tiger, Light-Dark, RockSample)

### **11.7. Open Questions**
- **11.7.1.** Efficient exploration with latent states
- **11.7.2.** Memory size vs. sample complexity trade-off

---

## **12. Bayesian Reinforcement Learning**

### **12.1. Bayes-Adaptive MDPs (BAMDPs)**
- **12.1.1.** Unknown kernel as a latent parameter $\theta$
- **12.1.2.** Augmented state $(s_t, \theta_t)$ and equivalence to POMDP

### **12.2. Exact & Tree-Search Planning**
- **12.2.1.** Bayes-adaptive forward-search (**BFS3**)
- **12.2.2.** **ADA-MCTS** and non-stationary safe exploration

### **12.3. Posterior-Sampling RL (PSRL)**
- **12.3.1.** Thompson sampling over the MDP posterior
- **12.3.2.** Regret bounds: $\tilde{O}(\sqrt{HSAT})$

### **12.4. Variational & Approximate Bayesian RL**
- **12.4.1.** Evidence Lower Bound (ELBO) on value functions
- **12.4.2.** Regret under approximation error

### **12.5. PAC-Bayes & Lifelong RL**
- **12.5.1.** PAC-Bayes generalisation bounds for RL
- **12.5.2.** **EPIC** algorithm and distilled priors

### **12.6. Computation vs. Statistical Efficiency**
- **12.6.1.** Ensemble methods and bootstrap exploration
- **12.6.2.** Connections to Fitted Q-Iteration (§4.4)

### **12.7. Open Questions**
- **12.7.1.** Bayesian exploration in continuous spaces
- **12.7.2.** Structural priors and safe Bayesian RL

---

## **13. Imitation & Inverse Reinforcement Learning**

### **13.1. Problem Formulation**
- **13.1.1.** MDP without a reward function; expert demonstrations $\mathcal{D}$
- **13.1.2.** Occupancy measures and divergence objectives

### **13.2. Behavioural Cloning (BC)**
- **13.2.1.** Supervised log-likelihood objective
- **13.2.2.** Compounding-error bound in horizon $T$

### **13.3. Dataset Aggregation (DAgger)**
- **13.3.1.** Interactive expert-label querying protocol
- **13.3.2.** No-regret analysis and constant imitation error

### **13.4. Offline Imitation Learning**
- **13.4.1.** Importance-weighted BC under covariate shift
- **13.4.2.** Statistical consistency and finite-sample rates

### **13.5. Adversarial Imitation Learning**
- **13.5.1.** **GAIL**: Min-max objective and JS-divergence interpretation
- **13.5.2.** Convergence guarantee via occupancy matching

### **13.6. Foundations of Inverse RL (IRL)**
- **13.6.1.** Ill-posedness and feature-expectation matching
- **13.6.2.** Apprenticeship-learning value bound

### **13.7. Maximum-Entropy IRL**
- **13.7.1.** MaxEnt principle and convex dual derivation
- **13.7.2.** Sample complexity of partition-function estimation

### **13.8. Bayesian & PAC-style IRL**
- **13.8.1.** Posterior over reward hypotheses
- **13.8.2.** PAC reward-set estimation bounds

### **13.9. Adversarial IRL (AIRL)**
- **13.9.1.** Potential-based reward recovery
- **13.9.2.** Dynamics-invariant transfer theorem

### **13.10. Preference-Based & Human-in-the-Loop RL**
- **13.10.1.** Binary preference models and active queries
- **13.10.2.** Regret bounds for preference elicitation

### **13.11. Open Questions & Complexity Gaps**
- **13.11.1.** Lower bounds under partial observability
- **13.11.2.** Robustness to corrupted demonstrations
- **13.11.3.** Reward-poisoning attacks and defences

---

## **14. Hierarchical Reinforcement Learning & Temporal Abstraction**

### **14.1. SMDP Foundations**
- **14.1.1.** Formal definitions and notation (episodic & continuing)
- **14.1.2.** Call-and-return semantics
- **14.1.3.** Bellman operators: discounted and average-reward
- **14.1.4.** Contraction proof and fixed points (bounded holding-time)
- **14.1.5.** Policy-evaluation algorithms (TD, MC, $\lambda$-returns)

### **14.2. The Options Framework**
- **14.2.1.** Core components: $\mathcal{I}, \pi_o, \beta_o$
- **14.2.2.** Policy-over-Options $\iff$ SMDP Lemma
- **14.2.3.** Fundamental Theorem of Options
- **14.2.4.** Execution variants: call-and-return vs. interruptible
- **14.2.5.** Worked example: bottleneck grid world (see §14.12)

### **14.3. Option-Learning & Hierarchical Policy Gradient**
- **14.3.1.** Intra-Option Policy Gradient Theorem
- **14.3.2.** Termination-Gradient Theorem
- **14.3.3.** Convergence via Two-Time-Scale Stochastic Approximation
- **14.3.4.** Variants: Natural Option-Critic, Diversity-Enriched OC
- **14.3.5.** Practical tips: experience replay across option boundaries

### **14.4. Skill / Option Discovery**
- **14.4.1.** Spectral and Eigen-Options (graph Laplacian)
- **14.4.2.** Skill-Chaining and reachability graphs
- **14.4.3.** Unsupervised MI methods (**DIAYN**, **DADS**, **VALOR**, **ADC**)
- **14.4.4.** Diversity regularisers (DPP, Wasserstein, VIC)
- **14.4.5.** Interest functions and state-density weighting
- **14.4.6.** Evaluation benchmarks (Procgen Maze, MuJoCo AntMaze, Minigrid)

### **14.5. Multi-Level Architectures**
- **14.5.1.** Feudal RL & **FuN**
- **14.5.2.** Hierarchical Actor-Critic (**HAC**)
- **14.5.3.** **HIRO** and off-policy correction
- **14.5.4.** **SHIRO**: Entropy-Regularised HIRO
- **14.5.5.** Planning architectures (**Dreamer-H**, **MCTS-HRL**)

### **14.6. Sample-Complexity & Regret Theory**
- **14.6.1.** Information-theoretic lower bounds (goal-conditioned, horizon $H$)
- **14.6.2.** Matching upper bounds (Robert et al. 2023)
- **14.6.3.** Polynomial speed-up conditions (cover time $\le \kappa$)
- **14.6.4.** Misspecification regret (imperfect option set)
- **14.6.5.** Exploration–exploitation trade-offs (open question)

### **14.7. Hierarchical Representation & World Models**
- **14.7.1.** Latent goal spaces via MI objectives (CPC, InfoNCE)
- **14.7.2.** Bisimulation and successor features
- **14.7.3.** World-model factorisation (manager latent vs. worker dynamics)
- **14.7.4.** Planning-error propagation bound: $\varepsilon_{\text{model}} \le \varepsilon_{\text{high}} + H \cdot \varepsilon_{\text{low}}$
- **14.7.5.** Latent-dimension vs. optimality gap theorem

### **14.8. Transfer, Continual & Lifelong HRL**
- **14.8.1.** Option-indexing meta-learning
- **14.8.2.** Zero-shot option reuse guarantees
- **14.8.3.** Skill-composition transformers
- **14.8.4.** Catastrophic interference mitigation (EWC, orthogonal regularisation)

### **14.9. Offline & Safe HRL**
- **14.9.1.** Batch-RL foundations for SMDPs
- **14.9.2.** Behaviour-Constrained Option Learning (BC-OC)
- **14.9.3.** Shielded and chance-constrained intra-option policies
- **14.9.4.** Safe termination verification via temporal logic
- **14.9.5.** Dataset shift and counterfactual corrections

### **14.10. Partial Observability & POMDP-HRL**
- **14.10.1.** Belief-space managers
- **14.10.2.** Recurrent option policies
- **14.10.3.** Information bottlenecks for options
- **14.10.4.** Applications: ViZDoom, real-world navigation

### **14.11. Multi-Agent Hierarchical Coordination**
- **14.11.1.** Shared skill libraries and Kronecker graphs
- **14.11.2.** Hierarchical credit assignment across agents
- **14.11.3.** Game-theoretic option selection
- **14.11.4.** Case study: StarCraft II macro-actions

### **14.12. Applications & Case Studies**
- **14.12.1.** Robotic manipulation (Meta-World, Franka Kitchen)
- **14.12.2.** Autonomous driving (lane merge, intersection)
- **14.12.3.** Surgical robotics (sub-task libraries, safe terminations)
- **14.12.4.** Listwise recommendation systems
- **14.12.5.** Vision-and-Language Navigation (**HierVLN++**)

### **14.13. Benchmarking & Experimental Protocols**
- **14.13.1.** Standardised task suites (sparse-reward continuous control)
- **14.13.2.** Option-discovery diagnostics
- **14.13.3.** Hyper-parameter grids and reporting standards
- **14.13.4.** Evaluation metrics (SPL, Option-Utility Score, regret)

### **14.14. Open Problems & Future Directions**
- **14.14.1.** Automatic granularity selection
- **14.14.2.** Hierarchical exploration–exploitation theory
- **14.14.3.** Reward / sub-goal mis-specification
- **14.14.4.** Scalable benchmark suites
- **14.14.5.** Multi-agent and decentralised credit assignment
- **14.14.6.** Human-in-the-loop and alignment-sensitive HRL

---

## **15. Multi-Agent Reinforcement Learning & Stochastic Games**

### **15.1. Overview & Taxonomy**
- **15.1.1.** From single-agent RL to economic and societal systems
- **15.1.2.** Task taxonomy: cooperative, zero-sum, mixed-motive, common-payoff
- **15.1.3.** Game types: fully observable, Dec-POMDP, POSG, Bayesian games
- **15.1.4.** Modelling dimensions: observability, communication, action coupling
- **15.1.5.** Deployment constraints: offline data, safety, scalability

### **15.2. Mathematical Foundations**
- **15.2.1.** Stochastic-game formalism
- **15.2.2.** Information structures: public/private info, I-POMDP
- **15.2.3.** Solution concepts: Nash, Markov-perfect, Correlated Equilibria
- **15.2.4.** Occupancy measures and convex programming
- **15.2.5.** Multi-agent Bellman operators and value functions
- **15.2.6.** Computational complexity landscape (PPAD, PSPACE, NEXP)

### **15.3. Planning & Dynamic Programming**
- **15.3.1.** **Shapley value** and policy iteration
- **15.3.2.** Linear programming for zero-sum/team games
- **15.3.3.** Fictitious play, **AlphaRank**, and **Counterfactual Regret Minimization (CFR)**
- **15.3.4.** Multi-agent tree search (**Max-n**, **MCTS-MARL**, AlphaZero variants)
- **15.3.5.** Planning under partial observability (**JESP**, **MAA***)

### **15.4. Model-Free Learning Algorithms**
- **15.4.1. Value-based learning**
    - Independent **Q-learning** (lenient, hysteretic)
    - Value decomposition: **VDN**, **QMIX**, **QPLEX**, **DCG**
    - Zero-sum solvers: **Minimax-Q**, Nash-Q
    - Opponent-aware methods: **LOLA-DICE**, **LIO**
- **15.4.2. Policy-gradient & actor–critic**
    - CTDE critics: **COMA**, **MADDPG**, **MAPPO**
    - Mean-field actor–critic (**MF-MARL**)
    - Population gradients: **LOLA**, **PSRO-RL**
- **15.4.3. Sequence-model policies**
    - **Multi-Agent Decision Transformers (MAM-DT)**
    - **Diffusion-based control (MADiff)**
- **15.4.4. Mean-field & population-based learning**

### **15.5. Model-Based Methods & Imagination-Augmented Control**
- **15.5.1.** Learning joint and factored dynamics models
- **15.5.2.** Imagination and tree search with learned models (**MuZero-MARL**, **Dreamer-Multi-Agent**)
- **15.5.3.** Model-predictive and receding-horizon control

### **15.6. Offline / Batch Multi-Agent RL**
- **15.6.1.** Problem definition and dataset shift
- **15.6.2.** Conservative value estimation (**C-MARL**, **COMBO-MA**)
- **15.6.3.** In-Sample Sequential Policy Optimisation (**InSPO**)
- **15.6.4.** Offline datasets and evaluation (**MARL-Bench Offline**, **D4MARL**)

### **15.7. Hierarchical & Skill-based MARL**
- **15.7.1.** Options and feudal team architectures
- **15.7.2.** Latent skill discovery and temporally extended actions
- **15.7.3.** Hierarchical communication and planning

### **15.8. Partial Observability, Communication & Coordination**
- **15.8.1.** Recurrent and transformer-based MARL (**DRQN**, **R-MAPPO**, **GTRX**)
- **15.8.2.** Differentiable communication (**CommNet**, **DIAL**, **IC3Net**)
- **15.8.3.** Emergent discrete language and LLM-token protocols
- **15.8.4.** Credit assignment (**COMA**, **MABC**)
- **15.8.5.** Graph-based coordination and GNNs

### **15.9. Scalability & Distributed Systems**
- **15.9.1.** Mean-field and continuum-limit approximations
- **15.9.2.** Swarm / crowd RL
- **15.9.3.** Parallel rollouts and distributed optimisation (**IMPALA**, **R2D2-MARL**)
- **15.9.4.** Asynchronous learner–actor pipelines (**RLlib-MARL**, **DeepSpeed-RL**)

### **15.10. Safety, Robustness & Societal Aspects**
- **15.10.1.** Robust MARL against adversarial training and distributional shift
- **15.10.2.** Safe exploration and constraint satisfaction
- **15.10.3.** Fairness, collusion, and welfare metrics
- **15.10.4.** Mechanism design and incentive alignment
- **15.10.5.** Alignment and normative constraints (Constitutional RL for swarms)

### **15.11. Benchmarks, Environments & Evaluation**
- **15.11.1.** Cooperative suites: **SMAC-v2**, Google Football, **Overcooked-MR-2024**
- **15.11.2.** Competitive/mixed suites: Hanabi, Stratego, **MPE-DG**
- **15.11.3.** Real-world control: traffic, warehouses, energy markets
- **15.11.4.** Metrics: win-rate, exploitability, NashConv, generalisation
- **15.11.5.** Reproducibility stacks: **PettingZoo**, **MARL-Bench**, **RLlib-MARL**

### **15.12. Theoretical Guarantees & Complexity**
- **15.12.1.** PAC bounds for team and zero-sum stochastic games
- **15.12.2.** Online regret minimisation (CFR, OMD)
- **15.12.3.** Convergence of Gradient Descent/Ascent (GDA)
- **15.12.4.** Propagation-of-chaos and mean-field convergence
- **15.12.5.** Sample-complexity gaps: CTDE vs. fully decentralised

### **15.13. Open Questions & Research Frontiers**
- **15.13.1.** Equilibrium selection with function approximation
- **15.13.2.** Scaling on-policy learning to 1000+ agents
- **15.13.3.** Standardised offline MARL benchmarks
- **15.13.4.** Human–AI collaboration and norm formation
- **15.13.5.** Sim-to-real transfer and cross-game generalisation

---

## **16. Task Distributions & Transfer Principles**

### **16.0. Orientation & Road-Map**
- **16.0.1.** Motivation and historical context
- **16.0.2.** Running toy-examples
- **16.0.3.** Reader’s guide (theory → algorithms → evaluation flow)

### **16.1. Sample-Space Formulation of an MDP Family**
1.  **16.1.1.** MDP recap & notation (states S, actions A, transitions P, rewards R, discount γ)
2.  **16.1.2.** Task-generating random variables: latent θ, generative map $f:\Theta\to\mathcal M$
3.  **16.1.3.** Sampling regimes: IID batches, non-IID streams, adversarial sequences
4.  **16.1.4.** Structural assumption library: shared (S,A), Lipschitz in θ, compact support
5.  **16.1.5.** Canonical task families: contextual bandits, linear systems, domain-randomised robotics
6.  **16.1.6.** Extensions: belief-MDPs for partial observability, causal task graphs, exchangeable processes
7.  **16.1.7.** Common pitfalls: support mismatch, hidden confounders, unverifiable priors

### **16.2. Quantifying Task Similarity**
> *The metric chosen here is what § 16.5 measures when reporting forward/backward-transfer scores.*
1.  **16.2.1.** Design desiderata: transfer correlation, sample computability, invariances
2.  **16.2.2.** Distributional metrics: KL, χ², TV, Jensen–Shannon; Wasserstein & OT
3.  **16.2.3.** Dynamics-focused metrics: state/action bisimulation, successor-feature distance
4.  **16.2.4.** Representation-driven metrics: learned task embeddings, contrastive InfoNCE
5.  **16.2.5.** Empirical estimation techniques: importance-weighting, kernel-MMD, GNN graph-matching
6.  **16.2.6.** Theoretical properties: stability, invariance classes, sample-complexity lower bounds
7.  **16.2.7.** Choosing a metric in practice: alignment with § 16.4 transfer mechanism, diagnostic checklist
8.  **16.2.8.** **Value-aware & EPIC distances**: AVD, EPIC, DARD; regret and optimal-policy bounds
9.  **16.2.9.** Metric-learning pitfalls: spurious similarity under sparse rewards, over-smooth embeddings

### **16.3. Generalisation Guarantees Across Tasks**
> *Bounds here guide algorithm design in § 16.6.*
1.  **16.3.1.** PAC & PAC-Bayes refresher (single task)
2.  **16.3.2.** Meta-PAC-Bayes bounds: hierarchical priors, task-conditioned posteriors
3.  **16.3.3.** Online & lifelong regret bounds: memory-limited agents, task streams
4.  **16.3.4.** Distribution-shift compensation terms: shift-aware KL, Wasserstein corrections
5.  **16.3.5.** Information-theoretic objectives: MDL, mutual-information regularisers
6.  **16.3.6.** Lower bounds & impossibility results: adversarial tasks, negative-transfer hardness
7.  **16.3.7.** Practical implications: posterior sampling, optimism, Bayesian meta-RL recipes

### **16.4. Taxonomy of Transfer Mechanisms**
> *Each mechanism links back to similarity (§ 16.2) and forward to evaluation (§ 16.5).*
1.  **16.4.1.** Four transferable objects: representation ϕ, dynamics P, policy π, reward R
2.  **16.4.2.** Representation transfer: SSL pre-training, successor features, Lipschitz guarantees
3.  **16.4.3.** Dynamics transfer: latent SSMs, simulators, robust MPC residuals
4.  **16.4.4.** Policy transfer: warm-starts, option libraries, distillation vs ensembling
5.  **16.4.5.** Reward & preference transfer: potential-based shaping, inverse RL, RLHF preference reuse
6.  **16.4.6.** Hybrid & hierarchical transfer: joint ϕ + π, meta-optimisation across levels
7.  **16.4.7.** Negative transfer diagnostics: covariate shift, entangled dynamics-reward, goal mis-generalisation
8.  **16.4.8.** Safety, fairness & privacy: robustness, demographic parity across tasks, data-privacy in meta-datasets
9.  **16.4.9.** Case studies: sim-to-real manipulation, multi-game agents, personalised tutoring
10. **16.4.10.** Unsupervised skill discovery & autonomous RL: DIAYN, APS, unsupervised RL pre-training
11. **16.4.11.** Multi-agent & LLM-augmented transfer: opponent modelling, emergent communication, LLM tool-use

### **16.5. Evaluation Protocols & Metrics**
> *Standardises what practitioners must report; cross-links to theory (§ 16.3) and ethics (§ 16.4.8).*
1.  **16.5.1.** Performance metrics: jump-start, asymptotic gain, forward/backward transfer
2.  **16.5.2.** Efficiency metrics: sample complexity, wall-clock, **energy / CO₂ cost**
3.  **16.5.3.** Continual-learning metrics: forgetting rate, knowledge-retention curves
4.  **16.5.4.** Statistical methodology: hierarchical bootstrap, effect sizes, confidence intervals
5.  **16.5.5.** Benchmark suites: Meta-World+, MT10/50, Procgen, RL-Unplugged-Meta
6.  **16.5.6.** Experimental design standards: paired seeds, hyper-parameter transparency
7.  **16.5.7.** Visualisation best practices: CI ribbons, FT/BT heat-maps, Pareto fronts
8.  **16.5.8.** Task-difficulty normalisation & scoring
9.  **16.5.9.** Responsible-AI metrics: safety violations, fairness gaps, privacy leakage
10. **16.5.10.** Reproducibility check-lists & badges (ICML/NeurIPS 2025 requirements)

### **16.6. Algorithmic Frameworks for Transfer & Meta-RL**
- **A. Model-Free Meta-RL**
    - A.1. Gradient-based (MAML, Reptile, ANIL)
    - A.2. Memory-based (RL², SNAIL, Meta-GRU)
    - A.3. Exploration-driven (MAESN, E-MAML, E3B)
- **B. Model-Based & World-Model Meta-RL**
    - B.1. Dreamer variants, PlaNet-meta, MBRL-in-context
    - B.2. Latent dynamics adaptation (E2C-meta, RSSM++), safety-filtered MPC
    - B.3. Skill-conditional world models & zero-shot planning
- **C. Sequence Models & RLHF Transfer**
    - C.1. Decision Transformers & Trajectory Diffusion Models
    - C.2. In-context reinforcement learning with large sequence models
    - C.3. RL from Human Feedback as a transfer pipeline (preference reuse, value alignment)
- **D. Curriculum & Task-Sequencing Algorithms**
    - D.1. Domain randomisation curricula
    - D.2. Level replay & PLR
    - D.3. Bayesian curriculum shaping
- **E. Skill Discovery & Option Libraries** (links to § 16.4.10)
    - E.1. DIAYN, CIC, APS
    - E.2. Transferable option critiquing
    - E.3. Skill distillation
- **F. Safety- & multi-objective-aware variants**
    - F.1. Constrained meta-RL
    - F.2. Distributionally robust baselines
    - F.3. Fairness-regularised objectives

### **16.7. Applications & Case Studies**
1.  **16.7.1.** Robotics (sim-to-real, multi-skill)
2.  **16.7.2.** Multilingual dialogue & NLP
3.  **16.7.3.** Healthcare personalisation
4.  **16.7.4.** Autonomous driving & fleet learning
5.  **16.7.5.** Game playing & procedural generalisation
6.  **16.7.6.** Finance & energy grids optimisation
7.  **16.7.7.** Multi-agent simulation & LLM-based agents

### **16.8. Open Problems & Future Directions**
- **16.8.1.** Lifelong meta-learning under non-stationary drift
- **16.8.2.** Causality-aware task transfer
- **16.8.3.** Data-efficient world-model learning
- **16.8.4.** Multi-objective safety-fairness trade-offs
- **16.8.5.** Benchmark-to-real gap measurement & fidelity
- **16.8.6.** Policy implications and societal impact

### **16.9. Summary & Further Reading**
- Key take-aways per section
- Annotated bibliography (classic + 2024-25 papers)
- Exercises, discussion prompts

### **Cross-Reference Map (examples)**

| Concept | Defined in | Evaluated/measured in |
| :--- | :--- | :--- |
| Equivalent-Policy Invariant Comparison (EPIC) | § 16.2.8 | § 16.5.1, 16.5.4 |
| Lifelong regret bound | § 16.3.3 | § 16.5.3 |
| Unsupervised skill library | § 16.4.10 | § 16.6.E, 16.5.2 |

---

## **17. Meta-Reinforcement Learning**

### **17.1. Foundations & Problem Formulation**
- **17.1.1.** Historical context and motivations
- **17.1.2.** Formal bilevel objective (inner/outer loop)
- **17.1.3.** Task-distribution modelling (meta-train/meta-test)
- **17.1.4.** Adaptation protocols (few-shot, online, one-shot)
- **17.1.5.** Evaluation metrics and baseline taxonomy

### **17.2. Theoretical Guarantees**
- **17.2.1.** Exploration–adaptation trade-off
- **17.2.2.** Minimax, Bayesian, and PAC-Bayes regret bounds
- **17.2.3.** Information-theoretic lower bounds
- **17.2.4.** Constrained and safe-cost regret
- **17.2.5.** Offline generalisation theory
- **17.2.6.** Multi-agent meta-regret in Markov games

### **17.3. Gradient-Based and Gradient-Free Meta-Learners**
- **17.3.1.** **MAML** family (**FOMAML**, **Reptile**, **ANIL**)
- **17.3.2.** Learn-to-Optimise approaches (LSTM/Transformer optimisers)
- **17.3.3.** Gradient-free and evolution strategies (**CMA-ES**, PBT, ARS)
- **17.3.4.** Meta-policy-gradient formulations
- **17.3.5.** Hyper-parameter search and BO-MRL

### **17.4. Recurrent & Transformer In-Context Meta-Learners**
- **17.4.1.** **RL²**, **L2RL**, and other RNN agents
- **17.4.2.** Memory-augmented architectures (NTM, DND)
- **17.4.3.** Sequence-model meta-RL (**Decision Transformer-Meta**)
- **17.4.4.** Hyper-networks and meta-controllers

### **17.5. Latent-Variable & Uncertainty-Aware Methods**
- **17.5.1.** Latent-context MDP formalism
- **17.5.2.** Posterior sampling and information-bottleneck encoders
- **17.5.3.** **PEARL**, **VariBAD**, **Bayes-Adapt**
- **17.5.4.** Context-VAE / latent-PPO & SAC families

### **17.6. Model-Based Meta-RL**
- **17.6.1.** **MB-MAML** and RLG-MAML
- **17.6.2.** Latent dynamics meta-learning (**Meta-PlaNet**, **Dreamer-Meta**)
- **17.6.3.** Planning-as-Inference and world-model adaptation

### **17.7. Offline & Dataset-Driven Meta-RL**
- **17.7.1.** Problem definition: learning from multi-task logs
- **17.7.2.** Conservative and pessimistic offline meta-RL algorithms
- **17.7.3.** Offline-to-online fine-tuning hybrids

### **17.8. Safe & Constrained Meta-RL**
- **17.8.1.** Safety-constrained objectives and risk budgets
- **17.8.2.** Dual-method safety (primal-dual, Lagrangian)
- **17.8.3.** Masked-action and shielded fine-tuning

### **17.9. Multi-Agent Meta-RL**
- **17.9.1.** Cooperative, competitive, and mixed-motivation settings
- **17.9.2.** Meta-learning of communication protocols
- **17.9.3.** Benchmarks: **Meta-SMAC**, **Meta-Hanabi**

### **17.10. Benchmarks & Evaluation Protocols**
- **17.10.1.** Continuous-control: **Meta-World**, **RLBench**
- **17.10.2.** Gridworld/procedural: **MiniGrid**, **Procgen**, **Meta-Procgen**
- **17.10.3.** Embodied/driving: **Meta-Drive**, **CARLA-Meta**
- **17.10.4.** Offline/safety: **COSTA**, **D4RL-Meta**

### **17.11. Applications & Case Studies**
- **17.11.1.** Robotic manipulation and locomotion
- **17.11.2.** Personalised recommendation and dialogue
- **17.11.3.** 6G network routing and edge computing
- **17.11.4.** LLM reasoning control loops (meta-RL for language agents)

### **17.12. Practical Considerations & Governance**
- **17.12.1.** Sample-efficiency vs. compute-cost scaling laws
- **17.12.2.** Robustness to task-distribution shift
- **17.12.3.** Safety, alignment, and regulatory compliance

---

## **18. Continual Reinforcement Learning**

### **18.0. Overview & Motivation**
- **18.0.1.** Why Continual Learning in RL?
- **18.0.2.** Historical Trajectory (1990–2025)
- **18.0.3.** Terminology & Scope
    - Continual vs Lifelong vs Online vs Open-World
    - Task-, Domain-, Class-Incremental RL
- **18.0.4.** Chapter Road-Map

### **18.1. Non-Stationary Environments & Task Sequences**
- **18.1.1.** Drift Taxonomy
    - a. Abrupt / piece-wise-stationary
    - b. Gradual stochastic drift
    - c. Periodic & seasonal
    - d. Latent-context switches
- **18.1.2.** Formal Definition: Non-Stationary MDP
    - Time-indexed kernels $(P_t, R_t)$
    - Task sequences & boundaries
- **18.1.3.** Detecting Change
    - Statistical tests & surprise signals
    - Hidden-Mode MDP inference
- **18.1.4.** Task Similarity & Transferability
- **18.1.5.** Exploration–Exploitation under Drift
- **18.1.6.** Problem Variants
    - Curriculum with known order
    - Unknown boundaries (online CL)
    - Open-world / open-set RL
- **18.1.7.** Practical Assumptions & Limits
- **18.1.8.** Formalisms & Taxonomies
    - Unified notation for task/domain/class-incremental RL
- **18.1.9.** Special Settings
    - a. POMDP & latent-state drift
    - b. Hybrid Offline-to-Online Continual RL

### **18.2. Algorithmic Approaches**
- **18.2.1. Regularisation-Based Methods**
    - **18.2.1.1.** Quadratic Priors (EWC, Online Laplace)
    - **18.2.1.2.** Path-Integral & Sensitivity (SI, MAS)
- **18.2.2. Replay-Based Methods**
    - **18.2.2.1.** Experience & Reservoir Replay
    - **18.2.2.2.** Generative / Model-Based Replay
    - **18.2.2.3.** Selective & Compressed Replay (coresets, KD)
- **18.2.3. Parameter Isolation & Modular Architectures**
    - **18.2.3.1.** Dynamic Expansion (Progressive Nets, DEN)
    - **18.2.3.2.** Mask-Based Reuse (PackNet, SupSup, Piggyback)
    - **18.2.3.3.** Gated Routing / Mixture-of-Experts
- **18.2.4. Meta-Learning & Hyper-Networks**
    - **18.2.4.1.** Meta-Gradient Adaptation (MAML, OML)
    - **18.2.4.2.** Learned Optimisers & Hyper-Policies
- **18.2.5. Continual World-Models & Latent Dynamics**
    - **18.2.5.1.** Recurrent & State-Space CL models
    - **18.2.5.2.** Memory-Augmented Predictive Models
- **18.2.6. Safety-Aware Algorithms**
    - Safe replay buffers & risk-aware losses
    - Constrained RL with certificate projection
- **18.2.7. Dual-Memory & Consolidation Systems**
    - Sleep replay, dream-erasure, latent rehearsal
- **18.2.8. LLM-Augmented Continual RL**
    - Skill-language models, tool-use agents
- **18.2.9. Resource-Aware Design**
    - Capacity-compute-plasticity trade-offs

### **18.3. Evaluation Methodology**
> *Note: Moved ahead of theory section for practical flow.*
- **18.3.1. Metrics**
    - **18.3.1.1.** CL-Score (BWT, FWT, Forgetting)
    - **18.3.1.2.** Efficiency (sample, compute, energy)
    - **18.3.1.3.** Safety & Robustness
    - **18.3.1.4.** Capacity & Memory Footprint
    - **18.3.1.5.** Privacy & Fairness
- **18.3.2. Benchmarks**
    - **18.3.2.1.** Synthetic & Toy Streams
    - **18.3.2.2.** Video-Game Suites (MineRL-CL, Lifelong-ProcGen, Atari-LL)
    - **18.3.2.3.** Robotics (Meta-World-Seq, CORA, CompoSuite, Lifelong Manipulation)
    - **18.3.2.4.** Multi-Agent & Language (LifelongAgentBench)
- **18.3.3. Experimental Protocols**
    - **18.3.3.1.** Single-Pass Online Evaluation
    - **18.3.3.2.** Joint Validation after Each Task
    - **18.3.3.3.** OOD & Safety Stress-Tests
    - **18.3.3.4.** Task-Free Streams & Anytime Evaluation
- **18.3.4. Reproducibility & Tooling**
    - Drift generators, seed control
    - Logging, visualisation, leaderboards

### **18.4. Theoretical Foundations**
- **18.4.1. Stability–Plasticity Information Theory**
    - **18.4.1.1.** Information Bottleneck View
    - **18.4.1.2.** Compression-Retention Trade-off
- **18.4.2. Regret & Sample Complexity under Drift**
    - **18.4.2.1.** Path-Regret Bounds (TV-drift)
    - **18.4.2.2.** Adaptive Policy Gradient Methods
    - **18.4.2.3.** Computational Complexity & Hardness
- **18.4.3. Generalisation with Memory Constraints**
    - **18.4.3.1.** PAC-Bayesian Replay Bounds
    - **18.4.3.2.** Coreset Size vs Forgetting
- **18.4.4. Constraint Retention Guarantees**
- **18.4.5. Formal Verification & Barrier Certificates**
- **18.4.6. Lower Bounds & No-Free-Lunch Results**
- **18.4.7. Open Theoretical Questions**

### **18.5. Applications & Case Studies**
- **18.5.1.** Robotics (manipulation, navigation)
- **18.5.2.** Game AI & Procedural Content
- **18.5.3.** Autonomous Vehicles & Traffic
- **18.5.4.** Industrial Process Control
- **18.5.5.** Healthcare & Personalised Assistants
- **18.5.6.** Multi-Agent & Societal-Scale Systems
- **18.5.7.** LLM-Driven Agents & Tool Use

### **18.6. Open Challenges & Future Directions**
- **18.6.1.** Robust OOD Adaptation
- **18.6.2.** Memory-Efficient CL at Scale
- **18.6.3.** Long-Horizon Safety & Certification
- **18.6.4.** Autonomous Task Discovery
- **18.6.5.** Sim-to-Real Deployment
- **18.6.6.** Unified Theory of Transfer & Forgetting
- **18.6.7.** Human-in-the-Loop Continual RL
- **18.6.8.** Ethical, Legal & Societal Impacts
- **18.6.9.** Foundation World-Models & Pre-Training
- **18.6.10.** Compositional Generalisation & Skill Libraries

### **18.7. Chapter Summary & Further Reading**

---

## **19. Synthesis & Open Questions**
- **19.1.** Unified Bayesian perspective linking representation, exploration, transfer, and continual learning
- **19.2.** Exploration–adaptation trade-off across timescales
- **19.3.** Scalability and safety in real-world lifelong RL
- **19.4.** Major research gaps: polynomial-time exploration with latent dynamics, reliable evaluation under distribution shift
