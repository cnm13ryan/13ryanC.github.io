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

---

## **Chapter 1: MDP Foundations & Optimality**

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
- **1.4.1.** Self-normalised concentration inequalities (**Elliptical Potential**, **Bernstein**)

---

## **Chapter 2: Exact Dynamic Programming**

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
- **2.2.1.** Definition of the Policy Iteration (**PI**) algorithm
- **2.2.2.** Advantage function and the Performance-Difference Identity
- **2.2.3.** Geometric-Progress Lemma
- **2.2.4.** Geometric convergence of value error
- **2.2.5.** Strict-Progress Lemma (sub-optimal action elimination)
- **2.2.6.** Overall runtime bound (Scherrer)
- **2.2.7.** Value Iteration vs. Policy Iteration comparison
- **2.2.8.** Proof that **PI** is generally faster than **VI**
- **2.2.9.** Mixing rates and span-seminorm contraction
- **2.2.10.** Upper and lower runtime bounds (Ye; Feinberg-Huang-Scherrer)
- **2.2.11.** Measure-theoretic view of occupancy-measure projection

### **2.3. Learned-Model Dynamic Programming**
- **2.3.1.** Model-bias bounds for Bellman operators
- **2.3.2.** Ensemble variance as a proxy for model error $\varepsilon_P$
- **2.3.3.** Optimism under model uncertainty and Thompson sampling

---

## **Chapter 3: Online Planning in Discounted MDPs**

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
- **3.4.1.** Open questions and extensions (e.g., links to **MCTS**)
- **3.4.2.** Model-Predictive Control with Learned Simulators
    - **3.4.2.1.** Cross-Entropy Method (**CEM**) and trajectory optimisation
    - **3.4.2.2.** **TD-MPC** algorithm and stability analysis
    - **3.4.2.3.** Comparative runtime vs. **MCTS**

---

## **Chapter 4: Value–Function Approximation, Policy Evaluation & Iteration**
*A unified outline for a 2025 graduate‑level treatment*

### **4.0. Preliminaries & Notation**
- **4.0.1.** Problem setting: discounted MDP $\langle\mathcal S,\mathcal A,P,R,\gamma\rangle$
- **4.0.2.** Norms, concentrability, error decomposition (approximation + estimation + propagation)
- **4.0.3.** Function classes: tabular, linear, kernel/RKHS, sparse, deep nets
- **4.0.4.** Statistical toolbox: Hoeffding/Bernstein, covering numbers, Rademacher, martingales

### **4.1. Value‑Function Approximation**
- **4.1.1.** Realizability & $\varepsilon$‑universal function classes (Deadly Triad motivation)
- **4.1.2.** Monte‑Carlo evaluation & variance‑reduction (control variates, IS/WIS)
- **Temporal‑Difference family**
    - **4.1.3.** Single‑step TD (**TD(0)**, **GTD2**, **TDC**)
    - **4.1.4.** Multi‑step & eligibility‑trace TD (**TD($\lambda$)**, **Tree‑Backup**, **Retrace($\lambda$)**, **V‑trace**)
    - **4.1.5.** Variance‑reduced / emphatic TD (**ETD**, $\sigma$-$\lambda$)
- **Least‑Squares & residual methods**
    - **4.1.6.** **LSTD($\lambda$)**, **LSPE**, **LS‑TD**; incremental vs. batch solvers
    - **4.1.7.** Residual‑minimisation & saddle‑point framing (**Minimax TD**, **GTD‑MP**)
- **Extrapolation & design**
    - **4.1.8.** Extrapolation error; $\kappa$‑factor; Kiefer–Wolfowitz G‑optimality (linear critics)
    - **4.1.9.** Beyond linear: coverage coefficients, concentrability, KW limitations
- **Non‑linear critics & generalisation**
    - **4.1.10.** Neural‑network critics, NTK view, over‑parameterised convergence
    - **4.1.11.** Generalisation bounds: covering‑number & Rademacher analyses
    - **4.1.12.** Distributional value functions & risk measures (**C51**, **QR‑DQN**, **CVaR**)
    - **4.1.13.** Off‑policy value estimation & OPE (**Per‑Decision IS**, **DR**, **MAGIC**)
    - **4.1.14.** Safe / verified evaluation (Lyapunov critics, certified bounds)

### **4.2. Approximate Policy Improvement & Iteration**
- **4.2.1.** Policy‑improvement operators: greedy, $\varepsilon$‑greedy, soft‑max, entropy regularisation
- **4.2.2.** Geometric progress lemma with additive error
- **4.2.3.** API master theorem:
    $$\|V^{\pi_k}-V^*\|_\infty \le \frac{2\gamma}{(1-\gamma)^2}\varepsilon + \frac{\gamma^k}{1-\gamma}V_{\max}$$
- **4.2.4.** Classification‑based PI (**RCPI**, **DAGGER**); VC‑dimension sample bounds
- **4.2.5.** Conservative / regularised PI (**CPI**, **DPI**, **TRPO**, **MPO**)
- **4.2.6.** Least‑Squares Policy Iteration (**LSPI**): algorithm & $\tilde O\bigl(\frac{d}{(1-\gamma)^3\varepsilon^2}\bigr)$ sample bound
- **Actor–Critic family**
    - **4.2.7.** Compatible function approximation & natural gradients (**NAC**, **A3C**)
    - **4.2.8.** Deterministic Policy Gradient methods (**DDPG**, **TD3**)
    - **4.2.9.** Soft / entropy‑regularised actor–critic (**SAC**, $\alpha$‑tuning)
- **Advanced topics**
    - **4.2.10.** Distribution shift & concentrability in API; over‑estimation bias
    - **4.2.11.** Risk‑sensitive & distributional PI (**CVaR‑PG**, distortion risk)
    - **4.2.12.** Safe & verified policy improvement (Lyapunov constraints, barrier functions)

### **4.3. State & Action Abstractions**
- **4.3.1.** Motivation: sample sharing, transfer, planning acceleration
- **4.3.2.** Exact abstraction hierarchy ($\pi^*$, $Q^*$, model‑irrelevance) & theorems
- **4.3.3.** Approximate abstractions & bisimulation metrics
    - **4.3.3.1.** $\varepsilon$-$\pi^*$ & $\varepsilon$-$Q^*$ abstractions
    - **4.3.3.2.** Bisimulation, **action‑bisimulation** (Rudolph 2024), **effect‑equivalent abstraction** (Mavor‑Parker 2025)
- **4.3.4.** Value‑loss bounds; Lipschitz & pseudo‑metric analyses
- **4.3.5.** Finite‑sample complexity & encoder bias $\varepsilon_{\text{enc}}$
- **4.3.6.** Symbolic $\leftrightarrow$ learned bridging
    - **4.3.6.1.** Self‑supervised latent‑state discovery (Sobal 2025)
    - **4.3.6.2.** Iterative *refine–plan* pipeline; conjectured regret $\tilde O\bigl(H\sqrt{dT}+\frac{\varepsilon_{\text{enc}}}{1-\gamma}\bigr)$
- **4.3.7.** Action abstraction: options, skills, initiation–termination, homomorphisms
- **4.3.8.** Group structure & utile distinctions

### **4.4. Offline / Batch Reinforcement Learning**
- **4.4.1.** Motivation: safety, data reuse, distribution shift
- **4.4.2.** Formal framework: static dataset $\mathcal D$, behaviour policy $\beta$, coverage $C$
- **4.4.3.** Core control algorithms: Fitted Q‑Iteration (**FQI**), Fitted V‑Iteration
- **4.4.4.** Offline policy evaluation (**OPE**): Fitted Q Evaluation (**FQE**), **IS/WIS**, **DR**, **MAGIC**
- **4.4.5.** Error analysis pipeline: uniform deviation → one‑step error → propagation
- **4.4.6.** Pessimistic & conservative control: **CQL**, **IQL**, **OPAL**
- **4.4.7.** Distributionally‑robust offline RL: **ROMAN**, **DRO‑RL**, **OPPO**
- **4.4.8.** Model‑based offline RL: **MOPO**, **COMBO**, pessimistic model rollouts
- **4.4.9.** Robustness to corrupted or confounded logs
- **4.4.10.** Representation learning for Bellman completeness & coverage
- **4.4.11.** Benchmarks & evaluation protocols: **D4RL‑2025**, **MineRL‑Offline**, **RL‑Unplugged**

### **4.5. Latent‑World Models & Model‑Based Approximate RL**
- **4.5.1.** Variational recurrent state‑space models (**RSSM**, **VRNN**) & ELBO
- **4.5.2.** **Dreamer** family (V2, V3): latent imagination & KL balancing
- **4.5.3.** Planning in learned latent spaces: **CEM**, gradient‑based shooting, **MuZero**
- **4.5.4.** Scaling laws & generalist models (parameter–performance curves)
- **4.5.5.** Uncertainty‑aware models: ensembles, Bayesian **RSSM**, epistemic vs aleatoric split
- **4.5.6.** Hybrid model‑based / model‑free synergy: Latent **TD($\lambda$)**, cross‑consistency losses

### **4.6. Safety, Verification & Compliance**
- **4.6.1.** Safe exploration: reachability, shielded RL, conservative baselines
- **4.6.2.** Lyapunov‑based critics & barrier‑function constraints
- **4.6.3.** Verified approximate value functions (interval arithmetic, SMT)
- **4.6.4.** Risk constraints: **CVaR**, chance‑constrained RL, robust MDPs

### **4.7. Further Directions & Open Problems**
- **4.7.1.** Approximate RL in POMDPs (predictive state representations, **CSC‑RL**)
- **4.7.2.** Multi‑agent approximate RL (**CTDE**, mean‑field critics, differential games)
- **4.7.3.** Scalable exploration & representation (**RND**, Laplacian embeddings, intrinsic control)
- **4.7.4.** Hardware & systems (neural compression, GPU/TPU kernels for LS methods)
- **4.7.5.** Human feedback & reward modelling (**RLHF**, preference‑based critics)

---

## **Chapter 5: Sampling and Computational Complexity**

### **5.1. Foundations & State-Representation Learning (SRL)**
- **5.1.1.** Markov Decision Processes (recap)
- **5.1.2.** Feature maps $\phi$: from hand-crafted bases to learned encoders $\phi_\psi$
- **5.1.3.** Six SRL families (metric, contrastive, augmentation, world-model, reconstruction, auxiliary-task)
- **5.1.4.** Evaluation protocols (**DMControl-100k**, **Atari-100k**, **ProcGen**)
- **5.1.5.** Norms and error metrics ($\|\cdot\|_\infty$, $\|\cdot\|_2$, bisimulation-metric)
- **5.1.6.** Pre-computed core sets and **LSPI** recap

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

## **Chapter 6: Robust & Safe Model‑Based Reinforcement Learning**
*Comprehensive Table of Contents (July 2025)*

### **6.0. Preface & Problem Setting**
- **6.0.1.** Motivation, Applications, Stakes
- **6.0.2.** Scope: Model‑Free vs Model‑Based Robust/Safe RL
- **6.0.3.** Notation & Symbol Table
- **6.0.4.** Running Example Preview (Quadrotor in Gusty Wind)

### **6.1. Mathematical Foundations**
- **6.1.1.** Markov Decision Processes & Constrained MDPs Recap
- **6.1.2.** Types of Uncertainty (Epistemic / Aleatoric; Parametric / Structural)
- **6.1.3.** Safety Formalisms (Hard constraints, Reach‑Avoid, Viability, Risk/CVaR)
- **6.1.4.** Robustness Paradigms (Worst‑Case, Bayesian, DRO, Risk‑Sensitive)
- **6.1.5.** Robust POMDPs & Belief‑Space Safety
    - *Implementation:* Particle‑filter belief updates; Chance‑constrained belief MPC

### **6.2. Model Learning under Uncertainty**
- **6.2.1.** Parametric System Identification
    - *Algorithms:* Least‑Squares, Koopman lifting, NN dynamics
- **6.2.2.** Bayesian & Ensemble Methods
    - *Algorithms:* Gaussian Processes, BNNs, Deep Ensembles
- **6.2.3.** Out‑of‑Distribution & Covariate‑Shift Detection
- **6.2.4.** Finite‑Sample & Uniform‑Convergence Guarantees
- **6.2.5.** Partial‑Observation Model Learning
    - *Algorithms:* Variational Auto‑Encoders for latent state discovery

### **6.3. Distributionally Robust Planning**
- **6.3.1.** Ambiguity‑Set Construction ($f$-divergence, Wasserstein, Moment balls)
- **6.3.2.** Robust Bellman Operators & Dynamic Programming
- **6.3.3.** Solution Algorithms
    - *Algorithms:* Robust Value/Policy Iteration; DRO Linear Programming; Ambiguous Policy Gradient
- **6.3.4.** Sample‑Complexity & Performance Bounds
- **6.3.5.** Interplay with Risk Metrics

### **6.4. Risk, Adversarial & Uncertainty Quantification**
- **6.4.1.** Epistemic–Aleatoric Decomposition
- **6.4.2.** Coherent Risk Measures (Variance, Entropic, CVaR, Spectral)
- **6.4.3.** Distributional RL & Quantile Regression
- **6.4.4.** Uncertainty‑Aware Exploration vs Pessimism
- **6.4.5.** Adversarial Perturbations & Security Robustness
    - *Topics:* Policy poisoning, reward hacking, robust training defenses

### **6.5. Adaptive & Meta‑Adaptive Control**
- **6.5.1.** Online Parameter Estimation Laws
- **6.5.2.** $\mathcal{L}_1$ Adaptive Control Architecture
    - *Implementation:* Low‑pass filter tuning; Robust stability proofs
- **6.5.3.** Bounded Tracking‑Error Theorems
- **6.5.4.** Integration with RL Critics / Actors
- **6.5.5.** Meta‑Adaptive Safe Control
    - *Topics:* Rapid system‑ID, context‑based policy adaptation

### **6.6. Safe Model‑Predictive Control (MPC)**
- **6.6.1.** Nominal, Robust & Stochastic MPC Taxonomy
- **6.6.2.** Tube MPC & Invariant/Terminal Sets
- **6.6.3.** Chance‑Constrained & CVaR‑MPC
- **6.6.4.** Constraint Tightening with Learned Uncertainty
- **6.6.5.** Real‑Time Optimisation & Warm‑Start Strategies

### **6.7. Verification, Monitoring & Runtime Assurance**
- **6.7.1.** Formal Specification Languages (LTL, STL, BLTL)
- **6.7.2.** Reachability Analysis & Barrier Certificates
- **6.7.3.** Shielding & Runtime Enforcement
- **6.7.4.** Online Falsification & Counterexample Search
- **6.7.5.** Fault Detection, Diagnosis & Recovery Policies

### **6.8. Safe & Sample‑Efficient Exploration**
- **6.8.1.** Confidence‑Based Safe Exploration (**PO‑CPS**, **C‑CBF**)
- **6.8.2.** Opt‑in‑Uncertain vs Conservative Pessimism Trade‑off
- **6.8.3.** Exploration in Continuous‐Control & High‑Dim Vision

### **6.9. Offline Robust & Safe MBRL** *(New major section)*
- **6.9.1.** Dataset Shift, Coverage & Support Mismatch
- **6.9.2.** Offline Constrained & Distributionally Robust RL Algorithms
    - *Algorithms:* **CAPS**, **ROOM**, **ROAM**, **OGSRL**
- **6.9.3.** Heavy‑Tailed Rewards & Trajectory‑Level Safety
- **6.9.4.** Offline‑to‑Online Fine‑Tuning with Safety Guarantees
- **6.9.5.** Benchmark Suites for Offline Safety (e.g., **OSRL‑Bench**)

### **6.10. Implementation & Scalability**
- **6.10.1.** Software Frameworks & Auto‑Diff Toolchains
    - *Tools:* PyTorch, JAX, CasADi, ACADOS
- **6.10.2.** Parallel DP & GPU/TPU Acceleration
- **6.10.3.** Numerical Robustness & Debugging Safety Violations
- **6.10.4.** Resource‑Aware Real‑Time Deployment

### **6.11. Multi‑Agent Robust & Safe RL** *(Promoted section)*
- **6.11.1.** SafeMARL Formulations (**CMDP‑MARL**, Mean‑Field)
- **6.11.2.** Robust Coordination & Communication under Uncertainty
- **6.11.3.** Adversarial/Competitive Safety in MARL
- **6.11.4.** Distributed Verification & Assurance
- **6.11.5.** Scalable Multi‑Robot Case Studies

### **6.12. Evaluation & Benchmarks**
- **6.12.1.** Simulation Suites (**RobustRL‑Bench**, **Safe‑Gymnasium**, **DMC**)
- **6.12.2.** Physical Testbeds (Quadrotors, Autonomous Racing, Micro‑grids)
- **6.12.3.** Metrics & Reporting Standards (Safety Rate, Robustness Gap)
- **6.12.4.** Reproducibility Checklists & Open‑Source Tooling

### **6.13. Case Studies**
- **6.13.1.** Quadrotor Navigation in Wind Gusts
- **6.13.2.** Autonomous Racing with Uncertain Grip
- **6.13.3.** Industrial Process Control under Sensor Noise
- **6.13.4.** Energy Micro‑grids with Demand Spikes
- **6.13.5.** Medical Treatment Planning with Patient Variability

### **6.14. Open Challenges & Future Directions**
- **6.14.1.** Formal Verification at Scale
- **6.14.2.** Lifelong Learning & Continual Robustness
- **6.14.3.** Human‑in‑the‑Loop Trust & Preference Integration
- **6.14.4.** Fairness‑Aware Safe MBRL
- **6.14.5.** Privacy‑Preserving Robust RL
- **6.14.6.** Regulatory, Ethical & Societal Implications
- **6.14.7.** Multi‑Modal & Foundation‑Model‑Based Control

---

## **Chapter 7: Model‑Free Value Prediction**
*An outline integrating on/off‑policy learning, multi‑step returns, eligibility traces, least‑squares methods, function approximation, average‑reward theory, distributional objectives, OPE, and finite‑sample guarantees.*

### **7.1. Foundations**
- **7.1.1.** MDP formalism, notation, and Markov property
- **7.1.2.** Task regimes: episodic, continuing (discounted), and average‑reward
- **7.1.3.** Returns: $G_t^{\gamma}$, bias function, and undiscounted/average formulations
- **7.1.4.** State‑ and action‑value functions; prediction vs control
- **7.1.5.** Mean‑squared value error, projected Bellman equation, MSPBE
- **7.1.6.** Data‑generation assumptions: on‑policy vs off‑policy (behavior & target)
- **7.1.7.** Function approximation preliminaries (feature maps, parametric models)

### **7.2. Function‑Approximation Fundamentals**
- **7.2.1.** Linear approximation and feature engineering
- **7.2.2.** Non‑linear approximation: kernels, neural networks, transformer critics
- **7.2.3.** Semi‑gradient vs true‑gradient methods; target networks & stabilization
- **7.2.4.** Representation learning and Bellman‑complete features
- **7.2.5.** Divergence pathologies (Baird’s star, deadly triad)

### **7.3. Monte‑Carlo Prediction**
- **7.3.1.** First‑visit vs every‑visit estimators; incremental sample averages
- **7.3.2.** Variance analysis of full‑return MC
- **7.3.3.** Continuing‑task and average‑reward MC variants
- **7.3.4.** Off‑policy MC with importance sampling (Ordinary & weighted IS, Per‑decision IS, Weighted PDIS)
- **7.3.5.** Confidence intervals & empirical Bernstein bounds

### **7.4. One‑Step Temporal‑Difference Learning**
- **7.4.1.** **TD(0)** target, TD‑error, on‑line update rule
- **7.4.2.** Bias–variance comparison: DP $\leftrightarrow$ TD $\leftrightarrow$ MC
- **7.4.3.** Tabular convergence and contraction argument
- **7.4.4.** Average‑reward TD (**RVI‑TD**, bias‑function estimation)

### **7.5. Multi‑Step Returns & Eligibility Traces**
- **7.5.1.** $n$‑step TD derivation and algorithm family
- **7.5.2.** $\lambda$‑return (forward view) and **TD($\lambda$)**
- **7.5.3.** Backward view: eligibility traces (Accumulating vs replacing, True‑online **TD($\lambda$)**, **GAE**)
- **7.5.4.** Choosing $n$ or $\lambda$; analytical bias–variance curves

### **7.6. Least‑Squares & Low‑Rank Batch Methods**
- **7.6.1.** Least‑Squares TD (**LSTD**) and **LS‑Sarsa** derivations
- **7.6.2.** Incremental / low‑rank **LSTD** for high‑dimensional features
- **7.6.3.** Connections to Kalman filtering and Gauss–Newton
- **7.6.4.** Memory–computation trade‑offs and regularization

### **7.7. Average‑Reward Prediction**
- **7.7.1.** Bias‑function formulation and relative value operator
- **7.7.2.** **RVI‑TD**, differential TD, and log‑average‑reward TD
- **7.7.3.** Multi‑step & $\lambda$ extensions for $\gamma \neq 1$
- **7.7.4.** Convergence proofs and finite‑sample rates

### **7.8. Off‑Policy TD & Variance‑Reduction**
- **7.8.1.** IS‑corrected **TD($\lambda$)**: ordinary, per‑decision, and weighted
- **7.8.2.** Emphatic TD (**ETD**, **ETD($\lambda$)**)
- **7.8.3.** Gradient‑TD family (**GTD**, **GTD2**, **TDC**, **PG‑TD**)
- **7.8.4.** Safe return‑based algorithms: **Retrace($\lambda$)**, $Q^\pi(\lambda)$, **V‑trace**
- **7.8.5.** Doubly‑robust TD and control‑variate techniques
- **7.8.6.** Latest convergence results for off‑policy multi‑step TD

### **7.9. Distributional & Risk‑Sensitive Prediction**
- **7.9.1.** Distributional Bellman operator; categorical & quantile TD
- **7.9.2.** Mixture‑of‑quantiles and implicit quantile networks (**IQN‑TD**)
- **7.9.3.** Finite‑sample guarantees for distributional TD
- **7.9.4.** Risk‑aware objectives: entropic, **CVaR**, and coherent measures
- **7.9.5.** **CVaR‑TD**, risk‑sensitive **GAE**, and policy‑gradient links

### **7.10. Offline Policy Evaluation (OPE)**
- **7.10.1.** Problem setup and diagnostics for dataset shift
- **7.10.2.** Importance‑sampling OPE: **WIS**, **PDIS**, **CWPDIS**
- **7.10.3.** Doubly‑robust, **MAGIC**, **OPERA**, and **CEOPL** estimators
- **7.10.4.** Model‑based OPE and Fitted Q‑Evaluation (**FQE**)
- **7.10.5.** High‑confidence / PAC and bootstrap intervals
- **7.10.6.** Benchmarking suites and practical pitfalls

### **7.11. Theoretical Analysis & Guarantees**
- **7.11.1.** Bellman operator contraction & projected fixed points
- **7.11.2.** Stochastic approximation: Robbins–Monro and ODE methods
- **7.11.3.** Asymptotic convergence (tabular & linear FA)
- **7.11.4.** Finite‑sample and high‑probability error bounds
- **7.11.5.** Divergence counter‑examples and negative results
- **7.11.6.** Stability proofs for **ETD**, gradient‑TD, and off‑policy $n$‑step TD
- **7.11.7.** Complexity lower bounds and ‘no‑free‑lunch’ theorems

### **7.12. Practical Implementation & Engineering**
- **7.12.1.** Step‑size schedules: adaptive (**Adam**, **RMSProp**) vs constant‑$\alpha$
- **7.12.2.** Feature normalization, reward scaling, and clipping
- **7.12.3.** Trace management: resetting, $\lambda$‑sweeps, **ETD** emphasis decay
- **7.12.4.** Replay buffers, prioritization, and sample‑reuse gearing
- **7.12.5.** Parallel rollout & hardware acceleration (GPU/JAX/TPU)
- **7.12.6.** Reproducibility checklists and open‑source libraries

### **7.13. Summary & Further Reading**
- **7.13.1.** Comparative algorithm table and bias–variance map
- **7.13.2.** Open research questions (variance‑reduced TD, OPE under deep FA, risk‑aware guarantees)
- **7.13.3.** Annotated bibliography: core textbooks, surveys, and seminal papers (2018–2025)

---

## **Chapter 8: Model‑Free Control (On‑ & Off‑Policy)**
*A rigorously structured roadmap reflecting both classical theory and contemporary advances.*

### **8.0. Preliminaries & Notation**
- **8.0.1.** MDP formalism, trajectories, data regimes (online / offline)
- **8.0.2.** Return definitions: discounted, average‑reward, episodic, undiscounted continuing
- **8.0.3.** On‑ vs. behaviour‑policy terminology; importance sampling ratios
- **8.0.4.** Error decomposition: approximation $\leftrightarrow$ estimation $\leftrightarrow$ optimisation

### **8.1. Tabular On‑Policy Control**
- **8.1.1.** One‑step **Sarsa**: update, GLIE condition, convergence theorem
- **8.1.2.** **Expected Sarsa**: bias–variance analysis
- **8.1.3.** Multi‑step extensions: **Sarsa($\lambda$)**, true‑online variants
- **8.1.4.** Average‑reward / differential **Sarsa** for continuing tasks

### **8.2. Tabular Off‑Policy Control**
- **8.2.1.** Watkins' **Q‑learning**: update, contraction proof
- **8.2.2.** Non‑asymptotic sample complexity: $\tilde{O}\bigl(SA/(1-\gamma)^{4}\varepsilon^{2}\bigr)$
- **8.2.3.** **Double Q‑learning**: maximisation‑bias correction
- **8.2.4.** Safe‑target algorithms: **Expected Q‑learning**, **Tree‑Backup**, **Retrace**, **V‑trace**

### **8.3. Finite‑Sample Theory**
- **8.3.1.** PAC‑MDP and worst‑case regret frameworks
- **8.3.2.** Upper bounds for **Q‑learning** + UCB exploration
- **8.3.3.** Matching lower bounds and minimax gaps
- **8.3.4.** Linear‑MDP and low‑rank structure reductions

### **8.4. Function Approximation Fundamentals**
- **8.4.1.** Linear architectures: feature coverage, **LSTD‑Q**, LQR corner cases
- **8.4.2.** Divergence pathology: **Baird’s counter‑example**
- **8.4.3.** **Gradient‑TD** family (**GTD2**, **TDC**, **GQ($\lambda$)**); projected Bellman operator
- **8.4.4.** Convergence rates under mixing & concentrability assumptions

### **8.5. Variance Reduction & Regularisation**
- **8.5.1.** **SVRG‑TD** / **VR‑R‑TD**: theoretical guarantees
- **8.5.2.** **VRCQ**, **RegQ**, and other linear‑FA algorithms with $\ell_\infty$ contraction
- **8.5.3.** Operator regularisation: entropy, proximal & mirror‑descent views
- **8.5.4.** Robust MDPs and adversarial reward perturbations

### **8.6. Deep Value‑Based Control**
- **8.6.1.** **DQN** core: replay buffer, target network, $\varepsilon$‑greedy
- **8.6.2.** Rainbow components: **Double‑DQN**, Dueling nets, prioritized replay, distributional targets, Noisy Nets, n‑step returns
- **8.6.3.** Stability heuristics: loss clipping, gradient noise scale
- **8.6.4.** Benchmarks & sample‑efficiency trends (**Atari 57** → **Crafter/Procgen**)

### **8.7. Representation Learning & State Abstraction**
- **8.7.1.** Auxiliary‑task pipelines: **ATC**, **CURL**, **BYOL‑Expl**
- **8.7.2.** Invariant & contrastive features; links to bisimulation metrics
- **8.7.3.** Impact on sample complexity and transfer

### **8.8. Hierarchical & Temporally‑Abstract Control**
- **8.8.1.** Options framework & semi‑MDPs
- **8.8.2.** **Option‑Critic**, **HIRO**, **HAC**; policy‑gradient variants
- **8.8.3.** Skill discovery and curriculum learning

### **8.9. Actor‑Critic & Policy‑Gradient Methods**
- **8.9.1.** Policy‑gradient theorem; variance reduction via **GAE($\lambda$)**
- **8.9.2.** On‑policy actors: **REINFORCE**, **A2C/A3C**, **TRPO**, **PPO**
- **8.9.3.** Off‑policy deterministic and stochastic actors
    - **8.9.3.1.** **DDPG**, **TD3** (twin critics, delayed policy)
    - **8.9.3.2.** **SAC**: entropy‑regularised objective, temperature adaptation
- **8.9.4.** Natural‑gradient and mirror‑descent actor updates

### **8.10. Objectives & Evaluation Criteria**
- **8.10.1.** Distributional RL: Bellman operator geometry, **C51**, **QR‑DQN**, **IQN**
- **8.10.2.** Risk‑sensitive control: **CVaR**, entropic, coherent risk measures
- **8.10.3.** Multi‑objective RL: scalarisation, Pareto fronts

### **8.11. Exploration Strategies**
- **8.11.1.** Count‑based & pseudo‑count bonuses
- **8.11.2.** Optimism & posterior sampling: **UCB‑Q**, **PSRL**
- **8.11.3.** Intrinsic‑motivation signals: **RND**, **ICM**, novelty search
- **8.11.4.** Directed exploration in continuous action spaces

### **8.12. Multi‑Agent & Game‑Theoretic Control**
- **8.12.1.** Independent learners and instability pitfalls
- **8.12.2.** Centralised‑training / decentralised‑execution (**CTDE**): **QMIX**, **MADDPG**
- **8.12.3.** Convergence and regret in MARL; mean‑field approximations

### **8.13. Offline (Batch) RL**
- **8.13.1.** Distribution‑shift & extrapolation error
- **8.13.2.** Conservative algorithms: **CQL**, **AWR**, **IQL**, **BRAC**
- **8.13.3.** Behaviour‑regularisation and pessimistic lower‑bounds
- **8.13.4.** Theoretical guarantees: concentrability coefficients

### **8.14. Practical Engineering & Hyper‑Parameter Guidelines**
- **8.14.1.** Replay‑buffer design: prioritisation, reservoir, segment trees
- **8.14.2.** Target‑network cadence & Polyak averaging
- **8.14.3.** Normalisation tricks: rewards, returns, layer‑norm
- **8.14.4.** Distributed actor‑learner architectures (**IMPALA**, **R2D2**)
- **8.14.5.** Large‑batch & data‑augmentation pipelines for pixel control

### **8.15. Robustness, Risk & Safety**
- **8.15.1.** Adversarial robustness: worst‑case perturbation bounds
- **8.15.2.** Safe exploration: shielded RL, Lyapunov‑based constraints
- **8.15.3.** Catastrophic‑action avoidance and runtime monitoring

### **8.16. Emerging Directions & Open Questions**
- **8.16.1.** Non‑linear TD theory: NTK & beyond
- **8.16.2.** Provably efficient exploration in continuous control
- **8.16.3.** Unified frameworks bridging online, offline & meta‑RL
- **8.16.4.** Benchmarking reproducibility and evaluation standards

---

## **Chapter 9: Off-Policy Learning: Prediction & Control**

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
- **9.5.1.** Cross-reference: Fitted Q-Iteration (see §4.4)
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

## **Chapter 10: Policy Search & Policy‑Gradient Methods**

### **10.0. Scope, Notation, and Historical Context**
- **10.0.1.** RL problem statement and objectives
- **10.0.2.** Policy‑search taxonomy (direct vs gradient‑based vs hybrid)
- **10.0.3.** Milestones in policy gradients (1992–2025)
- **10.0.4.** Common symbols, environments, and evaluation conventions

### **10.1. Gradient‑Estimation Fundamentals**
- **10.1.1.** Likelihood‑ratio / score‑function estimator
- **10.1.2.** Pathwise (reparameterisation) gradient
- **10.1.3.** Finite‑difference & simultaneous‑perturbation estimators
- **10.1.4.** Bias–variance trade‑offs and Cramér–Rao limits
- **10.1.5.** Variance‑reduced score‑function PG (**RELAX**, **REBAR**, **DiCE**)

### **10.2. Monte‑Carlo Policy Gradient Methods**
- **10.2.1.** **REINFORCE** and reward‑to‑go variants
- **10.2.2.** Baselines & control‑variates (optimal constant, state‑value)
- **10.2.3.** Generalised Advantage Estimation (**GAE**)
- **10.2.4.** Batch vs online PG; return normalisation and whitening

### **10.3. Actor–Critic & Natural Policy Gradient**
- **10.3.1.** Actor–critic separation and bootstrapped TD critics
- **10.3.2.** Compatible function approximation & Natural Policy Gradient (**NPG**) derivation
- **10.3.3.** **TD($\lambda$)** critics, eligibility traces, bias–variance control
- **10.3.4.** Asynchronous & distributed actor‑critics (**A3C**, **IMPALA**, **R2D2**)

### **10.4. KL‑Constrained & Proximal Policy Optimisation**
- **10.4.1.** Trust‑region view of **NPG**; Fisher geometry & KL constraints
- **10.4.2.** Trust‑Region Policy Optimisation (**TRPO**)
- **10.4.3.** Proximal Policy Optimisation (**PPO**): clip vs adaptive KL penalty
- **10.4.4.** Maximum‑likelihood & KL‑projection methods (**V‑MPO**, **MPO‑Q**)
- **10.4.5.** Curvature‑aware PG: **K‑FAC**, **Shampoo**, **L‑BFGS‑PG**

### **10.5. Deterministic & Delayed Policy Gradients**
- **10.5.1.** Deterministic Policy Gradient (**DPG**) theorem
- **10.5.2.** Deep DPG (**DDPG**): target networks, experience replay
- **10.5.3.** Twin‑Delayed DPG (**TD3**) & policy smoothing
- **10.5.4.** Distributional & ensemble variants (**D4PG**, **ED2**)

### **10.6. Entropy‑Regularised & Maximum‑Entropy RL**
- **10.6.1.** Entropy bonuses, temperature schedules, exploration incentives
- **10.6.2.** Soft Policy Iteration framework
- **10.6.3.** Soft Actor–Critic (**SAC**): twin Q critics & auto‑entropy tuning
- **10.6.4.** Energy‑based policies and stochastic relaxations

### **10.7. Off‑Policy & Hybrid Policy Gradients**
- **10.7.1.** Per‑decision importance sampling (IS) and variance control
- **10.7.2.** Truncated IS, **V‑trace**, **Retrace**, **ACER**
- **10.7.3.** **Q‑Prop** and doubly‑robust off‑policy PG
- **10.7.4.** Replay buffers, prioritised sampling, stabilisation tricks
- **10.7.5.** Hybrid offline‑enhanced PG (pre‑training with logged data)

### **10.8. Safety, Constraints, and Risk‑Sensitive PG**
- **10.8.1.** Constrained Policy Optimisation (**CPO**) & primal–dual methods
- **10.8.2.** Lyapunov‑based safe RL and barrier functions
- **10.8.3.** Risk measures: **CVaR**, variance‑constrained PG, distributional RL
- **10.8.4.** Anytime Safe PG (**RL‑SGF**, **IPO‑2**) and last‑iterate guarantees

### **10.9. Exploration & Intrinsic Motivation**
- **10.9.1.** Intrinsic‑motivation signals (curiosity, **RND**, **ICM**)
- **10.9.2.** Parameter‑space noise vs action‑space noise
- **10.9.3.** KL‑regularisation schedules for exploration stability

### **10.10. Temporal Credit Assignment & Regularisers**
- **10.10.1.** Eligibility traces, **GAE‑$\lambda$** horizon tuning
- **10.10.2.** KL & entropy regularisers as implicit credit‑shapers
- **10.10.3.** Reward shaping, potential‑based methods

### **10.11. Theoretical Analysis**
- **10.11.1.** Convergence guarantees and stability conditions
- **10.11.2.** Sample‑complexity upper bounds
- **10.11.3.** Distribution‑dependent lower bounds and impossibility results
- **10.11.4.** Mirror‑descent & policy‑iteration duality

### **10.12. Implementation Engineering & Incremental Training**
- **10.12.1.** Observation, reward, and advantage normalisation
- **10.12.2.** Learning‑rate schedules, adaptive optimisers, Polyak averaging
- **10.12.3.** Gradient clipping, orthogonal initialisation, parameter noise
- **10.12.4.** Distributed training (GPU/TPU), pipeline parallelism
- **10.12.5.** Incremental / resource‑bounded PG (**AVG**, buffer‑free updates)
- **10.12.6.** Diagnostics: KL, entropy, gradient norms, loss decomposition

### **10.13. Benchmarks & Domain‑Specific Applications**
- **10.13.1.** Classic control & continuous toy tasks
- **10.13.2.** Atari & image‑based discrete control
- **10.13.3.** MuJoCo locomotion & manipulation
- **10.13.4.** Robotics: sim‑to‑real transfer & impedance control
- **10.13.5.** Industrial systems: recommender RL, energy optimisation
- **10.13.6.** Language models & preference learning
    - **10.13.6.1.** KL‑regularised RLHF (**PPO‑RLHF**)
    - **10.13.6.2.** Direct Preference Optimisation (**DPO**)

### **10.14. Multi‑Agent, Hierarchical, and Population‑Based PG**
- **10.14.1.** Multi‑agent actor–critic frameworks
    - **10.14.1.1.** **MADDPG**, **QMIX‑PG**
    - **10.14.1.2.** **COMA** & counterfactual baselines
    - **10.14.1.3.** Learning‑aware PG (opponent‑aware gradients)
- **10.14.2.** Hierarchical PG & options
- **10.14.3.** Population‑based training and evolutionary curricula

### **10.15. Scaling Trends & Modern Variants**
- **10.15.1.** Transformer policies & memory augmentation
- **10.15.2.** Scaling laws for policy networks
- **10.15.3.** Offline policy gradients & conservative objectives
    - **10.15.3.1.** **CQL‑PG**, **IQL‑PG**
    - **10.15.3.2.** Policy‑guided offline optimisation (model‑based)
- **10.15.4.** Foundation agents & large‑action‑space optimisation
- **10.15.5.** Continual, lifelong, and open‑ended learning

### **10.16. Gradient‑Free & Evolutionary Policy Search**
- **10.16.1.** Natural Evolution Strategies (**NES**) & **OpenAI‑ES**
- **10.16.2.** Covariance‑Matrix Adaptation (**CMA‑ES**)
- **10.16.3.** Cross‑Entropy Method (**CEM**)
- **10.16.4.** Guided Policy Search (**GPS**) & hybrid supervised–RL pipelines
- **10.16.5.** Bayesian optimisation & bandit black‑box search

### **10.17. Open Problems & Future Directions**
- **10.17.1.** Long‑horizon & sparse‑reward credit assignment
- **10.17.2.** Robustness, generalisation, out‑of‑distribution shifts
- **10.17.3.** Partial observability & belief‑state PG
- **10.17.4.** Model‑based $\leftrightarrow$ policy‑gradient fusion
- **10.17.5.** Human‑in‑the‑loop alignment and ethics

---

## **Chapter 11: Partially Observable Reinforcement Learning**
*Revised, Detailed Scaffold*

### **11.1. Problem Formulation & Information States**
- **11.1.1.** Formal models: POMDP, mixed‑observability MDP, Dec‑POMDP, belief‑MDP embedding
- **11.1.2.** Information states: filtering vs. smoothing, beliefs, predictive state representations (**PSRs**), observable‑operator models (**OOMs**)
- **11.1.3.** Observability & identifiability metrics: entropy, mutual/Fisher information, structural rank
- **11.1.4.** Classes of policies: belief‑stationary, finite‑state controllers, history‑based, memory‑augmented neural policies
- **11.1.5.** Value‑function structure: PWLC for finite POMDPs; continuity & Lipschitz results in continuous spaces
- **11.1.6.** Computational complexity: PSPACE (finite POMDP), NEXP (Dec‑POMDP), undecidable hybrids
- **11.1.7.** Belief compression & amortised filtering: lossy KL projection, variational and learned auto‑encoders

### **11.2. Exact Planning in Belief Space**
- **11.2.1.** Dynamic programming: $\alpha$‑vector value iteration, incremental pruning, policy iteration
- **11.2.2.** Policy‑graph search: exhaustive controller enumeration; proofs of bounded‑memory optimality
- **11.2.3.** Special‑case tractability: deterministic transitions, tree‑structured POMDPs, mixed‑observability models
- **11.2.4.** Hardness proofs & scalability limits: tight bounds, worst‑case instance constructions
- **11.2.5.** Anytime deterministic $\alpha$‑updates: branch‑and‑bound refinements, real‑time guarantees

### **11.3. Offline / Batch Belief‑Space Planning**
- **11.3.1.** Point‑based methods: **PBVI**, **Perseus**, **HSVI**, **SARSOP**; belief‑set selection
- **11.3.2.** Sampling‑based dynamic programming: fitted $\alpha$‑iteration, **BELIEF‑QL**
- **11.3.3.** Function approximation: linear bases, RKHS, deep neural surrogates
- **11.3.4.** Continuous‑state solvers: local linearisation, Gaussian‑belief DDP, variational approaches
- **11.3.5.** Error analysis & anytime bounds: contraction arguments, Bellman residual control

### **11.4. Online / Real‑Time Planning**
- **11.4.1.** Rollout & QMDP heuristics: one‑step look‑ahead, hindsight Q‑updates
- **11.4.2.** Heuristic search in belief trees: **RTBSS**, **AEMS**, **LAO\*** adaptations
- **11.4.3.** Discrete Monte‑Carlo tree search: **POMCP**, **DESPOT**, virtual loss tricks
- **11.4.4.** Continuous‑space MCTS variants: **POMCPOW**, **PFT‑DPW**, weighted particle backups
- **11.4.5.** Interruptible & anytime guarantees: bounded‑sub‑optimality under compute budgets

### **11.5. Latent Dynamics & Observation‑Model Learning**
- **11.5.1.** Maximum‑likelihood & EM: Baum–Welch generalisations, stabilisation techniques
- **11.5.2.** Spectral & low‑rank PSR estimation: method‑of‑moments, **UCB‑PSR** sample‑complexity
- **11.5.3.** Bayesian non‑parametric inference: **HDP‑HMM**, Dirichlet‑process POMDPs
- **11.5.4.** Neural world models: recurrent state‑space models, contrastive latent dynamics, **Dreamer‑P**
- **11.5.5.** Joint model‑learning + planning (Dyna‑style): imagination rollout, model‑error mitigation

### **11.6. Representation & Memory**
- **11.6.1.** Belief compression & smoothing: lossy projections, neural importance sampling, amortised filters
- **11.6.2.** Learned latent embeddings: contrastive/**BYOL** objectives, masked modelling for belief sharpening
- **11.6.3.** Recurrent & transformer architectures: LSTM/GRU, **GTrXL**, **RWKV**, memory‑efficiency tricks
- **11.6.4.** Memory‑capacity theory: finite‑state controller bounds, windowed‑history PAC rates
- **11.6.5.** Causal representation learning: latent‑cause discovery, intervention‑aware policies

### **11.7. Online Model‑Free RL under Partial Observability**
- **11.7.1.** Recurrent Q‑learning: **DRQN**, **R2D2/R2D3**, auxiliary loss variants
- **11.7.2.** Actor–critic with memory: **A2C‑LSTM**, **IMPALA‑RNN**, Recurrent **PPO**
- **11.7.3.** Deterministic & distributional methods: Recurrent **SAC**, **D4PG‑RNN**
- **11.7.4.** Transformer‑based agents: **GTrXL‑RL**, Advantage Transformer, linear‑attention models
- **11.7.5.** Self‑supervised auxiliaries: **CPC**, **BYOL‑Explore**, world‑model consistency

### **11.8. Offline & Dataset‑Driven RL**
- **11.8.1.** Behaviour cloning & sequence modelling: **Decision Transformer**, **DTQN**, **V‑MPO**
- **11.8.2.** Conservative & regularised methods: **CQL‑RNN**, **BCQ‑RNN**, implicit Q‑learning with masks
- **11.8.3.** Provably efficient offline POMDP RL: **U‑PAC‑UCLK**, batch **PSRL**, pessimistic latent‑model Q‑learning
- **11.8.4.** Offline exploration & coverage diagnostics: density‑ratio tests, support mismatch bounds
- **11.8.5.** Zero‑shot & OOD generalisation: latent adversarial test‑beds, policy‑conditioned models

### **11.9. Exploration, PAC & Regret Guarantees**
- **11.9.1.** Information‑theoretic bonuses: predictive entropy, curiosity via prediction gain
- **11.9.2.** Optimism in belief space: **BA‑POMDP**, **EULER‑POMDP**, **KL‑UCRL**
- **11.9.3.** PAC sample‑complexity results: **UCFH‑POMDP**, uniform‑PAC with sliding windows
- **11.9.4.** Regret upper bounds: Thompson‑sampling **PSRL**, **O‑UCB‑POMDP**
- **11.9.5.** Regret lower bounds & revealing‑POMDP hardness: information bottleneck instances
- **11.9.6.** Hindsight‑privileged simulators: oracle‑augmented bounds, imitation‑kicker algorithms
- **11.9.7.** Memory size vs. exploration trade‑off: finite memory vs. Bayes regret curves

### **11.10. Safety, Robustness & Risk**
- **11.10.1.** Risk‑sensitive criteria: **CVaR‑POMDP** algorithms, percentile constraints, distributional RL
- **11.10.2.** Robust POMDPs: interval, polyhedral & Wasserstein uncertainty sets, minimax planning
- **11.10.3.** Safe exploration & shielding: belief constraints, reach‑avoid under uncertainty
- **11.10.4.** Verification & explainability: policy certificates, counterfactual tracing in latent space
- **11.10.5.** Active perception & sensor management: information‑reward trade‑offs, value of sensing

### **11.11. Multi‑Agent & Dec‑POMDPs**
- **11.11.1.** Complexity landscape: NEXP‑completeness, undecidable hybrids, hardness gaps
- **11.11.2.** Exact & approximate planning: **MAA\***, multi‑agent incremental pruning, finite‑state controllers
- **11.11.3.** Neural approaches: **I‑POMDP‑Net**, **QMIX‑MA‑POMDP**, transformer belief sharing
- **11.11.4.** Communication protocols: emergent signalling, bandwidth‑constrained coordination
- **11.11.5.** Co‑operative vs. competitive scenarios: imperfect‑information games, opponent modelling

### **11.12. Benchmarks & Datasets**
- **11.12.1.** Classic small‑scale tasks: **Tiger**, **Light‑Dark**, **RockSample**, **Hallway**
- **11.12.2.** Gridworld & procedural: **MiniGrid**, **Procgen‑POMDP**
- **11.12.3.** Vision‑based high‑dimensional: Partially‑Observable Atari, **DM‑Lab‑30** PO variants
- **11.12.4.** Continuous control & navigation: Dubins car, drone collision‑avoidance, Habitat
- **11.12.5.** Dialogue & language POMDPs: **bAbI‑RL**, **MultiWOZ‑RL**, schema‑guided tasks
- **11.12.6.** Robotics simulators: **Gazebo‑POMDP**, **Isaac‑Gym** partial‑obs suites
- **11.12.7.** Real‑world offline datasets: sepsis treatment, industrial telemetry, stock‑LOB streams

### **11.13. Software Ecosystem**
- **11.13.1.** Planning libraries: **SARSOP**, **APPL**, **pomdp‑solve**, **DESPOT**
- **11.13.2.** Deep RL frameworks: **RLlib**, **CleanRL**, **Stable‑Baselines3** (partial‑obs wrappers)
- **11.13.3.** Belief‑tracking toolkits: **pomdp_py**, **PyParticleEst**, **Bayes‑Filters‑Lib**
- **11.13.4.** Experiment management & reproducibility: **RL‑Zoo** configs, Weights‑and‑Biases templates
- **11.13.5.** R & Julia ecosystems: `pomdp` (R), **POMDPs.jl**, **POMCPOW.jl**

### **11.14. Empirical Methodology**
- **11.14.1.** Evaluation metrics: return, regret, sample efficiency, belief calibration
- **11.14.2.** Statistical power & confidence intervals: multiple seeds, effect sizes, bootstrap tests
- **11.14.3.** Hyper‑parameter sensitivity under observation noise: robustness sweeps, tuning grids
- **11.14.4.** Sim‑to‑real transfer diagnostics: dynamics mismatch tests, domain randomisation checks
- **11.14.5.** Visualising uncertainty: belief heat‑maps, particle animations, saliency on observations

### **11.15. Applications (Case Studies)**
- **11.15.1.** Autonomous navigation & SLAM‑aware control
- **11.15.2.** Healthcare: ICU sepsis, radiotherapy fraction planning
- **11.15.3.** Dialogue systems: task‑oriented & open‑domain conversational POMDPs
- **11.15.4.** Finance & trading: latent order‑book modelling, execution under partial info
- **11.15.5.** Surgical tele‑operation & industrial robotics
- **11.15.6.** Multi‑robot coordination & search‑and‑rescue

### **11.16. Open Problems & Future Directions**
- **11.16.1.** Scalable belief updates: neural importance sampling, amortised filters with guarantees
- **11.16.2.** Provable deep recurrent & transformer RL: generalisation and stability bounds
- **11.16.3.** Continual & lifelong POMDP learning: catastrophic forgetting under partial observability
- **11.16.4.** Causality‑aware agents: counterfactual reasoning, intervention planning
- **11.16.5.** Efficient offline exploration with limited coverage: theory & benchmarks
- **11.16.6.** LLM‑integrated agents: natural‑language observations, large‑context memory compression
- **11.16.7.** Standardised real‑world benchmarks & evaluation protocols: cross‑domain reproducibility

---

## **Chapter 12: Bayesian Reinforcement Learning**
*(Final, Research‑Driven Table of Contents)*

### **12.0. Overview & Historical Context**
- **12.0.1.** Motivation: exploration, calibrated uncertainty, and sample efficiency
- **12.0.2.** Timeline of milestones (1968–2025)
- **12.0.3.** Bayesian RL vs. frequentist, distributional, and control‑as‑inference paradigms

### **12.1. Bayesian Inference Foundations for RL**
- **12.1.1.** Bayesian decision‑theoretic formulation of MDP control
- **12.1.2.** Priors over dynamics & rewards (Conjugate, Non‑conjugate)
- **12.1.3.** Posterior updates from interaction histories
- **12.1.4.** Belief states & Bayes‑Adaptive MDP (**BAMDP**)
- **12.1.5.** Equivalence to POMDPs; sufficiency of beliefs
- **12.1.6.** Computational intractability of exact belief planning

### **12.2. Approximate Inference & Uncertainty Representation**
- **12.2.1.** Variational inference & ELBO‑regularised value functions
- **12.2.2.** Expectation propagation & Laplace approximations
- **12.2.3.** Sequential Monte‑Carlo / particle filtering in **BAMDPs**
- **12.2.4.** Bootstrapped ensembles & randomised prior functions
- **12.2.5.** Bayesian neural networks & hyper‑net priors for deep RL
- **12.2.6.** Probabilistic latent world models (**PlaNet**, **Dreamer**, **PETS**)

### **12.3. Algorithms for Belief‑Space Planning (Exact & Tree Search)**
- **12.3.1.** Dynamic programming on discretised beliefs
- **12.3.2.** Forward‑Search Sparse Sampling (**FSSS**, **BFS3**)
- **12.3.3.** Bayes‑Adaptive Monte‑Carlo Planning (**BAMCP**, **BA‑UCT**)
- **12.3.4.** **ADA‑MCTS** & safe non‑stationary extensions
- **12.3.5.** Neural‑particle **BAMCP** for high‑dimensional states
- **12.3.6.** Complexity bounds & anytime guarantees

### **12.4. Posterior‑Sampling & Randomised Value Functions**
- **12.4.1.** Thompson/Posterior‑Sampling RL (**PSRL**) — episodic & discounted
- **12.4.2.** Linear‑kernel **PSRL**, **LSVI‑PG**, and **Neural‑PSRL**
- **12.4.3.** Bayesian regret bounds (tabular, linear, RKHS, general FA)
- **12.4.4.** Gittins indices & indexability connections

### **12.5. Bayesian Confidence & Information‑Theoretic Exploration**
- **12.5.1.** Bayesian UCRL (posterior confidence sets)
- **12.5.2.** Value‑of‑information & expected information gain bonuses
- **12.5.3.** **VIME**, $\eta$‑greedy, $\phi$‑exploration in deep settings
- **12.5.4.** Risk‑constrained exploration and safe optimism

### **12.6. Deep Bayesian Model‑Free Methods**
- **12.6.1.** Bayesian Q‑learning variants (**RLSVI**, **B‑DQN**)
- **12.6.2.** Bayesian actor‑critic & natural‑gradient methods
- **12.6.3.** Bootstrapped DQN & deep ensemble exploration
- **12.6.4.** Uncertainty propagation vs. distributional RL

### **12.7. Offline & Offline‑to‑Online Bayesian RL** *(New section)*
- **12.7.1.** Pessimistic posteriors & conservative value estimation
- **12.7.2.** Uncertainty‑regularised offline Q‑learning
- **12.7.3.** Bayesian Policy Optimisation with behaviour‑cloning priors
- **12.7.4.** Safe offline‑to‑online fine‑tuning with calibrated risk bounds

### **12.8. Multi‑Agent & Game‑Theoretic Bayesian RL** *(New section)*
- **12.8.1.** Bayesian opponent modelling & belief inference in Dec‑POMDPs
- **12.8.2.** Bayesian Nash and correlated‑equilibrium MARL
- **12.8.3.** Robust Bayesian MARL under payoff uncertainty
- **12.8.4.** Risk‑sharing & cooperative Bayesian exploration

### **12.9. Continuous Control & Safe Bayesian RL**
- **12.9.1.** Gaussian‑process dynamics models (**PILCO**, **GP‑MPC**)
- **12.9.2.** **SafeOpt**, **SAFE‑CtrlBO**, and constraint‑feasible exploration
- **12.9.3.** Online Bayesian LQR & stochastic MPC
- **12.9.4.** Bayesian policy search for robotics & hardware‑in‑the‑loop

### **12.10. Uncertainty, Risk & PAC‑Bayes Theory** *(Merged focus)*
- **12.10.1.** Bayesian sample‑complexity & minimax lower bounds
- **12.10.2.** **CVaR**, entropic & mean‑variance risk measures
- **12.10.3.** Distributionally‑robust and Rényi‑divergence control‑as‑inference
- **12.10.4.** PAC‑Bayes generalisation bounds for RL & lifelong learning
- **12.10.5.** Information‑theoretic complexity (**IB**, **MERL**)

### **12.11. Hierarchical Priors, Meta‑ & Lifelong Bayesian RL**
- **12.11.1.** Hierarchical Bayes across tasks (meta‑BRL)
- **12.11.2.** Dirichlet‑process & CRP priors for infinite task pools
- **12.11.3.** **EPIC** & other PAC‑Bayes lifelong algorithms
- **12.11.4.** Structural/symmetry‑aware priors (graphs, group invariance)
- **12.11.5.** Neural function‑space priors & representation reuse

### **12.12. Bayesian Inverse RL & Preference Learning**
- **12.12.1.** Bayesian formulation of inverse RL
- **12.12.2.** Gaussian‑process preference learning & active queries
- **12.12.3.** Reward‑uncertainty calibration for **RLHF** & alignment

### **12.13. Scalable Engineering Practice**
- **12.13.1.** Posterior approximation at scale (variational, SMC, ensembles)
- **12.13.2.** Distributed belief updates, GPUs & TPUs
- **12.13.3.** Probabilistic programming frameworks (Pyro, NumPyro, Bean Machine)
- **12.13.4.** Benchmarking, evaluation protocols & reproducibility guidance

### **12.14. Applications**
- **12.14.1.** Robotics & industrial automation
- **12.14.2.** Healthcare & personalised medicine
- **12.14.3.** Finance & portfolio management
- **12.14.4.** Autonomous vehicles & UAVs
- **12.14.5.** Content recommendation & adaptive experimentation

### **12.15. Open Problems & Future Directions**
- **12.15.1.** Unified exploration–exploitation–safety theory
- **12.15.2.** Bayesian RL under environment non‑stationarity
- **12.15.3.** Scalable BRL with high‑dimensional perception
- **12.15.4.** Interpretable & formally verified Bayesian policies
- **12.15.5.** Bayesian causal RL & causal discovery
- **12.15.6.** BRL for foundation‑model alignment & hallucination control

### **12.16. Summary & Further Reading**

---

## **Chapter 13: Imitation & Inverse Reinforcement Learning**
*Revised & critically‑balanced scaffold incorporating emergent research lines through mid‑2025.*

### **Part 1: Orientation & Foundations**
- **13.0. Overview, Taxonomy & Historical Context**
    - Situate IL/IRL; contrast imitation, reward inference, **RLHF**.
    - Demonstration modalities: State–action, state‑only, preferences, corrections, language.
    - Evaluation axes: Online vs offline, interaction budget, safety, reproducibility.
- **13.1. Formal Problem Statements**
    - Reward‑free MDP, demonstrations dataset, occupancy‑measure machinery, divergence minimisation.
- **13.2. Statistical & Computational Pre‑liminaries**
    - Trajectory concentration bounds, function‑approximation classes, optimisation toolkit.

### **Part 2: Behavioural & Interactive Imitation**
- **13.3. Behavioural Cloning (BC)**
    - **13.3.1.** MLE objective & compounding‑error analysis.
    - **13.3.2.** Regularisation, data augmentation, and continual BC.
    - **13.3.3.** Decision transformers as sequence models.
- **13.4. Dataset Aggregation & Interactive IL**
    - **13.4.1.** **DAgger** protocol & $\alpha$‑regret.
    - **13.4.2.** **SafeDAgger**, **AggreVaTe**.
    - **13.4.3.** Active querying, cost‑sensitive allocation, human‑in‑the‑loop logistics.
- **13.5. Diffusion & Sequence‑Model Policies** *(New dedicated section)*
    - **13.5.1.** Score‑based policy objectives.
    - **13.5.2.** Theoretical open questions.
    - **13.5.3.** Diffusion‑policy robotics case‑studies; robustness & uncertainty.

### **Part 3: Offline Imitation & Inverse RL**
- **13.6. Offline Supervised IL**
    - **13.6.1.** Importance‑weighted BC & doubly‑robust estimators.
    - **13.6.2.** Distributional‑robust objectives and finite‑sample guarantees.
- **13.7. Offline Adversarial IL / IRL**
    - **13.7.1.** **Inverse Q‑learning (IQ‑Learn)** & Fisher‑divergence IL.
    - **13.7.2.** **OPT‑AIL** polynomial complexity & empirical benchmarks.

### **Part 4: Classical & Modern IRL**
- **13.8. Classical IRL Foundations**
    - Ill‑posedness, feature‑expectation matching, apprenticeship‑learning bound.
- **13.9. Maximum‑Entropy IRL**
    - MaxEnt duality, partition‑function estimation, soft‑optimality.
- **13.10. Bayesian, PAC & Causal IRL**
    - Priors & posteriors, PAC reward‑set bounds, causal identifiability.
- **13.11. Adversarial & Potential‑Based IRL**
    - **AIRL**, dynamics‑invariant rewards, cross‑domain transfer, programmatic‑reward IRL.

### **Part 5: Human Feedback & Preference‑Based Learning**
- **13.12. Preference‑Based RL & RLHF Pipeline**
    - Pairwise comparison models, active query design, reward‑model pathologies, KL‑regularised fine‑tuning.
- **13.13. Interactive Reward Modelling**
    - Coactive corrections, natural‑language supervision, multi‑modal signals, real‑time safety overrides.

### **Part 6: Evaluation, Benchmarks & Reproducibility**
- **13.14. Metrics & Protocols**
    - Imitation loss, return gap, human‑satisfaction surveys, negative‑result reporting.
- **13.15. Benchmark Suites & Leaderboards**
    - **MuJoCo**, **CARLA**, **MineRL**, **RoboNet**, **ASIMOV**; pitfalls of leaderboard‑driven research.
- **13.16. Reproducibility & Open Science**
    - Dataset licensing, logging standards, hyper‑parameter disclosure, continual‑benchmark initiatives.

### **Part 7: Safety, Robustness & Security**
- **13.17. Demonstration Corruption & Distribution Shift**
    - Outliers, poisoning, causal misspecification, adversarial examples.
- **13.18. Safe Policy Improvement & Risk‑Sensitivity**
    - Constrained IL, **CVaR** objectives, shielded execution.
- **13.19. Alignment Failures & Value Hand‑off**
    - Specification gaming, side‑effect avoidance, governance considerations.

### **Part 8: Foundation‑Model Era & Scaling**
- **13.20. Large‑Scale Pre‑training for Control**
    - Robot‑foundation‑model pipelines (**RT‑1/RT‑2**, **GR00T**), scaling laws.
- **13.21. Language‑Conditioned & Tool‑Augmented Policies**
    - Instruction following, code/tool use, embodied LLM benchmarks.
- **13.22. Integration of Demonstrations, Preferences & Language**
    - Unified objective functions, multi‑channel credit assignment.

### **Part 9: Advanced Topics & Applications**
- **13.23. Partial Observability** (Belief‑space occupancy, memory‑augmented IL/IRL).
- **13.24. Multi‑Agent Settings** (Cooperative/competitive demos, inverse game theory).
- **13.25. Hierarchical & Option‑Based Imitation** (Trajectory segmentation, sub‑goal IRL).
- **13.26. Cross‑Domain Transfer & Meta‑Imitation** (Domain‑invariant features, sim‑to‑real).
- **13.27. Domain‑Specific Case‑Studies** (Healthcare, dialogue, shared‑control robotics).
- **13.28. Implementation & Engineering** (Data pipelines, training stability, distributed roll‑outs).

### **Part 10: Theory, Open Problems & Resources**
- **13.29. Complexity Gaps & Lower Bounds**
- **13.30. Towards Robust Value Alignment**
- **13.31. Conclusion & Community Resources** (Seminal papers, libraries, benchmarks).

---

## **Chapter 14: Hierarchical Reinforcement Learning & Temporal Abstraction**

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
- **14.2.5.** Worked example: bottleneck grid world

### **14.3. Option-Learning & Hierarchical Policy Gradient**
- **14.3.1.** Intra-Option Policy Gradient Theorem
- **14.3.2.** Termination-Gradient Theorem
- **14.3.3.** Convergence via Two-Time-Scale Stochastic Approximation
- **14.3.4.** Variants: **Natural Option-Critic**, Diversity-Enriched **OC**
- **14.3.5.** Practical tips: experience replay across option boundaries

### **14.4. Skill / Option Discovery**
- **14.4.1.** Spectral and Eigen-Options (graph Laplacian)
- **14.4.2.** Skill-Chaining and reachability graphs
- **14.4.3.** Unsupervised MI methods (**DIAYN**, **DADS**, **VALOR**, **ADC**)
- **14.4.4.** Diversity regularisers (DPP, Wasserstein, VIC)
- **14.4.5.** Interest functions and state-density weighting
- **14.4.6.** Evaluation benchmarks (**Procgen Maze**, **MuJoCo AntMaze**, **Minigrid**)

### **14.5. Multi-Level Architectures**
- **14.5.1.** Feudal RL & **FuN**
- **14.5.2.** Hierarchical Actor-Critic (**HAC**)
- **14.5.3.** **HIRO** and off-policy correction
- **14.5.4.** **SHIRO**: Entropy-Regularised **HIRO**
- **14.5.5.** Planning architectures (**Dreamer-H**, **MCTS-HRL**)

### **14.6. Sample-Complexity & Regret Theory**
- **14.6.1.** Information-theoretic lower bounds (goal-conditioned, horizon $H$)
- **14.6.2.** Matching upper bounds (Robert et al. 2023)
- **14.6.3.** Polynomial speed-up conditions (cover time $\le \kappa$)
- **14.6.4.** Misspecification regret (imperfect option set)
- **14.6.5.** Exploration–exploitation trade-offs (open question)

### **14.7. Hierarchical Representation & World Models**
- **14.7.1.** Latent goal spaces via MI objectives (**CPC**, **InfoNCE**)
- **14.7.2.** Bisimulation and successor features
- **14.7.3.** World-model factorisation (manager latent vs. worker dynamics)
- **14.7.4.** Planning-error propagation bound: $\varepsilon_{\text{model}} \le \varepsilon_{\text{high}} + H \cdot \varepsilon_{\text{low}}$
- **14.7.5.** Latent-dimension vs. optimality gap theorem

### **14.8. Transfer, Continual & Lifelong HRL**
- **14.8.1.** Option-indexing meta-learning
- **14.8.2.** Zero-shot option reuse guarantees
- **14.8.3.** Skill-composition transformers
- **14.8.4.** Catastrophic interference mitigation (**EWC**, orthogonal regularisation)

### **14.9. Offline & Safe HRL**
- **14.9.1.** Batch-RL foundations for SMDPs
- **14.9.2.** Behaviour-Constrained Option Learning (**BC-OC**)
- **14.9.3.** Shielded and chance-constrained intra-option policies
- **14.9.4.** Safe termination verification via temporal logic
- **14.9.5.** Dataset shift and counterfactual corrections

### **14.10. Partial Observability & POMDP-HRL**
- **14.10.1.** Belief-space managers
- **14.10.2.** Recurrent option policies
- **14.10.3.** Information bottlenecks for options
- **14.10.4.** Applications: **ViZDoom**, real-world navigation

### **14.11. Multi-Agent Hierarchical Coordination**
- **14.11.1.** Shared skill libraries and Kronecker graphs
- **14.11.2.** Hierarchical credit assignment across agents
- **14.11.3.** Game-theoretic option selection
- **14.11.4.** Case study: **StarCraft II** macro-actions

### **14.12. Applications & Case Studies**
- **14.12.1.** Robotic manipulation (**Meta-World**, **Franka Kitchen**)
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

## **Chapter 15: Multi-Agent Reinforcement Learning & Stochastic Games**

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
    - Zero-sum solvers: **Minimax-Q**, **Nash-Q**
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
- **15.11.1.** Cooperative suites: **SMAC-v2**, **Google Football**, **Overcooked-MR-2024**
- **15.11.2.** Competitive/mixed suites: **Hanabi**, **Stratego**, **MPE-DG**
- **15.11.3.** Real-world control: traffic, warehouses, energy markets
- **15.11.4.** Metrics: win-rate, exploitability, NashConv, generalisation
- **15.11.5.** Reproducibility stacks: **PettingZoo**, **MARL-Bench**, **RLlib-MARL**

### **15.12. Theoretical Guarantees & Complexity**
- **15.12.1.** PAC bounds for team and zero-sum stochastic games
- **15.12.2.** Online regret minimisation (**CFR**, **OMD**)
- **15.12.3.** Convergence of Gradient Descent/Ascent (**GDA**)
- **15.12.4.** Propagation-of-chaos and mean-field convergence
- **15.12.5.** Sample-complexity gaps: CTDE vs. fully decentralised

### **15.13. Open Questions & Research Frontiers**
- **15.13.1.** Equilibrium selection with function approximation
- **15.13.2.** Scaling on-policy learning to 1000+ agents
- **15.13.3.** Standardised offline MARL benchmarks
- **15.13.4.** Human–AI collaboration and norm formation
- **15.13.5.** Sim-to-real transfer and cross-game generalisation

---

## **Chapter 16: Task Distributions & Transfer Principles**

### **16.0. Orientation & Road-Map**
- **16.0.1.** Motivation and historical context
- **16.0.2.** Running toy-examples
- **16.0.3.** Reader’s guide (theory → algorithms → evaluation flow)

### **16.1. Sample-Space Formulation of an MDP Family**
- **16.1.1.** MDP recap & notation ($S, A, P, R, \gamma$)
- **16.1.2.** Task-generating random variables: latent $\theta$, generative map $f:\Theta\to\mathcal M$
- **16.1.3.** Sampling regimes: IID batches, non-IID streams, adversarial sequences
- **16.1.4.** Structural assumption library: shared ($S,A$), Lipschitz in $\theta$, compact support
- **16.1.5.** Canonical task families: contextual bandits, linear systems, domain-randomised robotics
- **16.1.6.** Extensions: belief-MDPs, causal task graphs, exchangeable processes
- **16.1.7.** Common pitfalls: support mismatch, hidden confounders, unverifiable priors

### **16.2. Quantifying Task Similarity**
- **16.2.1.** Design desiderata: transfer correlation, sample computability, invariances
- **16.2.2.** Distributional metrics: KL, $\chi^2$, TV, Jensen–Shannon; Wasserstein & OT
- **16.2.3.** Dynamics-focused metrics: bisimulation, successor-feature distance
- **16.2.4.** Representation-driven metrics: learned task embeddings, contrastive **InfoNCE**
- **16.2.5.** Empirical estimation: importance-weighting, kernel-MMD, GNN graph-matching
- **16.2.6.** Theoretical properties: stability, invariance classes, sample-complexity lower bounds
- **16.2.7.** Value-aware & EPIC distances: **AVD**, **EPIC**, **DARD**; regret and optimal-policy bounds
- **16.2.8.** Metric-learning pitfalls: spurious similarity, over-smooth embeddings

### **16.3. Generalisation Guarantees Across Tasks**
- **16.3.1.** PAC & PAC-Bayes refresher (single task)
- **16.3.2.** Meta-PAC-Bayes bounds: hierarchical priors, task-conditioned posteriors
- **16.3.3.** Online & lifelong regret bounds: memory-limited agents, task streams
- **16.3.4.** Distribution-shift compensation terms: shift-aware KL, Wasserstein corrections
- **16.3.5.** Information-theoretic objectives: MDL, mutual-information regularisers
- **16.3.6.** Lower bounds & impossibility results: adversarial tasks, negative-transfer hardness
- **16.3.7.** Practical implications: posterior sampling, optimism, Bayesian meta-RL recipes

### **16.4. Taxonomy of Transfer Mechanisms**
- **16.4.1.** Four transferable objects: representation $\phi$, dynamics $P$, policy $\pi$, reward $R$
- **16.4.2.** Representation transfer: SSL pre-training, successor features
- **16.4.3.** Dynamics transfer: latent SSMs, simulators, robust MPC
- **16.4.4.** Policy transfer: warm-starts, option libraries, distillation
- **16.4.5.** Reward & preference transfer: potential-based shaping, inverse RL, **RLHF**
- **16.4.6.** Hybrid & hierarchical transfer: joint $\phi + \pi$, meta-optimisation
- **16.4.7.** Unsupervised skill discovery & autonomous RL: **DIAYN**, **APS**
- **16.4.8.** Multi-agent & LLM-augmented transfer: opponent modelling, tool-use

### **16.5. Evaluation Protocols & Metrics**
- **16.5.1.** Performance metrics: jump-start, asymptotic gain, forward/backward transfer
- **16.5.2.** Efficiency metrics: sample complexity, wall-clock, energy / CO₂ cost
- **16.5.3.** Continual-learning metrics: forgetting rate, knowledge-retention
- **16.5.4.** Statistical methodology: hierarchical bootstrap, effect sizes, CIs
- **16.5.5.** Benchmark suites: **Meta-World+**, **MT10/50**, **Procgen**, **RL-Unplugged-Meta**
- **16.5.6.** Responsible-AI metrics: safety violations, fairness gaps, privacy leakage
- **16.5.7.** Reproducibility check-lists & badges (ICML/NeurIPS 2025 requirements)

### **16.6. Algorithmic Frameworks for Transfer & Meta-RL**
- **16.6.1. Model-Free Meta-RL:** Gradient-based (**MAML**), Memory-based (**RL²**), Exploration-driven (**MAESN**)
- **16.6.2. Model-Based Meta-RL:** **Dreamer** variants, Latent dynamics adaptation (**E2C-meta**)
- **16.6.3. Sequence Models & RLHF Transfer:** Decision Transformers, In-context RL, Preference reuse
- **16.6.4. Curriculum & Task-Sequencing:** Domain randomisation, **PLR**, Bayesian curriculum shaping
- **16.6.5. Skill Discovery & Option Libraries:** **DIAYN**, **CIC**, **APS**, Skill distillation
- **16.6.6. Safety- & Multi-objective-aware variants:** Constrained meta-RL, Distributionally robust baselines

### **16.7. Applications & Case Studies**
- **16.7.1.** Robotics (sim-to-real, multi-skill)
- **16.7.2.** Multilingual dialogue & NLP
- **16.7.3.** Healthcare personalisation
- **16.7.4.** Autonomous driving & fleet learning
- **16.7.5.** Game playing & procedural generalisation

### **16.8. Open Problems & Future Directions**
- **16.8.1.** Lifelong meta-learning under non-stationary drift
- **16.8.2.** Causality-aware task transfer
- **16.8.3.** Data-efficient world-model learning
- **16.8.4.** Multi-objective safety-fairness trade-offs
- **16.8.5.** Policy implications and societal impact

### **16.9. Summary & Further Reading**

---

## **Chapter 17: Meta-Reinforcement Learning**

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
- **17.3.3.** Gradient-free and evolution strategies (**CMA-ES**, **PBT**, **ARS**)
- **17.3.4.** Meta-policy-gradient formulations
- **17.3.5.** Hyper-parameter search and BO-MRL

### **17.4. Recurrent & Transformer In-Context Meta-Learners**
- **17.4.1.** **RL²**, **L2RL**, and other RNN agents
- **17.4.2.** Memory-augmented architectures (**NTM**, **DND**)
- **17.4.3.** Sequence-model meta-RL (**Decision Transformer-Meta**)
- **17.4.4.** Hyper-networks and meta-controllers

### **17.5. Latent-Variable & Uncertainty-Aware Methods**
- **17.5.1.** Latent-context MDP formalism
- **17.5.2.** Posterior sampling and information-bottleneck encoders
- **17.5.3.** **PEARL**, **VariBAD**, **Bayes-Adapt**
- **17.5.4.** Context-VAE / latent-PPO & SAC families

### **17.6. Model-Based Meta-RL**
- **17.6.1.** **MB-MAML** and **RLG-MAML**
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

## **Chapter 18: Continual Reinforcement Learning**

### **18.0. Overview & Motivation**
- **18.0.1.** Why Continual Learning in RL?
- **18.0.2.** Historical Trajectory (1990–2025)
- **18.0.3.** Terminology & Scope (Continual vs Lifelong vs Online vs Open-World)
- **18.0.4.** Chapter Road-Map

### **18.1. Non-Stationary Environments & Task Sequences**
- **18.1.1.** Drift Taxonomy: Abrupt, Gradual, Periodic, Latent-context switches
- **18.1.2.** Formal Definition: Non-Stationary MDP with time-indexed kernels $(P_t, R_t)$
- **18.1.3.** Detecting Change: Statistical tests, surprise signals, Hidden-Mode MDP inference
- **18.1.4.** Task Similarity & Transferability
- **18.1.5.** Exploration–Exploitation under Drift
- **18.1.6.** Problem Variants: Curriculum, Unknown boundaries, Open-world RL
- **18.1.7.** Special Settings: POMDP & latent-state drift, Hybrid Offline-to-Online CL

### **18.2. Algorithmic Approaches**
- **18.2.1. Regularisation-Based Methods:** Quadratic Priors (**EWC**), Path-Integral (**SI**, **MAS**)
- **18.2.2. Replay-Based Methods:** Experience Replay, Generative Replay, Selective Replay (coresets, KD)
- **18.2.3. Parameter Isolation & Modular Architectures:** Dynamic Expansion (**Progressive Nets**), Mask-Based Reuse (**PackNet**), Gated Routing (MoE)
- **18.2.4. Meta-Learning & Hyper-Networks:** Meta-Gradient Adaptation (**MAML**, **OML**), Learned Optimisers
- **18.2.5. Continual World-Models:** Recurrent & State-Space CL models, Memory-Augmented Models
- **18.2.6. Safety-Aware Algorithms:** Safe replay buffers, constrained RL
- **18.2.7. Dual-Memory & Consolidation Systems:** Sleep replay, latent rehearsal
- **18.2.8. LLM-Augmented Continual RL:** Skill-language models, tool-use agents
- **18.2.9. Resource-Aware Design:** Capacity-compute-plasticity trade-offs

### **18.3. Evaluation Methodology**
- **18.3.1. Metrics:** CL-Score (BWT, FWT, Forgetting), Efficiency (sample, compute), Safety, Capacity
- **18.3.2. Benchmarks:** Synthetic streams, **MineRL-CL**, **Lifelong-ProcGen**, **Meta-World-Seq**, **LifelongAgentBench**
- **18.3.3. Experimental Protocols:** Single-Pass Online Evaluation, Joint Validation, OOD & Safety Stress-Tests
- **18.3.4. Reproducibility & Tooling:** Drift generators, logging, leaderboards

### **18.4. Theoretical Foundations**
- **18.4.1.** Stability–Plasticity Information Theory: Information Bottleneck, Compression-Retention Trade-off
- **18.4.2.** Regret & Sample Complexity under Drift: Path-Regret Bounds, Adaptive Policy Gradients
- **18.4.3.** Generalisation with Memory Constraints: PAC-Bayesian Replay Bounds, Coreset Size vs Forgetting
- **18.4.4.** Constraint Retention Guarantees & Formal Verification
- **18.4.5.** Lower Bounds & No-Free-Lunch Results

### **18.5. Applications & Case Studies**
- **18.5.1.** Robotics (manipulation, navigation)
- **18.5.2.** Game AI & Procedural Content
- **18.5.3.** Autonomous Vehicles & Traffic
- **18.5.4.** Industrial Process Control
- **18.5.5.** Healthcare & Personalised Assistants
- **18.5.6.** LLM-Driven Agents & Tool Use

### **18.6. Open Challenges & Future Directions**
- **18.6.1.** Robust OOD Adaptation
- **18.6.2.** Memory-Efficient CL at Scale
- **18.6.3.** Long-Horizon Safety & Certification
- **18.6.4.** Autonomous Task Discovery & Sim-to-Real Deployment
- **18.6.5.** Unified Theory of Transfer & Forgetting
- **18.6.6.** Human-in-the-Loop Continual RL
- **18.6.7.** Foundation World-Models & Pre-Training

### **18.7. Chapter Summary & Further Reading**

---

## **Chapter 19: Synthesis & Open Questions**
- **19.1.** Unified Bayesian perspective linking representation, exploration, transfer, and continual learning
- **19.2.** Exploration–adaptation trade-off across timescales
- **19.3.** Scalability and safety in real-world lifelong RL
- **19.4.** Major research gaps: polynomial-time exploration with latent dynamics, reliable evaluation under distribution shift
