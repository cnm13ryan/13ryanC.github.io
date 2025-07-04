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

# Comprehensive Reinforcement Learning: Theory and Practice

## 1. MDP Foundations & Optimality

### 1.1 Mathematical Foundations
- **1.1.1** Measure-theoretic set-up
- **1.1.2** Formal definition of an MDP $(S,A,P,r,\gamma)$
- **1.1.3** Discount factor & effective horizon $H_{\gamma,\varepsilon}$

### 1.2 Policy Framework
- **1.2.1** Policies and the agent–environment loop
- **1.2.2** Probability of trajectories (Ionescu–Tulcea)
- **1.2.3** Return, value functions & optimality notions
- **1.2.4** Memoryless vs general policies; fundamental optimality theorem

### 1.3 Optimality Theory
- **1.3.1** Objective functions & modelling choices
- **1.3.2** Discounted occupancy measures
- **1.3.3** Bellman operators $T^\pi,\,T$
- **1.3.4** Key analytical properties (contraction, error bounds)
- **1.3.5** Greedy policies & the Fundamental Theorem

### 1.4 Concentration Inequalities
- **1.4.1** Self‑normalised concentration inequalities (Elliptical Potential, Bernstein)

---

## 2. Exact Dynamic Programming

### 2.1 Value-Iteration Theory
- **2.1.1** Fundamental Theorem of Dynamic Programming (finite MDPs)
- **2.1.2** Effective horizon & Bellman update $v_{k+1}=Tv_k$
- **2.1.3** Policy-error (greedy) bound
- **2.1.4** Fixed-point iteration via Banach's lemma
- **2.1.5** Finite-horizon interpretation
- **2.1.6** Algorithmic description, convergence & $\varepsilon$-stopping rule
- **2.1.7** Geometry of value functions (Dadashi et al.)
- **2.1.8** Banach's Fixed-Point Theorem (background box)
- **2.1.9** Linear-programming view (primal/dual)
- **2.1.10** Value iteration as *approximate* planning
- **2.1.11** Runtime of $\varepsilon$-optimal planning with VI
- **2.1.12** Computational complexity of exact planning
- **2.1.13** $\delta$–$\varepsilon$ error-control summary

### 2.2 Policy-Iteration Analysis
- **2.2.1** Definition of Policy Iteration algorithm
- **2.2.2** Advantage function & Performance-Difference identity
- **2.2.3** Geometric-Progress Lemma
- **2.2.4** Geometric convergence of value error
- **2.2.5** Strict-Progress Lemma (sub-optimal action elimination)
- **2.2.6** Overall runtime bound (Scherrer)
- **2.2.7** Value- vs Policy-Iteration comparison
- **2.2.8** Proof that PI $\geq$ VI speed
- **2.2.9** Mixing rates & span-seminorm contraction
- **2.2.10** Upper & lower runtime bounds (Ye; Feinberg-Huang-Scherrer)
- **2.2.11** Measure-theoretic view of occupancy-measure projection

### 2.3 Learned‑Model Dynamic Programming
- **2.3.1** Model‑bias bounds for Bellman operators
- **2.3.2** Ensemble variance as a proxy for $\varepsilon_P$
- **2.3.3** Optimism under model uncertainty & Thompson sampling

---

## 3. Online Planning in Discounted MDPs

### 3.1 Problem Formulation
- **3.1.1** Motivation (curse of dimensionality; local planning)
- **3.1.2** Access modes: global → local → online
- **3.1.3** Environment model: black-box simulator
- **3.1.4** Formal statement of the online-planning problem

### 3.2 Algorithmic Framework
- **3.2.1** Optimisation language & oracle types
- **3.2.2** Agent–environment interaction loop
- **3.2.3** Online planner & $\delta$-soundness definition
- **3.2.4** Cost metrics (queries & arithmetic)

### 3.3 Analysis & Bounds
- **3.3.1** Baseline algorithm (recursive value evaluation)
- **3.3.2** Upper runtime bounds (deterministic & sparse-sampling)
- **3.3.3** Matching lower bounds $\Omega(A^H)$
- **3.3.4** Local vs online access trade-offs
- **3.3.5** Sampling & averaging fundamentals
- **3.3.6** Policy-error analysis ($\varepsilon$ & "almost-$\varepsilon$" cases)
- **3.3.7** Putting it together: choosing $m,H,\zeta$ for $\delta$-soundness

### 3.4 Extensions & Applications
- **3.4.1** Open questions & extensions (e.g. MCTS links)
- **3.4.2** Model‑Predictive Control with Learned Simulators
  - **3.4.2.1** Cross‑Entropy Method (CEM) and trajectory optimisation
  - **3.4.2.2** TD‑MPC algorithm and stability analysis
  - **3.4.2.3** Comparative runtime vs MCTS

---

## 4. Value-Function Approximation & Approximate Policy Iteration

### 4.1 Value-Function Approximation
- **4.1.1** Approximate Universal Value-Function Realizability (Assump. B2)
- **4.1.2** Approximate Policy Evaluation – MC rollout + weighted LS
- **4.1.3** Extrapolation-error control in weighted LS
- **4.1.4** Kiefer–Wolfowitz Theorem (G-optimal design)
- **4.1.5** Corollary – extrapolation control via optimal design
- **4.1.6** LSPE-G high-probability error bound

### 4.2 Approximate Policy Iteration
- **4.2.1** Geometric Progress Lemma with approximate improvement
- **4.2.2** Approximate Policy Iteration theorem – $\frac{\gamma^k}{1-\gamma}+\varepsilon$ bounds
- **4.2.3** API with approximate action-value functions (corollary)
- **4.2.4** Least-Squares Policy Iteration (LSPI) algorithm & guarantees

### 4.3 State Abstractions
- **4.3.1** Motivation (sample sharing; dimensionality reduction)
- **4.3.2** Statistical trade-off: sample size vs approximation error
- **4.3.3** Exact abstraction hierarchy ($\pi^\ast$, $Q^\ast$, model-irrelevance)
- **4.3.4** Key theorems on the hierarchy & error bounds
- **4.3.5** Improvements & variants (homomorphisms, utile distinctions)
- **4.3.6** Approximate abstractions & bisimulation metrics ($\varepsilon$-$\pi^\ast$, $\varepsilon$-$Q^\ast$, $\varepsilon_R,\varepsilon_P$ bisimulation)
  - **4.3.6.1** **Action‑bisimulation** (Rudolph et al., 2024) — aggregates state–action pairs that induce identical action‑conditional reward and transition distributions, tightening value‑loss bounds relative to standard bisimulation
  - **4.3.6.2** **Effect‑equivalent abstraction** (Mavor‑Parker et al., 2025) — learns a homomorphism that clusters pairs with equivalent *next‑state* value distributions, guaranteeing $\tilde O\!\bigl(\varepsilon/(1-\gamma)\bigr)$ regret when combined with optimistic exploration
- **4.3.7** Bounding value loss for approximate abstractions
- **4.3.8** Finite-sample analysis ($n_\phi(D)$, Hoeffding bound)
  - **4.3.8.1** **Encoder bias term** — Add lemma that bounds additional bias due to a learned encoder $\hat\phi$ with $\|\hat\phi-\phi^\star\|_\infty\le\varepsilon_\mathrm{enc}$; overall sample‑complexity becomes $\tilde O\!\bigl(\tfrac{|\hat S|}{(\varepsilon-\varepsilon_\mathrm{enc})^2}\bigr)$
- **4.3.9** Bridging symbolic & learned abstractions
  - **4.3.9.1** Motivation & open problem statement
  - **4.3.9.2** **Self‑supervised Markov latent‑state discovery** (Sobal et al., 2025, *PLDM*)
  - **4.3.9.3** Iterative *refine–plan* pipeline:
    1. Initialise latent encoder $\hat\phi_\psi$ with a contrastive loss
    2. Solve the abstract MDP on $\hat S$ using LSVI‑UCB; collect roll‑outs
    3. Update $\psi$ via action‑bisimulation contrastive objective
    4. Repeat until policy improvement $<\tau$
  - **4.3.9.4** Conjectured regret bound: $\tilde O\!\bigl(H\sqrt{dT}\;+\;\varepsilon_\mathrm{enc}/(1-\gamma)\bigr)$
 

### 4.4 Offline / Batch Reinforcement Learning
- **4.4.1** Motivation (logged data; safety‑critical deployment; distribution shift; sample‑efficiency in large/continuous spaces)
- **4.4.2** Formal framework & assumptions (MDP, static dataset $\mathcal D$, behaviour policy $\beta$, realizability, Bellman‑completeness, coverage constant $C$)
- **4.4.3** Core algorithm – Fitted Q‑Iteration (regression perspective)
- **4.4.4** Error‑analysis pipeline (uniform deviation, one‑step error, propagation) & finite‑sample / fast‑rate theorems
- **4.4.5** Pessimistic / Conservative algorithms (CQL, OPAL, IQL) & performance‑difference analyses
- **4.4.6** Robustness to corrupted or sub‑optimal logs (adversarial noise, dataset shift)
- **4.4.7** Representation learning for Bellman completeness (spectral, contrastive, diffusion‑based)
- **4.4.8** Empirical benchmarks & practical guidelines (D4RL‑2025, MineRL‑Offline; coverage diagnostics)

### 4.5 Latent‑World‑Model Learning
- **4.5.1** Recurrent State‑Space Models (RSSM) & ELBO
- **4.5.2** Dreamer variants (V2, V3) and KL balancing
- **4.5.3** Representation capacity and empirical scaling laws

---

## 5. Sampling and Computational Complexity

### 5.1 Foundations & State‑Representation Learning
- **5.1.1** Markov decision processes (states, actions, rewards, $\gamma/H$)
- **5.1.2** Feature maps $\phi$: from hand‑crafted bases to learned encoders $\phi_\psi$
- **5.1.3** Six SRL families (metric, contrastive, data‑aug., world‑model, reconstruction, auxiliary‑task)
- **5.1.4** Evaluation protocols: DMControl‑100k, Atari‑100k, ProcGen; linear‑probe & bisim metrics
- **5.1.5** Norms and error metrics ($\|\cdot\|_\infty$, $\|\cdot\|_2$, bisim‑metric)
- **5.1.6** Pre‑computed core sets and LSPI recap

### 5.2 Probabilistic Tools for Sampling Analysis
- **5.2.1** Hoeffding's inequality – statement, proof sketch, intuition
- **5.2.2** Azuma–Hoeffding for martingales (variance proxy, step sizes)
- **5.2.3** Union-bound "upgrade" for simultaneous guarantees
- **5.2.4** Worked example: uniform-sampling best-arm identification

### 5.3 Covering Numbers & Uniform Convergence
- **5.3.1** $\ell_\infty$ covers and growth with dimension $d$
- **5.3.2** Lipschitz compositions and loss-class covering
- **5.3.3** Sample-complexity bound $\tilde{O}\left(\sqrt{\frac{\log N_\varepsilon}{n}}\right)$

### 5.4 Limits of Query‑Efficient Planning
- **5.4.1** Definition of $({\delta, \varepsilon})$-sound online planners
- **5.4.2** Large-$A$ lower bound
  - **5.4.2.1** Johnson–Lindenstrauss packing lemma
  - **5.4.2.2** High-probability "needle" lemma
  - **5.4.2.3** Exponential query cost $e^{\Omega(d)}$ (Weisz et al., 2021)
- **5.4.3** Fixed-horizon, small-$A$ lower bound
  - **5.4.3.1** Horizon-dependent fundamental theorem
  - **5.4.3.2** Query complexity $\tilde{\Omega}(AH/H)$ vs "large-$H$" regime
  - **5.4.3.3** Tightened few‑actions lower bound (Weisz et al., 2022)
- **5.4.4** Open: computational gap when $A$ fixed but $H\!\to\!\infty$

### 5.5 Planning under Realizability
- **5.5.1** $q^*$-realizability – linear assumption, global planner variant
- **5.5.2** $v^*$-realizability (TensorPlan)
  - **5.5.2.1** Interaction protocol & local simulator calls
  - **5.5.2.2** Ridge regression hypothesis set $\Theta$
  - **5.5.2.3** Optimism via square-root bonus ($\beta$)
  - **5.5.2.4** Covering-number analysis (size $\tilde{O}(d^2)$)
  - **5.5.2.5** Open: polynomial‑time implementation

### 5.6 Exploration in Linear MDPs
- **5.6.1** Problem set-up and boundedness assumptions
- **5.6.2** LSVI-UCB algorithm – step-wise ridge with UCB bonus
- **5.6.3** Martingale concentration (Azuma) for adaptive data
- **5.6.4** Elliptical‑potential lemma & regret $\tilde{O}(H^2 d \sqrt{T})$
- **5.6.5** Bernstein‑bonus refinement (LSVI‑UCB$^+$) & near‑minimax regret $\tilde{O}(H d \sqrt{T})$
- **5.6.6** Variance‑aware bonuses and tuning heuristics in practice
- **5.6.7** Robustness to encoder error — modify elliptical‑potential lemma to include $\varepsilon_\mathrm{enc}$ term; regret bound becomes $\tilde O\!\bigl(H^2\sqrt{dT}\;+\;H\sqrt{T}\,\varepsilon_\mathrm{enc}\bigr)$

### 5.7 Linear‑Programming View of MDPs
- **5.7.1** Primal LP: minimise $d_0^\top V$ s.t. $V \geq TV$
- **5.7.2** Dual LP: occupancy-measure constraints, $d^\pi$ cone
- **5.7.3** Monotonicity of $T$ and convergence to $V^*$

### 5.8 Sample‑Complexity of Model Learning
- **5.8.1** PAC bounds for $\varepsilon_P$‑accurate models
- **5.8.2** Lower bounds under model misspecification
- **5.8.3** Sample‑efficient exploration via information gain

### 5.9 Synthesis & Open Questions
- **5.9.1** Comparison table: lower vs upper bounds ($d, A, H, \gamma$)
- **5.9.2** Gaps: factor‑$H$ mismatch in discounted long‑horizon setting (upper $\tilde{O}(H^2)$ vs lower $\tilde{\Omega}(H)$)
- **5.9.3** Representation‑learning for exploration – towards implicit feature discovery
- **5.9.4** Computational vs query complexity – can TensorPlan be made **polynomial‑time**?

---

## 6. Robust & Safe Model‑Based Reinforcement Learning

### 6.1 Distributionally Robust MDPs
- **6.1.1** KL/TVD ambiguity sets & robust Bellman operators
- **6.1.2** Contraction & value‑error guarantees under robustness

### 6.2 $\mathcal L_1$‑Adaptive Model‑Based Control
- **6.2.1** Online parameter estimator & adaptation law
- **6.2.2** Bounded tracking‑error theorem

### 6.3 Ensemble Uncertainty & Risk Metrics
- **6.3.1** Epistemic–aleatoric decomposition
- **6.3.2** CVaR & other coherent risk measures

### 6.4 Safe Model‑Predictive Control
- **6.4.1** Terminal set & tube MPC with learned dynamics
- **6.4.2** Constraint‑tightening for probabilistic safety

### 6.5 Runtime Monitoring & Verification
- **6.5.1** Temporal‑logic specifications
- **6.5.2** Online falsification & fallback policies

---

## 7. Model‑Free Prediction

### 7.1 Problem Formulation
- **7.1.1** MDP refresher (states, actions, $\gamma$, policy)
- **7.1.2** Episodic vs continuing tasks
- **7.1.3** Return $G_t$; value-function target $v^\pi$
- **7.1.4** Mean-squared-error objective

### 7.2 Monte-Carlo Prediction — Why We Need TD
- **7.2.1** Full-return estimate & the variance problem
- **7.2.2** Incremental MC update
- **7.2.3** First bias–variance discussion

### 7.3 TD(0): One-Step Bootstrapping
- **7.3.1** TD target & TD-error $\delta_t$
- **7.3.2** On-line incremental update
- **7.3.3** Geometric view (DP vs MC)
- **7.3.4** Tabular convergence proof sketch

### 7.4 $n$-Step TD & $\lambda$-Return (Forward View)
- **7.4.1** Derivation of $G_t^{(n)}$
- **7.4.2** Continuum to MC
- **7.4.3** Weighted mixture $G_t^\lambda$
- **7.4.4** Analytical bias–variance curve

### 7.5 Eligibility Traces (Backward View)
- **7.5.1** Accumulating vs replacing traces
- **7.5.2** Proof of forward $\leftrightarrow$ backward equivalence (tabular)
- **7.5.3** TD($\lambda$), Sarsa($\lambda$), Watkins Q($\lambda$) update rules
- **7.5.4** *True-online TD($\lambda$)* motivation & algorithm

### 7.6 Analysis & Guarantees
- **7.6.1** Detailed bias–variance trade-off across $n$ and $\lambda$
- **7.6.2** Robbins-Monro conditions; linear-function-approx. convergence of TD(0) / TD($\lambda$)
- **7.6.3** Divergence counter-example under off-policy + function approx.
- **7.6.4** Practical heuristics (step-size schedules, resetting traces, $\lambda$-sweeps)

---

## 8. Model‑Free Control (On‑ & Off‑Policy)

### 8.1 Problem Formulation
- **8.1.1** Control objective $\displaystyle\max_\pi \mathbb{E}\!\left[\sum_{t=0}^\infty \gamma^{t} r_t\right]$
- **8.1.2** "Deadly triad" recap – bootstrapping ✕ function approximation ✕ off‑policy

### 8.2 Sarsa & On‑Policy TD Control
- **8.2.1** Tabular Sarsa update and convergence proof sketch
- **8.2.2** Expected‑Sarsa; variance comparison
- **8.2.3** $\lambda$‑extension with eligibility traces

### 8.3 Q‑Learning
- **8.3.1** Watkins Q‑learning update rule
- **8.3.2** Non‑asymptotic sample‑complexity $\tilde{O}\!\bigl(\tfrac{SA}{(1-\gamma)^3\varepsilon^2}\bigr)$
- **8.3.3** Double‑Q & Expected‑Q (bias correction)

### 8.4 Variance‑Reduced / Regularised Variants
- **8.4.1** Cascade Q‑learning; RegQ (provably convergent with linear FA)
- **8.4.2** Stability analysis under function approximation

### 8.5 Deep Q‑Networks & Rainbow
- **8.5.1** Replay buffer, target network heuristics
- **8.5.2** Distributional Q, prioritized replay, noisy nets, etc.

### 8.6 Actor‑Critic Methods
- **8.6.1** Policy‑gradient theorem & importance‑sampling ratios
- **8.6.2** DDPG, TD3, SAC; entropy regularisation

### 8.7 Convergence with Function Approximation
- **8.7.1** Baird counter‑example revisited
- **8.7.2** Gradient‑TD view; projected Bellman operator contraction conditions

### 8.8 Finite‑Sample Regret Bounds
- **8.8.1** Q‑learning + UCB exploration upper bounds
- **8.8.2** Matching lower bounds; optimality gaps

### 8.9 Open Questions
- **8.9.1** Non‑linear FA theory
- **8.9.2** Sample‑efficient exploration strategies

---

## 9. Off‑Policy Learning — Prediction & Control

### 9.1 Taxonomy
- **9.1.1** Distinguish OPE, prediction, control
- **9.1.2** Importance‑sampling vs density‑ratio methods

### 9.2 Importance‑Sampling Fundamentals
- **9.2.1** Ordinary vs weighted IS; variance properties
- **9.2.2** Capping, clipping, relative IS

### 9.3 Multi‑Step Off‑Policy TD
- **9.3.1** Tree‑Backup$(\lambda)$, ABTD$(\zeta)$, V‑trace$(\lambda)$
- **9.3.2** Bias‑variance trade‑offs

### 9.4 Off‑Policy Actor‑Critic
- **9.4.1** Deterministic policy gradient with IS
- **9.4.2** Twin‑critic and clipped‑weight tricks

### 9.5 Batch / Offline RL
- **9.5.1** Cross‑ref: Fitted Q‑Iteration (§4.4)
- **9.5.2** Conservative Q‑Learning (CQL); pessimistic bootstrapping

### 9.6 High‑Dimensional Action Spaces
- **9.6.1** State‑value‑only critics (e.g. Vlearn)

### 9.7 Theoretical Guarantees
- **9.7.1** PAC bounds for linear FA
- **9.7.2** Variance lower bound $O\!\bigl(\tfrac{1}{(1-\gamma)^4}\bigr)$

### 9.8 Practical Heuristics
- **9.8.1** Behaviour‑policy regularisation, trust regions, replay prioritisation

### 9.9 Summary & Open Problems
- **9.9.1** Safe policy improvement
- **9.9.2** Data‑quality diagnostics and benchmarks

---

## 10. Policy Search & Policy‑Gradient Methods

### 10.1 Foundations
- **10.1.1** Objective functions & the likelihood‑ratio gradient
- **10.1.2** Monte‑Carlo Policy Gradient (REINFORCE)
- **10.1.3** Baselines & variance‑reduction theory

### 10.2 Actor–Critic Architecture
- **10.2.1** Actor–Critic architecture and GAE
- **10.2.2** Natural Policy Gradient & compatible function approximation

### 10.3 Trust Region Methods
- **10.3.1** Trust‑Region PG and TRPO (monotonic‑improvement proof)
- **10.3.2** Proximal Policy Optimization (PPO) – clip vs KL penalty

### 10.4 Deterministic Policy Gradients
- **10.4.1** Deterministic PG, DDPG, TD3
- **10.4.2** Maximum‑Entropy RL and Soft Actor‑Critic (SAC)

### 10.5 Advanced Topics
- **10.5.1** Exploration, entropy regularisation, and KL constraints
- **10.5.2** Off‑policy corrections & importance sampling (IS, V‑trace, Q‑Prop)
- **10.5.3** Gradient‑free policy search (ES, CEM, GPS) – contrast to PG

### 10.6 Theoretical Analysis
- **10.6.1** Sample‑complexity bounds, bias‑variance analysis, lower bounds
- **10.6.2** Safe & constrained PG (CPO, Lagrangian, Lyapunov)

### 10.7 Implementation & Applications
- **10.7.1** Implementation pragmatics (normalisation, clipping, LR schedules)
- **10.7.2** Case studies (MuJoCo locomotion, Atari, robotics manipulation)

### 10.8 Research Frontiers
- **10.8.1** Research frontiers & open questions (credit assignment, large‑action spaces)

---

## 11. Partially Observable Reinforcement Learning

### 11.1 POMDP Foundations
- **11.1.1** Formal definition of a POMDP $(S,A,O,T,\Omega,R,\gamma)$
- **11.1.2** Bayes filter and belief state $b_{t+1}=\tau(b_t,a_t,o_{t+1})$
- **11.1.3** Optimality of belief‑stationary policies

### 11.2 Exact Planning in Belief Space
- **11.2.1** $\alpha$‑vector value iteration; piece‑wise‑linear‑convex value function
- **11.2.2** Complexity (PSPACE‑complete) of exact POMDP planning

### 11.3 Approximate Planning
- **11.3.1** Point‑Based Value Iteration (PBVI) and variants (HSVI, SARSOP)
- **11.3.2** Anytime guarantees and error bounds

### 11.4 Representation Learning under Partial Observability
- **11.4.1** Predictive State Representations (PSRs)
- **11.4.2** Recurrent memory architectures and finite‑memory controllers
- **11.4.3** Sample‑complexity bounds with windowed histories

### 11.5 PAC & Regret Guarantees
- **11.5.1** PAC‑RL for POMDPs with privileged simulators
- **11.5.2** Regret lower & upper bounds in latent‑state environments

### 11.6 Software & Benchmarks
- **11.6.1** SARSOP bindings (Python) and *pomdp* R package
- **11.6.2** Recommended small‑scale benchmarks (Tiger, Light‑Dark, RockSample)

### 11.7 Open Questions
- **11.7.1** Efficient exploration with latent states
- **11.7.2** Memory size vs. sample complexity trade‑off

---

## 12. Bayesian Reinforcement Learning

### 12.1 Bayes‑Adaptive MDPs (BAMDPs)
- **12.1.1** Unknown kernel as latent parameter $\theta$
- **12.1.2** Augmented state $(s_t,\theta_t)$ and equivalence to POMDP

### 12.2 Exact & Tree‑Search Planning
- **12.2.1** Bayes‑adaptive forward‑search (BFS3)
- **12.2.2** ADA‑MCTS and non‑stationary safe exploration

### 12.3 Posterior‑Sampling RL (PSRL)
- **12.3.1** Thompson sampling over MDP posterior
- **12.3.2** Regret bounds $\tilde{O}(\sqrt{HSA T})$

### 12.4 Variational & Approximate Bayesian RL
- **12.4.1** Evidence lower bound (ELBO) on value functions
- **12.4.2** Regret under approximation error

### 12.5 PAC‑Bayes & Lifelong RL
- **12.5.1** PAC‑Bayes generalisation bounds for RL
- **12.5.2** EPIC algorithm and distilled priors

### 12.6 Computation vs. Statistical Efficiency
- **12.6.1** Ensemble methods and bootstrap exploration
- **12.6.2** Connections to Section 4.4 (Fitted Q)

### 12.7 Open Questions
- **12.7.1** Bayesian exploration in continuous spaces
- **12.7.2** Structural priors and safe Bayesian RL

---

## 13. Imitation & Inverse Reinforcement Learning

### 13.1 Problem Formulation
- **13.1.1** MDP without a designed reward; expert demonstrations $\mathcal{D}$
- **13.1.2** Occupancy measures & divergence objectives

### 13.2 Behavioural Cloning
- **13.2.1** Supervised log‑likelihood objective
- **13.2.2** Compounding‑error bound in horizon $T$

### 13.3 Dataset Aggregation (DAgger)
- **13.3.1** Interactive expert‑label querying protocol
- **13.3.2** No‑regret analysis ⇒ constant imitation error

### 13.4 Offline Imitation Learning
- **13.4.1** Importance‑weighted BC under covariate shift
- **13.4.2** Statistical consistency & finite‑sample rates

### 13.5 Adversarial Imitation Learning (GAIL & variants)
- **13.5.1** Min‑max objective & JS‑divergence interpretation
- **13.5.2** Convergence guarantee via occupancy matching

### 13.6 Foundations of Inverse RL
- **13.6.1** Ill‑posedness & feature‑expectation matching
- **13.6.2** Apprenticeship‑learning value bound

### 13.7 Maximum‑Entropy IRL
- **13.7.1** MaxEnt principle & convex dual derivation
- **13.7.2** Sample complexity of partition‑function estimation

### 13.8 Bayesian & PAC‑style IRL
- **13.8.1** Posterior over reward hypotheses
- **13.8.2** PAC reward‑set estimation bounds

### 13.9 Adversarial IRL (AIRL)
- **13.9.1** Potential-based reward recovery
- **13.9.2** Dynamics-invariant transfer theorem

### 13.10 Preference-Based & Human-in-the-Loop RL
- **13.10.1** Binary preference models & active queries
- **13.10.2** Regret bounds for preference elicitation

### 13.11 Open Questions & Complexity Gaps
- **13.11.1** Lower bounds under partial observability
- **13.11.2** Robustness to corrupted demonstrations
- **13.11.3** Reward-poisoning attacks & defences

---

## 14. Hierarchical Reinforcement Learning & Temporal Abstraction

### 14.1 SMDP Foundations
- **14.1.1** Formal definition, call-and-return semantics, contraction proof

### 14.2 The Options Framework
- **14.2.1** Initiation sets, intra-option policies, termination; fundamental theorem

### 14.3 Option-Critic & Hierarchical Policy Gradient
- **14.3.1** Intra-option PG theorem, termination-gradient, convergence conditions

### 14.4 Option Discovery
- **14.4.1** Interest functions, diversity regularisation, skill chaining, eigen-options

### 14.5 Multi-Level Architectures
- **14.5.1** Manager/worker (Feudal, HAC), off-policy correction (HIRO)

### 14.6 Sample-Complexity Theory
- **14.6.1** Lower bounds for goal-conditioned HRL; conditions for polynomial speed-up

### 14.7 Representation Learning for Goals
- **14.7.1** Mutual-information objectives; latent-dimension vs optimality gap theorem

### 14.8 Transfer & Continual HRL
- **14.8.1** Option-indexing meta-learning; zero-shot reuse guarantees

### 14.9 Model-Based HRL
- **14.9.1** Hierarchical world-model factorisation; planning-error propagation bound

### 14.10 Applications & Case Studies
- **14.10.1** Robotics manipulation, listwise recommendation, vision-and-language tasks

### 14.11 Open Problems
- **14.11.1** Automatic granularity, hierarchical credit assignment, safe HRL

---

## 15. Multi-Agent RL & Stochastic Games

### 15.1 Foundations
- **15.1.1** Stochastic-game formalism, equilibrium concepts, occupancy measures

### 15.2 Dynamic Programming for Games
- **15.2.1** Shapley VI, double-oracle, hardness results, zero-sum Minimax-Q

### 15.3 Learning Algorithms
- **15.3.1** Independent Q, CTDE value-decomposition (VDN, QMIX, QTRAN, QPLEX), policy-gradient (MADDPG, MAPPO), mean-field RL

### 15.4 Theoretical Guarantees & Complexity
- **15.4.1** PAC & regret bounds, POSG hardness, mean-field convergence

### 15.5 Scalability, Credit Assignment & Communication
- **15.5.1** COMA, graph-based credit, robustness, curriculum/population training

### 15.6 Benchmarks & Applications
- **15.6.1** SMAC-v2, Google Football, multi-robot swarms, trading, traffic-lights

### 15.7 Open Questions
- **15.7.1** Equilibrium selection, hundreds-of-agents scalability, theory–practice gaps

---

## 16. Task Distributions & Transfer Principles
### 16.1 Sample-space formulation of an MDP family
### 16.2 Task-similarity metrics (f-divergence, bisimulation)
### 16.3 PAC-Bayes & regret decompositions across tasks
### 16.4 Transfer taxonomy (representation, dynamics, policy, reward)
### 16.5 Evaluation metrics (jump-start, forward/backward transfer)

---

## 17. Meta-Reinforcement Learning

### 17.1 Problem Statement
- **17.1.1** Bilevel optimisation over task distributions

### 17.2 Gradient-Based Meta-Learners
- **17.2.1** MAML, BO-MRL

### 17.3 Recurrent / Black-Box Meta-Learners
- **17.3.1** RL², hyper-networks

### 17.4 Probabilistic-Latent Methods
- **17.4.1** PEARL, context-VAE
 

### 17.5 Adaptation-Regret Theorems & Generalisation
- **17.5.1** Regret bounds vs task diversity; exploration–adaptation trade-off

---

## 18. Continual Reinforcement Learning

### 18.1 Non-Stationary MDPs & Task Sequences
- **18.1.1** Drift models, stability-plasticity dilemma

### 18.2 Algorithm Families
- **18.2.1** Regularisation (EWC, SI), memory replay, parameter isolation, meta-gradient CL

### 18.3 Theoretical Results
- **18.3.1** Stability-plasticity information bound; path-regret under drift

### 18.4 Benchmarks & Metrics
- **18.4.1** CL-score = BWT + FWT − Forgetting; MineRL-CL, Lifelong-ProcGen

### 18.5 Open Problems
- **18.5.1** Robust OOD adaptation, memory-efficient CL, long-horizon safety

---

## 19. Synthesis & Open Questions

### 19.1 Unified Bayesian Perspective
- **19.1.1** Linking representation, exploration, transfer, and continual learning
 

### 19.2 Exploration–Adaptation Trade-Off Across Timescales
- **19.2.1** Short-term uncertainty vs long-term generalisation

### 19.3 Scalability & Safety in Real-World Lifelong RL
- **19.3.1** Deployment constraints, risk metrics, monitoring

### 19.4 Research Gaps
- **19.4.1** Polynomial-time exploration with latent dynamics
- **19.4.2** Sample-efficient meta-adaptation in non-stationary tasks
- **19.4.3** Reliable evaluation under distribution shift
