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

## 1. MDP Foundations & Optimality

1.1 Measure-theoretic set-up  
1.2 Formal definition of an MDP $(S,A,P,r,\gamma)$  
1.3 Discount factor & effective horizon $H_{\gamma,\varepsilon}$  
1.4 Policies and the agent–environment loop  
1.5 Probability of trajectories (Ionescu–Tulcea)  
1.6 Return, value functions & optimality notions  
1.7 Memoryless vs general policies; fundamental optimality theorem  
1.8 Objective functions & modelling choices  
1.9 Discounted occupancy measures  
1.10 Bellman operators $T^\pi,\,T$  
1.11 Key analytical properties (contraction, error bounds)  
1.12 Greedy policies & the Fundamental Theorem  

## 2. Exact Dynamic Programming

### 2.1 Value-Iteration Theory

2.1.1 Fundamental Theorem of Dynamic Programming (finite MDPs)  
2.1.2 Effective horizon & Bellman update $v_{k+1}=Tv_k$  
2.1.3 Policy-error (greedy) bound  
2.1.4 Fixed-point iteration via Banach's lemma  
2.1.5 Finite-horizon interpretation  
2.1.6 Algorithmic description, convergence & $\varepsilon$-stopping rule  
2.1.7 Geometry of value functions (Dadashi et al.)  
2.1.8 Banach's Fixed-Point Theorem (background box)  
2.1.9 Linear-programming view (primal/dual)  
2.1.10 Value iteration as *approximate* planning  
2.1.11 Runtime of $\varepsilon$-optimal planning with VI  
2.1.12 Computational complexity of exact planning  
2.1.13 $\delta$–$\varepsilon$ error-control summary  

### 2.2 Policy-Iteration Analysis

2.2.1 Definition of Policy Iteration algorithm  
2.2.2 Advantage function & Performance-Difference identity  
2.2.3 Geometric-Progress Lemma  
2.2.4 Geometric convergence of value error  
2.2.5 Strict-Progress Lemma (sub-optimal action elimination)  
2.2.6 Overall runtime bound (Scherrer)  
2.2.7 Value- vs Policy-Iteration comparison  
2.2.8 Proof that PI $\geq$ VI speed  
2.2.9 Mixing rates & span-seminorm contraction  
2.2.10 Upper & lower runtime bounds (Ye; Feinberg-Huang-Scherrer)  
2.2.11 Measure-theoretic view of occupancy-measure projection  

## 3. Online Planning in Discounted MDPs

3.1 Motivation (curse of dimensionality; local planning)  
3.2 Access modes: global → local → online  
3.3 Environment model: black-box simulator  
3.4 Formal statement of the online-planning problem  
3.5 Optimisation language & oracle types  
3.6 Agent–environment interaction loop  
3.7 Online planner & $\delta$-soundness definition  
3.8 Cost metrics (queries & arithmetic)  
3.9 Baseline algorithm (recursive value evaluation)  
3.10 Upper runtime bounds (deterministic & sparse-sampling)  
3.11 Matching lower bounds $\Omega(A^H)$  
3.12 Local vs online access trade-offs  
3.13 Sampling & averaging fundamentals  
3.14 Policy-error analysis ($\varepsilon$ & "almost-$\varepsilon$" cases)  
3.15 Putting it together: choosing $m,H,\zeta$ for $\delta$-soundness  
3.16 Open questions & extensions (e.g. MCTS links)  

## 4. Value-Function Approximation & Approximate Policy Iteration

### 4.1 Value-Function Approximation

4.1.1 Approximate Universal Value-Function Realizability (Assump. B2)  
4.1.2 Approximate Policy Evaluation – MC rollout + weighted LS  
4.1.3 Extrapolation-error control in weighted LS  
4.1.4 Kiefer–Wolfowitz Theorem (G-optimal design)  
4.1.5 Corollary – extrapolation control via optimal design  
4.1.6 LSPE-G high-probability error bound  

### 4.2 Approximate Policy Iteration

4.2.1 Geometric Progress Lemma with approximate improvement  
4.2.2 Approximate Policy Iteration theorem – $\frac{\gamma^k}{1-\gamma}+\varepsilon$ bounds  
4.2.3 API with approximate action-value functions (corollary)  
4.2.4 Least-Squares Policy Iteration (LSPI) algorithm & guarantees  

### 4.3 State Abstractions

4.3.1 Motivation (sample sharing; dimensionality reduction)  
4.3.2 Statistical trade-off: sample size vs approximation error  
4.3.3 Exact abstraction hierarchy ($\pi^\ast$, $Q^\ast$, model-irrelevance)  
4.3.4 Key theorems on the hierarchy & error bounds  
4.3.5 Improvements & variants (homomorphisms, utile distinctions)  
4.3.6 Approximate abstractions ($\varepsilon$-$\pi^\ast$, $\varepsilon$-$Q^\ast$, $\varepsilon_R,\varepsilon_P$ bisimulation)  
4.3.7 Bounding value loss for approximate abstractions  
4.3.8 Finite-sample analysis ($n_\phi(D)$, Hoeffding bound)  

### 4.4 Fitted Q-Iteration (Batch / Off-line RL)

4.4.1 Motivation (large/continuous spaces; sample-efficiency)  
4.4.2 Formal set-up & assumptions (realizability, Bellman-completeness, coverage $C$)  
4.4.3 Key lemmas (uniform deviation, one-step error, propagation)  
4.4.4 Main theorems (finite-sample & fast-rate bounds)  
4.4.5 Alternative analyses (non-stationary policies, perf-difference lemma)  
4.4.6 General-case discussion (relaxing assumptions; links to DQN)

## 5. Sampling and Computational Complexity

### 5.1 Foundations & Notation

5.1.1 Markov decision processes (states, actions, rewards, $\gamma/H$)  
5.1.2 Feature maps $\phi$, linear value/action-value parametrisations  
5.1.3 Norms and error metrics ($\|\cdot\|_\infty$, $\|\cdot\|_2$)  
5.1.4 Pre-computed core sets and LSPI recap  

### 5.2 Probabilistic Tools for Sampling Analysis

5.2.1 Hoeffding's inequality – statement, proof sketch, intuition  
5.2.2 Azuma–Hoeffding for martingales (variance proxy, step sizes)  
5.2.3 Union-bound "upgrade" for simultaneous guarantees  
5.2.4 Worked example: uniform-sampling best-arm identification   

### 5.3 Covering Numbers & Uniform Convergence

5.3.1 $\ell_\infty$ covers and growth with dimension $d$  
5.3.2 Lipschitz compositions and loss-class covering  
5.3.3 Sample-complexity bound $\tilde{O}\left(\sqrt{\frac{\log N_\varepsilon}{n}}\right)$  

### 5.4 Limits of Query‑Efficient Planning

5.4.1 Definition of $({\delta, \varepsilon})$-sound online planners  
5.4.2 **Large-$A$ lower bound**  
   - Johnson–Lindenstrauss packing lemma  
   - High-probability "needle" lemma  
   - Exponential query cost in $A$ and $\sqrt{d}$  
5.4.3 **Fixed-horizon, small-$A$ lower bound**  
   - Horizon-dependent fundamental theorem  
   - Query complexity $\tilde{\Omega}(AH/H)$ vs "large-$H$" regime  

### 5.5 Planning under Realizability

5.5.1 **$q^*$-realizability** – linear assumption, global planner variant  
5.5.2 **$v^*$-realizability (TensorPlan)**  
   - Interaction protocol & local simulator calls  
   - Ridge regression hypothesis set $\Theta$  
   - Optimism via square-root bonus ($\beta$)  
   - Covering-number analysis (size $\tilde{O}(d^2)$)  

### 5.6 Exploration in Linear MDPs

5.6.1 Problem set-up and boundedness assumptions  
5.6.2 LSVI-UCB algorithm – step-wise ridge with UCB bonus  
5.6.3 Martingale concentration (Azuma) for adaptive data  
5.6.4 Elliptical-potential lemma & regret $\tilde{O}(H^2\sqrt{dT})$  

### 5.7 Linear‑Programming View of MDPs

5.7.1 Primal LP: minimise $d_0^\top V$ s.t. $V \geq TV$  
5.7.2 Dual LP: occupancy-measure constraints, $d^\pi$ cone  
5.7.3 Monotonicity of $T$ and convergence to $V^*$  

### 5.8 Synthesis & Open Questions

5.8.1 Comparison table: lower vs upper bounds ($d, A, H, \gamma$)  
5.8.2 Gaps: horizon terms in discounted setting, constant-$d$ long-horizon planning  
5.8.3 Computational vs query complexity – can TensorPlan be made efficient?  

## 6. Model Free Prediction

### 6.0 Problem Formulation

6.0.1 MDP refresher (states, actions, $\gamma$, policy)  
6.0.2 Episodic vs continuing tasks  
6.0.3 Return $G_t$; value-function target $v^\pi$  
6.0.4 Mean-squared-error objective  

### 6.1 Monte-Carlo Prediction — Why We Need TD

6.1.1 Full-return estimate & the variance problem  
6.1.2 Incremental MC update  
6.1.3 First bias–variance discussion  

### 6.2 TD(0): One-Step Bootstrapping

6.2.1 TD target & TD-error $\delta_t$  
6.2.2 On-line incremental update  
6.2.3 Geometric view (DP vs MC)  
6.2.4 Tabular convergence proof sketch  

### 6.3 $n$-Step TD & $\lambda$-Return (Forward View)

6.3.1 Derivation of $G_t^{(n)}$  
6.3.2 Continuum to MC  
6.3.3 Weighted mixture $G_t^\lambda$  
6.3.4 Analytical bias–variance curve  

### 6.4 Eligibility Traces (Backward View)

6.4.1 Accumulating vs replacing traces  
6.4.2 Proof of forward $\leftrightarrow$ backward equivalence (tabular)  
6.4.3 TD($\lambda$), Sarsa($\lambda$), Watkins Q($\lambda$) update rules  
6.4.4 *True-online TD($\lambda$)* motivation & algorithm  

### 6.5 Analysis & Guarantees

6.5.1 Detailed bias–variance trade-off across $n$ and $\lambda$  
6.5.2 Robbins-Monro conditions; linear-function-approx. convergence of TD(0) / TD($\lambda$)  
6.5.3 Divergence counter-example under off-policy + function approx.  
6.5.4 Practical heuristics (step-size schedules, resetting traces, $\lambda$-sweeps)
