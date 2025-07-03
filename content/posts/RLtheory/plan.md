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
2.1.6 Algorithmic description, convergence & ε-stopping rule  
2.1.7 Geometry of value functions (Dadashi et al.)  
2.1.8 Banach's Fixed-Point Theorem (background box)  
2.1.9 Linear-programming view (primal/dual)  
2.1.10 Value iteration as *approximate* planning  
2.1.11 Runtime of ε-optimal planning with VI  
2.1.12 Computational complexity of exact planning  
2.1.13 δ–ε error-control summary  
2.1.14 Upper & lower convergence bounds (tightness)  

### 2.2 Policy-Iteration Analysis

2.2.1 Definition of Policy Iteration algorithm  
2.2.2 Advantage function & Performance-Difference identity  
2.2.3 Geometric-Progress Lemma  
2.2.4 Geometric convergence of value error  
2.2.5 Strict-Progress Lemma (sub-optimal action elimination)  
2.2.6 Overall runtime bound (Scherrer)  
2.2.7 Value-Difference identity revisited  
2.2.8 Value- vs Policy-Iteration comparison  
2.2.9 Proof that PI ≥ VI speed  
2.2.10 Mixing rates & span-seminorm contraction  
2.2.11 Upper & lower runtime bounds (Ye; Feinberg-Huang-Scherrer)  
2.2.12 Measure-theoretic view of occupancy-measure projection  

## 3. Online Planning in Discounted MDPs

3.1 Motivation (curse of dimensionality; local planning)  
3.2 Access modes: global → local → online  
3.3 Environment model: black-box simulator  
3.4 Formal statement of the online-planning problem  
3.5 Optimisation language & oracle types  
3.6 Agent–environment interaction loop  
3.7 Online planner & δ-soundness definition  
3.8 Cost metrics (queries & arithmetic)  
3.9 Baseline algorithm (recursive value evaluation)  
3.10 Upper runtime bounds (deterministic & sparse-sampling)  
3.11 Matching lower bounds Ω($A^H$)  
3.12 Local vs online access trade-offs  
3.13 Sampling & averaging fundamentals  
3.14 Concentration tools (Hoeffding, union bounds, sub-Gaussian)  
3.15 Policy-error analysis (ε & "almost-ε" cases)  
3.16 Putting it together: choosing $m,H,ζ$ for δ-soundness  
3.17 Open questions & extensions (e.g. MCTS links)  

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
4.2.2 Approximate Policy Iteration theorem – $(\gamma^k)/(1-\gamma)+\varepsilon$ bounds  
4.2.3 API with approximate action-value functions (corollary)  
4.2.4 Least-Squares Policy Iteration (LSPI) algorithm & guarantees  

### 4.3 State Abstractions

4.3.1 Motivation (sample sharing; dimensionality reduction)  
4.3.2 Statistical trade-off: sample size vs approximation error  
4.3.3 Exact abstraction hierarchy (π*, Q*, model-irrelevance)  
4.3.4 Key theorems on the hierarchy & error bounds  
4.3.5 Improvements & variants (homomorphisms, utile distinctions)  
4.3.6 Approximate abstractions (ε-π*, ε-Q*, $ε_R,ε_P$ bisimulation)  
4.3.7 Bounding value loss for approximate abstractions  
4.3.8 Finite-sample analysis (n_φ(D), Hoeffding bound)  

### 4.4 Fitted Q-Iteration (Batch / Off-line RL)

4.4.1 Motivation (large/continuous spaces; sample-efficiency)  
4.4.2 Formal set-up & assumptions (realizability, Bellman-completeness, coverage $C$)  
4.4.3 Key lemmas (uniform deviation, one-step error, propagation)  
4.4.4 Main theorems (finite-sample & fast-rate bounds)  
4.4.5 Alternative analyses (non-stationary policies, perf-difference lemma)  
4.4.6 General-case discussion (relaxing assumptions; links to DQN)
