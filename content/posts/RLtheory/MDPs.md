---
date: "2025-06-30"
title: "Collection of MDPs"
summary: "Collection of MDPs"
category: "Taxonomy"
series: ["RL Theory"]
author: "ChatGPT"
hero: /assets/images/hero3.png
image: /assets/images/card3.png
---

Below is a **fully‑expanded, up‑to‑date taxonomy of Markov Decision Processes (MDPs)** that keeps *nothing* off the table.
Rather than a single tree, it is organised as **ten orthogonal axes**; any concrete control problem is a point in this 10‑dimensional design space.  For each axis you will find

* formal definition(s) and assumptions,
* representative sub‑classes,
* illustrative examples,
* common solution tools or theorems that become available when the assumptions hold.

The axes are numbered so you can reference them later (e.g. “axis 3 = POMDP, axis 7 = risk‑sensitive”).

---

## 0 Canonical kernel

An MDP is a 5‑tuple

$$
\mathcal M=\langle\mathcal S,\mathcal A,P,R,\Xi\rangle ,
$$

where $\Xi$ collects *horizon / return* hyper‑parameters (γ, time limit $T$, goal set $G$, etc.).
A *policy* $\pi$ induces a controlled Markov chain and return $G^{\pi}$ whose exact definition depends on axis 2 below.

---

## Axis 1 Time model

| Sub‑class                               | Clock                           | Governing equation                                     | Key algorithm(s)                            |
| --------------------------------------- | ------------------------------- | ------------------------------------------------------ | ------------------------------------------- |
| **Discrete‑time MDP**                   | $t\in\mathbb N$                 | Bellman optimality: $T [V] (s)=\max_a \lbrace R+\mathbb E[V] \rbrace$ | Value Iteration (VI), Policy Iteration (PI) |
| **Semi‑Markov decision process (SMDP)** | Random holding time $\tau(s,a)$ | Generalised Bellman with duration                      | Options framework, LAO\*                    |
| **Continuous‑time MDP (CTMDP)**         | $t\in\mathbb R_{\ge0}$          | Hamilton–Jacobi–Bellman (HJB) PDE or generator $Q$     | Uniformisation, policy gradient in CT       |

---

## Axis 2 Horizon & return criterion (discounted vs. undiscounted lives here)

| Sub‑class                                 | Return $G^{\pi}$                                                      | Structural assumption ensuring finiteness                | Comments                                          |
| ----------------------------------------- | --------------------------------------------------------------------- | -------------------------------------------------------- | ------------------------------------------------- |
| **Finite horizon**                        | $\sum_{t=0}^{T-1} R_t$                                                | $T<\infty$                                               | Dynamic programming backwards in time             |
| **Infinite horizon, γ‑discounted**        | $\sum_{t=0}^{\infty}\gamma^{t} R_t \; \gamma \in [0,1)$             | $\gamma < 1$ ⇒ Bellman operator is a contraction                  | Most RL textbooks default                         |
| **Average‑reward (ergodic)**              | $\displaystyle\rho^{\pi}=\lim_{T\to\infty}\frac1T\sum_{t=0}^{T-1}R_t$ | Every stationary policy induces positive‑recurrent chain | Requires *relative* value iteration or R‑learning |
| **Stochastic shortest path (total cost)** | $\mathbb E \left[\sum_{t=0}^{\tau_G-1}c_t\right]$                    | $\exists$ *proper* policy with $P(\tau_G<\infty)=1$              | Used for goal‑directed planning                   |
| **Risk‑adjusted returns**                 | CVaR, exponential utility, mean–variance, etc.                        | See axis 7 for risk models                               | Turns Bellman into nonlinear operator             |

---

## Axis 3 Observability

1. **Fully observable MDP** – decision maker sees $s_t$.
2. **Partially observable MDP (POMDP)** – agent gets $o_t\sim O(o\mid s_t)$; belief $b_t$ is sufficient statistic.
3. **Mixed observability (MOMDP)** – state splits into visible and hidden factors, speeding up belief updates.

---

## Axis 4 State & action cardinality / structure

* **Finite / tabular** – exact DP and tabular RL possible.
* **Continuous ($\mathbb{R}^n$)** – needs function approximation; special case *LQR* (linear dynamics, quadratic cost) is analytically solvable via Riccati.
* **Factored MDP** – state is a tuple with sparse dynamic Bayesian‑network factorisation; enables structured VI.
* **Hybrid discrete–continuous** – common in robotics; solved with hybrid MPC or mixed‑integer RL.

---

## Axis 5 Transition determinism

* **Deterministic MDP** – $P(s'|s,a)$ is Dirac; planning ≈ graph search (A\*, Dijkstra).
* **Stochastic MDP** – genuine uncertainty; need expectations in Bellman equation.

---

## Axis 6 Stationarity in dynamics & rewards

| Class                               | Definition                                              | Typical solver                                       |
| ----------------------------------- | ------------------------------------------------------- | ---------------------------------------------------- |
| **Stationary**                      | $P,R$ independent of $t$.                               | Standard DP / RL.                                    |
| **Time‑inhomogeneous**              | $P_t, R_t$ depend on $t$.                               | Augment state with clock or use non‑stationary DP.   |
| **Adversarial / non‑stationary RL** | Environment may change arbitrarily; no Markov guarantee | Regret‑minimising online algorithms (UCRL2, SW‑DQN). |

---

## Axis 7 Risk, robustness & constraints

| Variant                                  | Objective / constraint                             | Representative methods                |
| ---------------------------------------- | -------------------------------------------------- | ------------------------------------- |
| **Risk‑neutral**                         | Max expected return (default).                     | DP, Q‑learning                        |
| **Risk‑sensitive**                       | Minimise CVaR, entropic risk, etc.                 | CVaR‑VI, variance‑aware PG            |
| **Constrained MDP (CMDP)**               | Max return s.t. $\mathbb E[C_i]\le \beta_i$.       | Lagrangian relaxation, Primal‑dual RL |
| **Robust / distributionally‑robust MDP** | Max worst‑case return over $\mathcal P$.           | Min‑max DP, ambiguity‑set LP          |
| **Safe MDP**                             | Hard safety predicates on states or probabilities. | Shielded RL, reach‑avoid VI           |

---

## Axis 8 Multi‑agent scope

| Model                           | Interaction                        | Typical solution concept                                 |
| ------------------------------- | ---------------------------------- | -------------------------------------------------------- |
| **Multi‑agent MDP (MMDP)**      | Shared full state, cooperative.    | Joint VI; centralised training.                          |
| **Decentralised MDP (Dec‑MDP)** | Local observations, shared reward. | Joint‑belief planning, coordination graphs.              |
| **Markov / stochastic game**    | Competing or mixed motives.        | Nash or correlated equilibrium algorithms (VI++, Foe‑Q). |

---

## Axis 9 Temporal abstraction & hierarchy

* **Options / macro‑actions (SMDP)** – each option is a policy with termination condition; planning in option space accelerates learning.
* **MAXQ, Ha‑MDP, feudal RL** – decompose value function or policy hierarchically; retains optimality under certain assumptions.

---

## Axis 10 Algebraic or domain‑specific specialisations

| Name                                | Defining property                                    | Pay‑off                                      |                                              |
| ----------------------------------- | ---------------------------------------------------- | -------------------------------------------- | -------------------------------------------- |
| **Linearly solvable MDP (LMDP)**    | Passive dynamics + KL‑control cost ⇒ linear Bellman. | Exact inference via eigenvector.             |                                              |
| **Entropy‑regularised / soft MDP**  | Adds $(\alpha\mathcal H(\pi(\cdot s)))$ to reward.                              | Smooths optimisation; underlies soft‑Q, SAC. |
| **Restless‑bandit / Whittle index** | Independent evolving arms, one active.               | Near‑optimal index policies.                 |                                              |
| **Queueing / inventory MDPs**       | Special monotone structure.                          | Policy monotonicity ⇒ threshold rules.       |                                              |
| **Impulsive / hybrid jump MDPs**    | Combine continuous flows with discrete jumps.        | Impulse control, mixed‑integer optimisation. |                                              |

---

## How to read the map

1. **Locate the physics** (axis 1): discrete, semi‑Markov, or continuous time?
2. **Specify the performance metric** (axis 2): discounted vs. undiscounted variants.
3. **Write down what you can sense** (axis 3) and the shape of state/action spaces (axis 4).
4. **Ask whether uncertainty, non‑stationarity, risk, or multiple agents** are core (axes 5–8).
5. **Exploit structure**—factorisation, hierarchy, or special linear‑quadratic forms (axes 9–10).

Because the axes are **independent**, a single problem can sit in many boxes—e.g.:

*Autonomous warehouse robot:*
  axis 1 = SMDP (macro‑actions),
  axis 2 = discounted infinite horizon,
  axis 3 = POMDP,
  axis 4 = continuous state & action,
  axis 6 = slow seasonal non‑stationarity,
  axis 7 = CMDP with collision constraints.

---

### Closing advice

*Deciding which variant you truly have is not bookkeeping—it determines the **theoretical guarantees** and **computational tractability** of every algorithm you might run.*
When drafting a model, explicitly list the axis values you are assuming; if an assumption is uncertain (e.g. ergodicity for average‑reward), mark it as a hypothesis and design an empirical test or a robust formulation to hedge against violation.
