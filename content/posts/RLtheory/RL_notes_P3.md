---
date: "2025-07-04"
title: "(Part 3) Online vs Local Planning in Sequential Decision‑Making"
summary: "Dissecting two practical faces of planning in RL‑style control. Traeat as advanced planning heuristics; optional deep dive"
category: "Tutorial"
series: ["RL Theory"]
author: "Bryan Chan"
hero: /assets/images/hero_online_local.png
image: /assets/images/card_online_local.png
sources:
  - title: "Sutton & Barto — RL (2nd Ed.) §8"
    url: "https://web.stanford.edu/class/psych209/Readings/SuttonBartoIPRLBook2ndEd.pdf"
  - title: "Silver et al., *Mastering the Game of Go without Human Knowledge* (2017)"
    url: "https://www.nature.com/articles/nature24270"
---

## TL;DR

### The Core Idea

**Planning** acts as a *thinking loop* wedged between sensing and acting. Two staples are:

* **Online planning** – re‑solve a *small* search problem every time step, conditioning on the **current** state.
* **Local planning** – optimize over a *short finite horizon* around the present, often with a rolling or "receding" window.

### Why It Matters

Real‑world agents face tight compute budgets and non‑stationary data streams. Both paradigms trade optimality for *responsiveness* – but in *different* ways.

### Take‑Home

Think of online planning as *"plan‑as‑you‑go"*; think of local planning as *"look‑ahead‑but‑not‑too‑far"*. Mastering when to prefer which is crucial in robotics, games, and embedded control.

---

## Landscape of Planning in RL

High‑level map situating **online** and **local** planning inside the broader taxonomy of sequential decision‑making.

## Mathematical Foundations

### The Planning Lens on MDPs

Recap of the 5‑tuple $\langle S,A,P,R,\gamma\rangle$ with an emphasis on *search vs. rollout* viewpoints.

### Policies, Value Functions & Look‑ahead Operators

Define $V$, $Q$, and an **$h$‑step look‑ahead Bellman operator** $\mathcal{T}^h$.

---

## Online Planning

### Formal Definition

Let $\mathcal{M}=\langle S,A,P,R,\gamma\rangle$ be an MDP and let
$\text{SEARCH}(s,\text{budget})$ denote any bounded‑depth tree‑search operator returning an **action distribution**.
An **online planner** executes

$$
a_t \;\sim\; \pi_t(\cdot\mid s_t)\;:=\;\text{SEARCH}\!\Bigl(s_t,\; \text{budget}_t\Bigr),
$$

where

* $s_t$ is *only* the **current** state (no history),
* $\text{budget}_t$ is measured in node expansions, rollouts, or wall‑clock milliseconds,
* the planner is re‑invoked **every time‑step** before acting.

Hence online planning is *plan‑as‑you‑go*: solve a *fresh*, *shallow* search problem at each decision point.

### Canonical Algorithms

| Family                                    | Key Mechanism                                                             | Hallmark Applications                   |
| ----------------------------------------- | ------------------------------------------------------------------------- | --------------------------------------- |
| **LRTA★ / RTDP**                          | 1‑step look‑ahead with *on‑policy* value backups; learns heuristic online | Robotics, grid navigation               |
| **D*‑Lite**                              | Incremental A★; quickly repairs previous tree after environment changes   | Mars rovers, path‑finding in games      |
| **Monte‑Carlo Tree Search (MCTS)**        | Stochastic rollout evaluation + UCB exploration; *anytime*                | Go (AlphaGo → AlphaZero), Atari, MuZero |
| **Forward Search Sparse Sampling (FSSS)** | PAC‑optimal tree growth with rollouts to depth $d$                      | Theoretical benchmarks                  |

All share two traits:

1. **State‑rooted trees** (vs. trajectories) so each replanning step conditions on *latest* perception.
2. **Anytime quality**: a valid (though sub‑optimal) action is available once the interrupt flag is raised.

### Anytime & Interruptible Properties

Define the *value residual* after budget $B$ as

$$
\varepsilon_B(s_t)\;=\;\max_{a}\Bigl|Q^{*}(s_t,a) - Q^{\text{tree}}_B(s_t,a)\Bigr|,
$$

where $Q^{\text{tree}}_B$ is the depth‑ and rollout‑truncated estimate.
Typical planners guarantee

$$
\varepsilon_B(s_t)\;=\;\mathcal{O}\!\bigl(B^{-\alpha}\bigr),
$$

for some algorithm‑specific rate $\alpha>0$ (e.g., $\alpha=\tfrac{1}{2}$ for pure Monte‑Carlo rollouts under Hoeffding bounds). Practically, a developer picks $B$ so that $\varepsilon_B$ falls below a *latency‑dictated* threshold.

### Computational Footprint

For branching factor $b$ and depth $d$:

* **Full expansion**: $\Theta(b^d)$ — infeasible except for toy problems.
* **Stochastic rollout planners (MCTS, RTDP)**: $\Theta(B)$ where $B\ll b^d$; **linear** in budget.
* **Memory**: Either $\mathcal{O}(B)$ (explicit tree) or $\mathcal{O}(d)$ (single rollout path).

Latency is thus tunable **online**: agents can trade decision quality for real‑time responsiveness by modulating $B$ at run‑time.

### Why Use Online Planning?

* **Dynamic environments** — model updates invalidate long offline plans; online search adapts instantly.
* **Limited models** — even with unknown transitions, learned simulators (e.g., MuZero's latent dynamics) feed online search.
* **Embedded hardware** — real‑time systems (quadrotors, mobile games) can't afford global DP but can spare milliseconds for a focused look‑ahead.

---

## Local Planning

### Formal Definition

Given the same MDP $\mathcal{M}=\langle S,A,P,R,\gamma\rangle$, an **$H$‑step local planner** solves

$$
\max_{a_{0:H-1}}\; \mathbb{E}\!\Bigl[\,
\sum_{k=0}^{H-1}\gamma^{k}\,R\bigl(s_{t+k},a_k\bigr)\;+\;\gamma^{H}\,V_{\text{tail}}\bigl(s_{t+H}\bigr)
\;\Bigr]\!,
\tag{1}
$$

subject to the dynamics $s_{t+k+1}\sim P(\cdot \mid s_{t+k},a_k)$ and *then* executes **only the first control**
$a_t^* = a_0^*$.
After the environment returns $s_{t+1}$, the horizon window *slides* and (1) is re‑solved.

Key ingredients:

1. **Finite horizon $H$** – a design hyper‑parameter (often $H\ll\tfrac{1}{1-\gamma}$).
2. **Tail value $V_{\text{tail}}$** – heuristic, learned critic, or simply $0$ when no estimate exists.
3. **Receding‑horizon loop** — replan at *every* step, combating model error and disturbances.

### Model Predictive Control (MPC)

Local planning under the name **MPC** dominates modern robotics and process control:

| Primitive       | MPC Instantiation                                             |
| --------------- | ------------------------------------------------------------- |
| **Objective**   | Quadratic cost or (1) with arbitrary reward                   |
| **Dynamics**    | Known nonlinear ODEs / learned neural models                  |
| **Solver**      | Sequential Quadratic Programming, iLQR, shooting, collocation |
| **Constraints** | State and action bounds, safety sets, energy budgets          |

A canonical receding‑horizon algorithm is:

```pseudo
loop  t = 0,1,...
    (a_0:H-1)*  ←  argmin_{a_{0:H-1}} J_H(s_t, a_{0:H-1})
    apply a_t = a_0*
    observe s_{t+1}
end loop
```

where $J_H$ is the *negative* of (1) if we phrase control as minimization. Stability and recursive feasibility are certified when (i) constraints are convex and (ii) $V_{\text{tail}}$ is a *control‑Lyapunov* function – classic MPC theory guarantees closed‑loop boundedness under those assumptions.

### Trajectory‑Optimization Toolkits

* **iLQR / DDP** – second‑order expansion of dynamics + cost; warm‑start each new horizon with last iterate.
* **Cross‑Entropy Method (CEM)** – sample‑based optimizer over action sequences; popular in *PETS*, *Dreamer‑MPC*.
* **Gradient‑based shooting** – direct back‑prop through dynamics; leverages automatic differentiation for learned models.
* **Collocation (direct multiple shooting)** – convert dynamics into equality constraints on knot points; large sparse QP.

Each method approximates the argmax in (1) under different smoothness or black‑box assumptions.

### Sample‑Efficiency & Model‑Error Sensitivity

Short horizons **truncate error propagation**: linearization inaccuracies accumulate only over $H$ steps, not forever.
Yet the agent pays a *computational rent*: solving (1) at **every** time step costs $\mathcal{O}\!\bigl(\text{solver}(H)\bigr)$.

* **Perfect‑model regime**: longer $H$ → higher optimality, diminishing returns after the effective discount horizon.
* **Imperfect‑model regime**: there exists a *sweet‑spot* $H^*$ where truncation error ≈ model error; empirical tuning required.

### Computational Footprint

Assume action dimension $m$, horizon $H$, and cost‑function evaluation $\mathcal{O}(1)$.

* **Gradient methods**: $\Theta(Hm)$ per iteration; iterations depend on non‑convexity.
* **Shooting with $N$ rollouts (CEM)**: $\Theta(NH)$; trivially parallelizable on GPUs.
* **Memory**: $\mathcal{O}(H)$ for state trajectory or $\mathcal{O}(NH)$ for ensemble methods.

### Why Use Local Planning?

* **High‑bandwidth control** — quadrotors at 200 Hz can still afford $H=20$ steps into the future.
* **Safety‑critical constraints** — explicit incorporation of hard bounds unachievable by tree search.
* **Continuous high‑dim spaces** — gradient‑based optimizers exploit smooth structure; tree search explodes exponentially.

---

## Comparative Lens

| Axis                         | **Online Planning**                                          | **Local Planning**                                               |
| ---------------------------- | ------------------------------------------------------------ | ---------------------------------------------------------------- |
| **Model Requirement**        | Optional (simulator or learned dynamics suffices)            | Required (explicit dynamics in solver)                           |
| **Horizon**                  | Adaptive, potentially unbounded but shallow per call         | Fixed, short ($H\ll(1-\gamma)^{-1}$)                         |
| **Anytime Quality**          | **Yes** — action available after any budget                  | Rare — solvers need convergence for feasibility                  |
| **Constraints**              | Hard to encode; soft penalties only                          | Native support for hard state/action constraints                 |
| **State/Action Spaces**      | Works in discrete or mixed; struggles in high‑dim continuous | Excels in smooth, high‑dim continuous spaces                     |
| **Compute per Step**         | Tunable $\Theta(B)$; tree reuse limited                    | Solver complexity $\Theta(\text{solver}(H))$; warm‑starts help |
| **Typical Use‑Cases**        | Board games, stochastic domains, unknown models              | Robotics, autonomous driving, process control                    |
| **Failure Mode**             | Exponential blow‑up if branching factor $b$ large          | Infeasible or unsafe if optimization interrupted                 |
| **Representative Algorithm** | MCTS (AlphaZero)                                             | MPC with iLQR or CEM                                             |

---

## When to Prefer Which?

1. **Latency‑critical + Uncertain Model → Online Planning**
   *Single‑board game engines, real‑time strategy (RTS) AI, learned‑model RL (e.g., MuZero).*

2. **Constraint‑dominated + Smooth Dynamics → Local Planning**
   *Drones at 200 Hz, race‑car MPC, industrial chemical reactors.*

3. **Hybrid Stacks**
   *State lattice planning:* global coarse tree search (online) to propose corridors; local MPC refines trajectory inside corridor.

4. **Compute Budget Scaling**
   *If GPU rollout budget grows faster than solver speed‑ups, favor online. If solver warm‑starts and gradients amortize, favor local.*

5. **Safety/Risk Profile**
   *Safety‑critical missions demand constraint satisfaction ⇒ local planning with robust tubes. Exploratory environments tolerate occasional sub‑optimal moves ⇒ online planning.*

---

## Wrap‑Up

Understanding **where** each paradigm shines lets practitioners architect layered systems: global **online planning** for strategic foresight, nested **local planning** loops for fast, constraint‑respecting control. Mastery of this tool‑set widens the design space of RL‑enabled agents in robotics, games, and autonomous systems.

---

## Appendix

### Pseudocode: Real-Time Dynamic Programming (RTDP)

```pseudo
procedure RTDP(s_start, budget)
    while budget > 0 do
        s ← s_start
        while s ≠ terminal do
            a ← greedy(s, V)  // ε-greedy for exploration
            s' ← sample_successor(s, a)
            V[s] ← R(s,a) + γ * V[s']  // backup
            s ← s'
            budget ← budget - 1
        end while
    end while
    return greedy_action(s_start, V)
end procedure
```

### Pseudocode: Model Predictive Control (MPC)

```pseudo
procedure MPC(s_t, horizon_H)
    x_init ← [s_t, 0, 0, ..., 0]  // initial guess
    for iter = 1 to max_iterations do
        grad ← ∇_a J_H(s_t, a_{0:H-1})
        a_{0:H-1} ← a_{0:H-1} - α * grad
        if ||grad|| < tolerance then break
    end for
    return a_0  // first action only
end procedure
```
