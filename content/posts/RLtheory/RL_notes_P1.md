---
date: "2025-06-27"
title: "(Part 1) MDP Fundamentals and fixed-point theory" 
summary: "Aim to provide more insight on RL foundations for beginners. Includes MDP definitions, bellman eq, funda thm, value functions, contrction mappings. Ensure rigorous definitions of MDP, value functions, contraction mappings."
category: "Tutorial"
series: ["RL Theory"]
author: "Bryan Chan"
hero: /assets/images/hero3.png
image: /assets/images/card3.png
sources:
  - title: "Lecture 1 (2022-01-05)"
    url: "http://www.youtube.com/watch?v=rjwxqcVrVws"
  - title: "Lecture 1 (2021-01-12)"
    url: "http://www.youtube.com/watch?v=0oJmSULoj3I"
---

## TL;DR: 

### The Core Idea
Reinforcement Learning: agents learn optimal decisions in uncertain environments through trial and error, formalized using **Markov Decision Processes (MDPs)**.

### Mathematical Foundation
**MDP**: States, Actions, Transition probabilities, Rewards, discount factor. 

Agent follows a **policy** to maximize expected future rewards. 

Key insight: current state contains all info needed for optimal decisions.

### Three Domains of RL
**1. Planning**: Environment is known → find optimal actions via dynamic programming/search

**2. Offline RL**: Learn from fixed dataset → avoid trying actions not in data  

**3. Online RL**: Learn while interacting → balance exploration vs exploitation

### The Central Challenge
**Problem**: Learn optimal behavior without knowing how the environment works

**Solution**: Must explore (try new actions) even if it hurts short-term performance

### Bottom Line
RL provides a unified framework for sequential decision-making under uncertainty. The core challenge is balancing exploration (learning) with exploitation (performing well) across three different practical scenarios.

---

# Landscape of Reinforcement Learning

## Mathematical Foundations

### Measure-Theoretic Preliminaries

The exposition that follows relies on measurable-space machinery, which we state up-front for self-containment.

* **Measurable spaces.** $(S,\Sigma_S)$ and $(A,\Sigma_A)$ are measurable spaces (typically Borel spaces when $S$ or $A$ are Polish).

* **Probability kernel.** A map $K:(X,\Sigma_X)\times\Sigma_Y \to [0,1]$ is a kernel if $K(x,\cdot)$ is a probability measure and $K(\cdot,B)$ is $\Sigma_X$-measurable. The transition kernel $P(\cdot \mid s,a)$ and the policy distributions $\pi_t(\cdot \mid h_t)$ are both kernels.

* **Trajectory measure.** Given an initial distribution $\mu$ on state space $S$, a sequence of policy kernels $(\pi_t)_{t \geq 0}$ and a transition kernel $P$, the Ionescu-Tulcea theorem guarantees a unique probability measure $\mathbb{P}$ on the trajectory space $\mathcal{T} = (S \times A)^{\mathbb{N}}$.

### The Markov Decision Process Framework

The unifying mathematical foundation for reinforcement learning is the Markov Decision Process (MDP), formally defined as the 5-tuple:

$$\mathcal{M} = \langle S, A, P, R, \gamma \rangle$$

where:
- **States ($S$):** The set of all possible environment configurations (equipped with $\sigma$-algebra $\Sigma_S$)
- **Actions ($A$):** The set of all possible agent actions (equipped with $\sigma$-algebra $\Sigma_A$)
- **Transition kernel ($P(\cdot \mid s,a)$):** Probability distribution over next states given current state-action pair
- **Reward function ($R: S \times A \to \mathbb{R}$):** Expected immediate reward for taking action $a$ in state $s$
- **Discount factor ($\gamma \in [0,1)$):** Controls the trade-off between immediate and future rewards

The discount factor serves multiple purposes: it ensures finite returns in infinite-horizon problems, creates an effective horizon of approximately $\frac{1}{\varepsilon(1-\gamma)}$ steps, and provides implicit regularization favoring earlier rewards.

### Policies and the Agent-Environment Loop

A **policy** $\pi$ defines the agent's strategy for action selection. In its most general form, a policy is a sequence of conditional probability distributions $\{\pi_t\}_{t \geq 0}$:

$$\pi_t : \mathcal{H}_t \to M_1(A)$$

where $\mathcal{H}_t = (S \times A)^{t-1} \times S$ represents the history space at time $t$, and $M_1(A)$ denotes probability measures over actions.

The **history** at time $t$ is: $H_t = (S_0, A_0, S_1, \ldots, S_{t-1}, A_{t-1}, S_t)$

The agent-environment interaction forms a closed feedback loop:
1. Initial state $S_0 \sim \mu$ (initial state distribution)
2. Action selection $A_t \sim \pi_t(H_t)$
3. State transition $S_{t+1} \sim P(\cdot \mid S_t, A_t)$
4. Reward observation $R_t = R(S_t, A_t)$

A **stationary Markov policy** is a single kernel $\pi : S \times \Sigma_A \to [0,1]$ that depends only on the current state.

### Value Functions and Bellman Equations

The agent's goal is to maximize expected **discounted return**:

$$
J(\pi) = \mathbb E_{\mu}^{\pi} \left[ \sum_{t=0}^{\infty} \gamma^t R(S_t,A_t) \right]
$$

This leads to the definition of **value functions**:

$$V^\pi(s) := \mathbb{E}^\pi \left[\sum_{t=0}^{\infty}\gamma^{t}R_t \mid S_0=s\right]$$

$$Q^\pi(s,a) := \mathbb{E}^\pi \left[\sum_{t=0}^{\infty}\gamma^{t}R_t \mid S_0=s, A_0=a\right]$$

The **optimal value functions** are:

$$
V^\ast (s) = \sup_{\pi} V^\pi(s), \quad Q^\ast (s,a) = \sup_{\pi}Q^\pi(s,a)
$$

Define the policy-induced reward and transition functions:
$$R_\pi(s):=\int_A\pi(\mathrm{d}a\mid s)\,R(s,a), \quad P_\pi(\mathrm{d}s'\mid s):=\int_A\pi(\mathrm{d}a\mid s) P(\mathrm{d}s'\mid s,a)$$

The **Bellman equations** for policy evaluation are:

$$
V^\pi(s) = R_\pi(s) +\gamma\int_S P_\pi(\mathrm{d}s'\mid s)\,V^\pi(s')
$$

$$
Q^\pi(s,a) = R(s,a) +\gamma\int_S P(\mathrm{d}s'\mid s,a) \int_A \pi(\mathrm{d}a'\mid s')\,Q^\pi(s',a')
$$

The **Bellman optimality equations** are:

$$
V^\ast (s)=\max_{a} \lbrace R(s,a) +\gamma\int_S P(\mathrm{d}s'\mid s,a)\,V^*(s') \rbrace
$$

$$
Q^\ast (s,a)=R(s,a) +\gamma\int_S P(\mathrm{d}s'\mid s,a)\, \max_{a'}Q^*(s',a')
$$

Any deterministic policy greedy with respect to $Q^*$ is optimal.

### State Occupancy Measures

The **state occupancy measures** characterize the distribution of states visited under policy $\pi$:

$$d_{\mu,T}^\pi(s):=\frac{1}{T}\sum_{t=0}^{T-1} \mathbb{P}_\mu^\pi(S_t=s) \quad \text{(finite-horizon)}$$

$$d_\mu^\pi(s):=(1-\gamma)\sum_{t=0}^{\infty}\gamma^{t} \mathbb{P}_\mu^\pi(S_t=s) \quad \text{(discounted)}$$

The performance objective can be rewritten as:

$$
J(\pi)=\frac{1}{1-\gamma} \mathbb E_{s\sim d_\mu^\pi}[R_\pi(s)]
$$

### Alternative Performance Criteria

Beyond discounted return, other objectives include:
- **Finite-horizon return:** $G^H_t = \sum_{k=0}^{H-1} R_{t+k+1}$
- **Average reward:** $\lim_{T\to\infty}\frac{1}{T}\sum_{t=1}^{T}R_t$
- **Cumulative regret:** $\text{Regret}(T) = \sum_{t=0}^{T-1}(V^*(S_t) - R_t)$

### Theoretical Considerations

**Key theoretical results:**
- **Sufficiency of Markovian policies:** For MDPs, the current state is a sufficient statistic—optimal policies need only depend on current state, not full history
- **Deterministic optimality:** While policies can be stochastic, a deterministic optimal policy always exists
- **Existence of optimal policies:** Under regularity conditions (continuous bounded rewards, continuous transitions, compact action spaces), measurable optimal policies are guaranteed to exist

**Simplifying assumptions** commonly made to avoid measure-theoretic complexity:
1. **Finitude:** State and action spaces are finite
2. **Full observability:** Agent has direct access to current state

## The Three Domains of Reinforcement Learning

The field can be conceptualized as three overlapping domains addressing different aspects of sequential decision-making:

![Venn diagram](https://raw.githubusercontent.com/13ryanC/13ryanC.github.io/main/content/posts/RLtheory/images/venn_diagram.png)

### 1. Planning (Model-Based Control)

**Setting:** Known transition kernel $P(s' \mid s,a)$ and reward function $R(s,a)$

**Objective:** Determine optimal actions without further environment interaction

**Key approaches:**
- **Dynamic programming:** Solving Bellman optimality equation for closed-loop optimal policy
- **Trajectory optimization:** Open-loop optimization of action sequences $(a_{0:H-1})$ ignoring feedback during execution  
- **Online search:** Look-ahead methods (e.g., MCTS, AlphaZero) that replan at each step under computational constraints

**Computational complexity:** Even with perfect models, exact planning is computationally hard (P-complete for finite MDPs; PSPACE-hard for POMDPs).

### 2. Batch (Offline) RL

**Setting:** Static dataset $\mathcal{D} = \{(s_i, a_i, r_i, s'_i)\}$ from unknown behavior policy $\mu$, with no further interaction allowed

**Core challenge:** **Extrapolation error** - learned policy $\pi$ may query state-action pairs outside $\mu$'s support

**Solution approaches:**
- **Behavioral constraints:** Constraining $\pi$ near $\mu$ (behavior cloning, KL penalties)
- **Pessimistic value estimation:** Learning conservative value functions with uncertainty penalties (Conservative Q-Learning)
- **Importance sampling:** Density-ratio estimation for off-policy correction

### 3. Online RL (Interactive Learning)

**Protocol:** At each time-step $t = 0, 1, \ldots$:
1. Agent observes state $s_t$
2. Chooses action $a_t \sim \pi_t(\cdot \mid H_t)$ 
3. Environment yields reward $r_t$ and next state $s_{t+1}$
4. Agent updates parameters, producing $\pi_{t+1}$

**Central challenge:** **Exploration-exploitation trade-off** - gathering informative data while maximizing reward

**Exploration strategies:**
- **Random exploration:** ε-greedy, Boltzmann exploration
- **Optimism under uncertainty:** UCB, RLSVI
- **Posterior sampling:** Thompson sampling, PSRL

**Learning paradigms:**
- **On-policy:** Updates use current policy data (e.g., PPO)
- **Off-policy:** Reuses past trajectories with importance sampling corrections (e.g., Q-learning, DDPG)

## The Central Challenge: Learning Under Uncertainty

The fundamental RL problem: *How can an agent learn an optimal policy when transition dynamics $P$ and rewards $R$ are unknown?*

This necessitates **exploration**—sometimes sacrificing immediate reward to gather information for future exploitation. The theoretical limit is captured by the **identifiability barrier**:

> **Identifiability Barrier**
> 
> If two MDPs differ only in $P(\cdot \mid s,a)$ for some state-action pair $(s,a)$, any algorithm avoiding $(s,a)$ with probability 1 produces identical trajectories in both environments and cannot be simultaneously optimal.

Success is measured by how efficiently agents balance exploration and exploitation across diverse environments, guided only by self-generated trajectory data.

### Extensions and Alternative Frameworks

**Common MDP extensions:**

| Framework | Key Distinction | New Challenges |
|-----------|-----------------|----------------|
| **POMDP** | Partial observability via $O(o \mid s)$ | Belief-state explosion; intractable control |
| **Semi-MDP** | Actions with variable duration | Temporal abstraction; hierarchical planning |
| **Multi-agent MDP** | Multiple decision-makers | Coupled rewards; game-theoretic considerations |
| **CMDP** | Constrained optimization | Feasible-set identification; dual learning |
| **Multi-objective RL** | Vector-valued rewards | Preference elicitation; Pareto optimality |

Each framework preserves the core state → action → reward structure while addressing specific real-world complexities.

## References

* RL Theory. (2021, January 19). *Lecture 1 (2021-01-12)* [Video]. YouTube. [http://www.youtube.com/watch?v=0oJmSULoj3I](http://www.youtube.com/watch?v=0oJmSULoj3I)
* RL Theory. (2022, January 9). *Lecture 1 (2022-01-05)* [Video]. YouTube. [http://www.youtube.com/watch?v=rjwxqcVrVws](http://www.youtube.com/watch?v=rjwxqcVrVws)
* RL Theory. (2025, February 5). *The fundamental theorem* [Lecture notes]. [https://rltheory.github.io/w2021-lecture-notes/planning-in-mdps/lec2/](https://rltheory.github.io/w2021-lecture-notes/planning-in-mdps/lec2/)
* RL Theory. (2025, February 5). *Introductions* [Lecture notes]. [https://rltheory.github.io/lecture-notes/planning-in-mdps/lec1/](https://rltheory.github.io/lecture-notes/planning-in-mdps/lec1/)
* Jiang, N. (2024, September 27). *MDP preliminaries* [Lecture notes]. [https://nanjiang.cs.illinois.edu/files/cs542f22/note1.pdf](https://nanjiang.cs.illinois.edu/files/cs542f22/note1.pdf)
