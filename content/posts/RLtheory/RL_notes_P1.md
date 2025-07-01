---
date: "2025-06-27"
title: "(Part 1) Personal Notes on the Foundations of Reinforcement Learning"
summary: "Aim to provide more insight on RL foundations for beginners"
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
**MDP**: States, Actions, Transition probabilities, Rewards, discount factor. Agent follows a **policy** to maximize expected future rewards. Key insight: current state contains all info needed for optimal decisions.

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

### The Markov Decision Process Framework

The unifying mathematical foundation for reinforcement learning is the Markov Decision Process (MDP), formally defined as the 5-tuple:

$$\mathcal{M} = \langle S, A, P, R, \gamma \rangle$$

where:
- **States ($S$):** The set of all possible environment configurations (equipped with $\sigma$-algebra $\Sigma_S$ for measurability)
- **Actions ($A$):** The set of all possible agent actions (equipped with $\sigma$-algebra $\Sigma_A$)
- **Transition Kernel ($P(\cdot \mid s,a)$):** Probability distribution over next states given current state-action pair
- **Reward Function ($R: S \times A \to \mathbb{R}$):** Expected immediate reward for taking action $a$ in state $s$
- **Discount Factor ($\gamma \in [0,1)$):** Controls the trade-off between immediate and future rewards

The discount factor serves multiple purposes: it ensures finite returns in infinite-horizon problems, creates an effective horizon of approximately $\frac{1}{\varepsilon(1-\gamma)}$ steps, and can be interpreted as implicit regularization favoring earlier rewards or as a constant termination probability $(1-\gamma)$.

### Policies and the Agent-Environment Loop

A **policy** $\pi$ defines the agent's strategy for action selection. In its most general form, a policy is a sequence of conditional probability distributions $\{\pi_t\}_{t \geq 0}$:

$$\pi_t : \mathcal{H}_t \to M_1(A)$$

where $\mathcal{H}_t = (S \times A)^{t-1} \times S$ represents the history space, and $M_1(A)$ denotes probability measures over actions.

The **history** at time $t$ is: $H_t = (S_0, A_0, S_1, \ldots, S_{t-1}, A_{t-1}, S_t)$

The agent-environment interaction forms a closed feedback loop:
1. Initial state $S_0 \sim \mu$ (initial state distribution)
2. Action selection $A_t \sim \pi_t(H_t)$
3. State transition $S_{t+1} \sim P(\cdot \mid S_t, A_t)$
4. Reward observation $R_t = R(S_t, A_t)$

This interconnection of $(\mu, \pi, P)$ induces a probability measure $\mathbb{P}_{\mu}^{\pi}$ over the trajectory space $\mathcal{T} = (S \times A)^{\mathbb{N}}$.

### Objectives and Value Functions

The agent's goal is to maximize expected **discounted return**:

$$
J(\pi) = \mathbb{E}_{\mu}^{\pi} \left[ \sum_t \gamma^t R(S_t,A_t) \right]
$$

This leads to the definition of **value functions**:
- **State value function:** $V^\pi(s) = \mathbb{E}^\pi\left[\sum_{t=0}^{\infty}\gamma^{t}R_t \mid S_0=s\right]$

- **Optimal value function:** $V^*(s) = \sup_{\pi}V^\pi(s)$

- **Action-value function:** $Q^\pi(s,a) = \mathbb{E}^\pi\left[\sum_{t=0}^{\infty}\gamma^{t}R_t \mid S_0=s, A_0=a\right]$

The **Bellman optimality equation** characterizes the optimal value function:

$$
V^{\ast}(s) = \max_{a} \lbrace R(s,a) + \gamma\sum_{s'}P(s'|s,a)V^{\ast}(s') \rbrace
$$

**Alternative performance criteria** include:
- **Finite-horizon return:** $G^H_t = \sum_{k=0}^{H-1} R_{t+k+1}$

- **Average reward:** $\lim_{T\to\infty}\frac{1}{T}\sum_{t=1}^{T}R_t$

- **Cumulative regret:** $\text{Regret}(T) = \sum_{t=0}^{T-1}(V^*(S_t) - R_t)$

### Theoretical Considerations

**Key theoretical results:**
- **Sufficiency of Markovian policies:** For MDPs, the current state is a sufficient statistic—optimal policies need only depend on current state, not full history
- **Deterministic optimality:** While policies can be stochastic, a deterministic optimal policy always exists
- **Existence of optimal policies:** Under regularity conditions (continuous bounded rewards, continuous transitions, compact action spaces), measurable optimal policies are guaranteed to exist

**Measure-theoretic foundations:** For rigorous treatment of infinite trajectories and continuous spaces, one must ensure:
- Measurability of policy functions for well-defined expectations
- Application of Ionescu-Tulcea extension theorem for trajectory measures
- Topological conditions for optimal policy existence

**Simplifying assumptions** commonly made to avoid measure-theoretic complexity:
1. **Finitude:** State and action spaces are finite
2. **Full observability:** Agent has direct access to current state

## The Three Domains of Reinforcement Learning

The field can be conceptualized as three overlapping domains addressing different aspects of sequential decision-making:

![Venn Diagram](../images/venn_diagram.png)

### 1. Planning (Model-Based Control)

**Setting:** Known transition kernel $P(s' \mid s,a)$ and reward function $R(s,a)$

**Objective:** Determine optimal actions without further environment interaction

**Key approaches:**
- **Closed-loop optimal policy** via dynamic programming solving Bellman optimality equation
- **Open-loop trajectory optimization** of action sequences $(a_{0:H-1})$ ignoring feedback during execution  
- **Online look-ahead search** (e.g., MCTS, AlphaZero) that replans at each step under computational constraints

**Computational complexity:** Even with perfect models, exact planning is computationally hard (P-complete for finite MDPs; PSPACE-hard for POMDPs).

**Model-based RL connection:** When models are learned and reused (e.g., Dyna, MuZero, PETS), planning becomes model-based reinforcement learning.

### 2. Batch (Offline) RL

**Setting:** Static dataset $\mathcal{D} = \{(s_i, a_i, r_i, s'_i)\}$ from unknown behavior policy $\mu$, with no further interaction allowed

**Core challenge:** **Extrapolation error** - learned policy $\pi$ may query state-action pairs outside $\mu$'s support

**Solution approaches:**
- **Behavioral constraints:** Constraining $\pi$ near $\mu$ (behavior cloning, KL penalties)
- **Pessimistic value estimation:** Learning conservative value functions with uncertainty penalties (Conservative Q-Learning)
- **Importance sampling:** Density-ratio estimation for off-policy correction

### 3. Online RL (Interactive Learning)

**Protocol:** At each time-step $t = 0, 1, \ldots$:
1. Agent observes state $x_t$
2. Chooses action $a_t \sim \pi_t(\cdot \mid H_t)$ 
3. Environment yields reward $r_t$ and next state $x_{t+1}$
4. Agent updates parameters, producing $\pi_{t+1}$

**Central challenge:** **Exploration-exploitation trade-off** - gathering informative data while maximizing reward

**Exploration strategies:**
- **Random exploration:** ε-greedy, Boltzmann exploration
- **Optimism under uncertainty:** UCB, RLSVI
- **Posterior sampling:** Thompson sampling, PSRL

**Learning paradigms:**
- **On-policy:** Updates use current policy data (e.g., PPO)
- **Off-policy:** Reuses past trajectories with importance sampling corrections (e.g., Q-learning, DDPG)

**Distinguishing feature:** Online RL uniquely unifies data collection and learning in a single feedback loop, contrasting with planning (model known) and offline RL (data fixed).

## The Central Challenge: Learning Under Uncertainty

The fundamental RL problem: *How can an agent learn an optimal policy when transition dynamics $P$ and rewards $R$ are unknown?*

This necessitates **exploration**—sometimes sacrificing immediate reward to gather information for future exploitation. The theoretical limit is captured by the **identifiability barrier**:

> **Identifiability Barrier**
> 
> If two MDPs differ only in $P(\cdot \mid s,a)$ for some state-action pair $(s,a)$, any algorithm avoiding $(s,a)$ with probability 1 produces identical trajectories in both environments and cannot be simultaneously optimal.

Success is measured by how efficiently agents balance exploration and exploitation across diverse environments, guided only by self-generated trajectory data.

## Extensions and Alternative Frameworks

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
