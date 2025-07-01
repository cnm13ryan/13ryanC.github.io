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

## TL;DR

- Reinforcement Learning unifies **Planning** (known model), **Batch/Offline RL** (fixed data), and **Online RL** (live interaction) via the **Markov Decision Process**, a tuple $(\mathcal S,\mathcal A,P,r,\gamma)$ capturing states, actions, unknown transition dynamics, rewards, and a discount factor that regularises infinite horizons.

- Because $P$ and $r$ are unknown, the agent must **learn**: from its history it chooses a *policy* that maximises the **expected discounted return**—even though each policy induces a *distribution* of returns. 
 
- In finite, fully observable MDPs the current state suffices; an optimal policy can be deterministic.

- To avoid deep measure‑theoretic complications (infinite trajectories, continuous spaces), this first instalment restricts attention to finite, observable settings—clearing theoretical underbrush before tackling larger, risk‑sensitive or partially observable problems later in the series.

## The Landscape of Reinforcement Learning

The field can be conceptualized as a Venn diagram of three distinct, yet overlapping, domains:

![Venn Diagram](../images/venn_diagram.png)

### 1. Planning (Model-Based Control)

Given a known transition kernel $P(s' \mid s,a)$ and reward function $R(s,a)$, planning decides how to act **without further environment interaction**. Three main approaches exist:

* **Closed-loop optimal policy** $(\pi^* : S \rightarrow A)$ via dynamic programming solving the Bellman optimality equation:

  $$
  V^{\ast}(s) = \max_{a} \lbrace R(s,a) + \gamma\sum_{s'}P(s'|s,a)V^{\ast}(s') \rbrace
  $$

* **Open-loop trajectory optimization** of action sequences $(a_{0:H-1})$ ignoring feedback during execution

* **Online look-ahead search** (e.g. MCTS, AlphaZero) that replans at each step under limited compute

Even with perfect models, exact planning is computationally hard (P-complete for finite MDPs; PSPACE-hard for POMDPs). When models are *learned* and reused (e.g. Dyna, MuZero, PETS), this becomes **model-based reinforcement learning**.

### 2. Batch (Offline) RL

**Setting:** A static dataset $\mathcal{D} = \{(s_i, a_i, r_i, s'_i)\}$ from an unknown behavior policy $\mu$, with no further interaction allowed.

**Challenge:** The learned policy $\pi$ may query state-action pairs outside $\mu$'s support, causing *extrapolation error*. Solutions include:
- Constraining $\pi$ near $\mu$ (behavior cloning, KL penalties)
- Learning **pessimistic value functions** that subtract uncertainty bonuses (Conservative Q-Learning)
- Importance sampling with density-ratio estimation

### 3. Online RL (Interactive Learning)

**Protocol:** At each discrete time-step $t = 0, 1, \ldots$:
1. Agent observes state $x_t$ from the environment
2. Chooses action $a_t \sim \pi_t(\cdot \mid H_t)$ where policy $\pi_t$ may depend on history $H_t = (x_0, a_0, r_0, \ldots, x_t)$
3. Environment yields reward $r_t$ and next state $x_{t+1}$
4. Agent **updates** its parameters, producing $\pi_{t+1}$

**Objective:** Maximize expected discounted return $J(\pi) = \mathbb{E}[\sum_{t=0}^{\infty}\gamma^{t}r_t]$ or minimize **cumulative regret**:
$\text{Regret}(T) = \sum_{t=0}^{T-1}(V^*(x_t) - r_t)$

**Exploration-exploitation trade-off:** Actions must gather informative data while accruing reward via:
- ε-greedy and Boltzmann exploration
- **Optimism under uncertainty** (UCB, RLSVI) 
- **Posterior sampling** (Thompson sampling, PSRL)

**Learning paradigms:**
- *On-policy*: Updates use current policy $\pi_t$ data (e.g., PPO)
- *Off-policy*: Reuses past policy trajectories with importance sampling corrections (e.g., Q-learning, DDPG)

Online RL unifies *data collection* and *learning* in a single feedback loop—contrasting with planning (model known) and offline RL (data fixed).

---

### The Markov Decision Process Foundation

The unifying formalism is the MDP 5-tuple:
$$\mathcal{M} = \langle S, A, P, R, \gamma \rangle$$

where $S$ is the state space, $A$ the action space, $P(\cdot \mid s,a)$ the transition kernel, $R(s,a)$ expected immediate reward, and $\gamma \in [0,1]$ the discount factor.

A policy $\pi$ induces value functions:
$$V^\pi(s) = \mathbb{E}\left[\sum_{t=0}^{\infty}\gamma^{t}r_t \mid s_0=s, \pi\right], \quad V^*(s) = \sup_{\pi}V^\pi(s)$$

**Common Extensions:**
* **POMDP** – partial observability via observation kernel $O(o \mid s)$
* **Semi-MDP** – actions with variable duration
* **Multi-agent MDP** – multiple decision-makers with coupled rewards

### Performance Criteria

Beyond the discounted return objective detailed in online RL, alternative formulations include:

* **Finite-horizon reward:** $G^{(H)}_t = \sum_{k=0}^{H-1}R_{t+k+1}$
* **Average reward:** $\lim_{T\to\infty}\frac{1}{T}\sum_{t=1}^{T}R_t$

Each criterion leads to different algorithmic designs and theoretical guarantees.

## The Central Challenge: Learning Under Uncertainty

The fundamental RL problem: *How can an agent learn an optimal policy when transition dynamics $P$ and rewards $r$ are unknown?*

This necessitates **exploration**—sometimes sacrificing immediate reward to gather information for future exploitation. The challenge is captured by the **identifiability barrier**:

> **Identifiability Barrier**
> 
> If two MDPs differ only in $P(\cdot \mid s,a)$ for some state-action pair $(s,a)$, any algorithm avoiding $(s,a)$ with probability 1 produces identical trajectories in both environments and cannot be simultaneously optimal.

Success is measured by how efficiently agents accomplish both exploration and exploitation across diverse environments, guided only by self-generated trajectory data.

### Alternative Frameworks

| Framework | Key Distinction | New Challenges |
|-----------|-----------------|----------------|
| **POMDP** | Noisy state observations | Belief-state explosion; intractable control |
| **CMDP** | Constrained optimization | Feasible-set identification; dual learning |
| **Multi-objective RL** | Vector-valued rewards | Preference elicitation; Pareto optimality |

Each framework preserves the core state → action → reward structure while addressing specific real-world complexities.

---

### **The Core Framework: Markov Decision Processes (MDPs)**

A MDP is the tuple
$$
\mathcal{M} = \langle S, A, P, R, \gamma \rangle
$$
with 
* **States ($S$):** A set representing every possible configuration of the environment.
* **Actions ($A$):** A set of all possible actions the agent can take.
* **Transition Function ($P(\cdot|s, a)$):** A probability kernel that maps a state-action pair $(s,a)$ to a probability distribution over the next state $S$.
* **Reward Function ($r(s,a)$):** A scalar reward received for taking action $a$ in state $s$.
* **Objective function related:** discounting $\gamma$, or otherwise.

Rigorously, an MDP is a tuple $(\mathcal{S}, \mathcal{A}, P, r, \gamma)$, where $\mathcal{S}$ and $\mathcal{A}$ are measurable spaces (equipped with respective $\sigma$-algebras, $\Sigma_S$ and $\Sigma_A$). 

The reward function $r: S \times A \to \mathbb{R}$ and the transition kernel $P$ are assumed to be measurable functions.

### **The Agent's Objective: Maximizing a Stochastic Return**

An agent's interaction with an MDP produces a **trajectory**, $\tau$, which is a sequence of state-action pairs: $\tau = (S_0, A_0, S_1, A_1, \dots)$. 

The agent's goal is to maximize its **return**, $G(\tau)$, which is the cumulative sum of rewards obtained along this trajectory, where $G(\tau) = r_{A_0}(S_0) + r_{A_1}(S_1) + \dots$

A critical aspect of modeling is the introduction of a **discount factor, $\gamma \in [0, 1)$**, which ensures that for infinite-horizon problems, the return is a finite, well-defined value. 

This also creates an "effective horizon" (formally, $\dfrac{1}{\varepsilon(1-\gamma)}$), prioritizing nearer rewards over distant ones. It can also be interpreted as a constant probability, $1-\gamma$, of the process terminating at any given step, and its presence is crucial for ensuring the convergence of many RL algorithms.

Note that the epsilon $\varepsilon$ is useful if you are measuring an infinite sum up to $\varepsilon$ accuracy, then you can truncate the sum after roughly this many terms, such that the gap between the truncated sum and the original un-truncated sum (actual sum, may not be computed), is within the error range of $\varepsilon$.

One can frame discounting as a form of **"implicit regularization,"** which biases the agent toward solutions that accumulate rewards sooner.
 
The problem with the definition is that the return of the trajectory $G(\tau)$ does not tell us whether the transitons are stochastic. But since we know that the transitons are stochastic, the return $G(\tau)$ in general has a distribution over the trajectories that is induced by how one is interacting with or controlling the MDP.

Another insight, due to the stochastic nature of the transitions, we know that the return from any policy is not a single number but a **probability distribution**. This fact complicates the notion of "maximization" and raises a profound question:

* *What does it even mean to maximize a quantity that is itself a distribution?*

The standard resolution in RL is to maximize the **expected value** of this distribution.

Formally, the objective is to find a policy $\pi^*$ that maximizes the expected discounted return for a given starting state distribution $\mu$: 

$$
\pi^\ast = \arg\max_{\pi}  E_{\mu}^{\pi} [ \sum_{t=0}^{\infty} \gamma^{t}\, r(S_t,A_t)]
$$

$E := \mathbb{E}$ since rendering sucks on long equations.

### **The Agent's Strategy: Policies, History, and Observability**

The agent's strategy is its **policy ($\pi$)**, which specifies how it chooses actions. 

In its most general form, a policy can be conditioned on the entire **history ($H_t$)** of interaction up to the present moment. 

A history is the sequence of past states and the actions taken in them:

$H_t = (S_0, A_0, S_1, \dots, S_{t-1}, A_{t-1}, S_t)$. 

Formally, a policy is a sequence of conditional probability distributions $\{\pi_t\}_{t \geq 0}$, where each $\pi_t$ is a mapping from the space of histories $\mathcal{H}_t$ to a probability distribution over actions $M_1(A)$:

$$\pi_t : \mathcal{H}_t \to M_1(A)$$

where $\mathcal{H}_t = (S \times A)^{t-1} \times S$

This formalism highlights the fundamental **interconnection of the MDP**, which forms a closed feedback loop. The process unfolds sequentially:
1.  An initial state $S_0$ is drawn from an initial state distribution $\mu \in M_1(S)$.
2.  The policy $\pi$ selects an action $A_0 \sim \pi_0(H_0)$, where $H_0 = S_0$.
3.  The transition kernel $P$ yields a subsequent state $S_1 \sim P(\cdot|S_0, A_0)$.

This cycle—where the policy's output (an action) becomes an input for the environment's transition kernel, whose output (a new state) then becomes an input for the policy—is the core feedback loop. The interconnection of $(\mu, \pi, P)$ is what generates the probability measure over the space of all trajectories.
 
Fix a policy $\pi$, adn an initial state distribution $\mu \in M_1(S)$.

Fix a MDP transition state structure $(P_a(S))_{s,a}$. 

This puts a distribution over a space of trajectories, so the trajectory space is $T = (S \times A)^{\mathbb{N}}$.

We denote the distribution $\mathbb{P}_{\mu}^{\pi} (\tau)$


$= \mu(s_0) \pi(s_0, a_0) (P_{a_0} (s_0, s_1)) $

$= \lbrace (s_t, a_t)_{t \leq 0} | s_t \in S, a_t \in A \rbrace $.


This formal definition surfaces another layer of nuanced questions about the nature of an optimal strategy:
* **Is the entire history necessary for optimal decisions, or is a more compact representation of the past sufficient, such as the current state?**
  * For MDPs, we know of a core result (the "fundamental theorem of MDPs") is that the current state is a sufficient statistic of the past. An optimal policy can be found that only depends on the current state.
* **How important is randomization in a policy? Could an optimal policy be purely deterministic?**
  * While policies *can* be stochastic, we know that an optimal policy can always be found that is purely deterministic. Randomization is not a requirement for optimality in this context.
* **Does a policy's dependence on the state imply that the state is fully observable? And is effective learning even possible without this assumption?**
* **If the state space is vast, how should an agent compress this information?**
* **Does a single policy exist that is uniformly optimal for all starting states?**
  * Yes. The objective is to find a single policy $\pi^{\ast}$ that achieves the maximum possible expected return (the "optimal value function" $V^*(s)$) from *any* starting state $s$.
* **What are the drawbacks of maximizing only the expected return? What if we wanted to minimize risk or variance?** 
  * This is the standard approach, and our current scope is confined to it. Broader RL research explores risk-sensitive or variance-aware objectives for more robust decision-making.

### **Theoretical Underpinnings and Foundational Assumptions**

Answering these questions rigorously, especially when the trajectory is infinite or spaces are continuous, reveals why the simplifying assumptions are made. Without them, one cannot escape measure theory and topology.

* **Measurability and Existence of Measures:** 
  * For an infinite trajectory, the space of all possible histories $(S \times A)^\infty$ must be endowed with a product $\sigma$-algebra. 
  * For the expectation of the return to be well-defined, the policy $\pi_t$ must be a measurable function from the space of histories to the space of actions. 
  * The Ionescu-Tulcea extension theorem guarantees that if the policy is constructed from such measurable functions, the closed loop induces a unique probability measure on the trajectory space.
  * Can we avoid axiom of choice or use a weaken version of it?

* **Existence of an Optimal Policy:** 
  * The goal is to find a policy that achieves the supremum of the expected return. However, does a measurable optimal policy even exist? 
  * To guarantee existence, especially in continuous spaces, further regularity conditions are needed. 
  * For instance, if the reward function $r$ is continuous and bounded, the transition kernel $P$ is continuous (in total variation), and the action space $A$ is compact, then a measurable (and often deterministic and stationary) optimal policy is guaranteed to exist. This is where concepts from topology become critical.
 
To "free up mental space while avoid messing around with measure theory deeply" and build a solid foundation, the theoretical analysis begins with two powerful simplifying assumptions:
1.  **Finitude:** The state and action spaces, $S$ and $A$, are assumed to be finite.
2.  **Observability:** The agent is assumed to have direct access to the current state of the environment, which is okay for planning problems, given we know the model of the environment with which we don't care much about rewards, focusing on finding the best policy for choosing good actions.We will come to rewards later when we talk about evaluating policies.

The observability assumption also poses a question on how come these systems can learn and gather rewards? Do they actually know the environment states? Is it possible to have a system that does not know the state of the system.

In particular, we would want to know the upper bound of rewards and the limits of MDP formulation.

These assumptions, while restrictive, neatly sidestep these measure-theoretic and topological complexities, making the problem tractable. They allow for the establishment of clear theoretical bounds, creating a solid base from which the theory can be extended.

## References

* RL Theory. (2021, January 19). *Lecture 1 (2021-01-12)* [Video]. YouTube. [http://www.youtube.com/watch?v=0oJmSULoj3I](http://www.youtube.com/watch?v=0oJmSULoj3I)
* RL Theory. (2022, January 9). *Lecture 1 (2022-01-05)* [Video]. YouTube. [http://www.youtube.com/watch?v=rjwxqcVrVws](http://www.youtube.com/watch?v=rjwxqcVrVws)
