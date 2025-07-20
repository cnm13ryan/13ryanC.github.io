---
date: "2025-07-20"
title: "A1. Sequential Decision Processes"
summary: "A1. Sequential Decision Processes"
lastmod: "2025-07-20"
category: "Notes"
series: ["RL Topics", "MARL"]
author: "Bryan Chan"
hero: /assets/images/hero3.png
image: /assets/images/card3.png
---

# 1 Sequential Decision Processes

## **1. An Overview**

A **sequential decision process** refers to a problem in which an agent must choose actions over multiple time steps, knowing that each choice influences both the immediate reward and the future situation it will face.

In other words:
> How do we formally describe and analyze situations where decisions are made in a chain, with each step shaping what comes next?

A sequential decision process is fundamentally defined by an agent making decisions over multiple time steps within an environment to achieve a specified goal. This framework forms the core of Reinforcement Learning (RL) problems. This process involves a continuous interaction loop: at each time step, the agent receives an observation from the environment and, based on this observation, chooses an action. In response to the chosen action, the environment may change its state and send a scalar reward signal back to theagent. This cycle continues until a stopping condition is met or indefinitely.
 

The core challenge lies in balancing short-term gains with long-term consequences. A greedy, single-step optimization approach can look good locally but lead to catastrophic long-term outcomes. For example, over-exploiting a resource might yield high initial returns but result in its depletion. Nearly all real-world control tasks, from managing inventory to robotic navigation, are inherently sequential.

The core difficulty is often the **curse of dimensionality**, where the state and action spaces explode in size with the problem's details and time horizon.

---

### Formalizing the Objective: The Return

The agent's goal is to maximize a cumulative function of the rewards it receives. This cumulative reward is known as the **return**. For a sequence of rewards $R_{t+1}, R_{t+2}, \dots$ received after time step $t$, the discounted return $G_t$ is defined as:

$$
G_t \doteq R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \dots = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1}
$$

Here, $\gamma$ is the **discount factor** ($0 \le \gamma \le 1$). It determines the present value of future rewards. A discount factor of $\gamma=0$ results in a myopic agent that only cares about the immediate reward, while a value closer to 1 means the agent is more farsighted.

---

## **2. The Core Components**

Every sequential decision process, regardless of the specific domain, is defined by five essential components. The choice of model drives everything else; if one ingredient is missing, the framework collapses into a descriptive simulation (no goal) or a simple stochastic process (no actions). Because actions today affect states tomorrow, design errors like omitting a key state variable can create compounding mistakes.
 

| # | Characteristic | Essence | Canonical MDP Notation |
|---|---|---|---|
| 1 | **Decision Epochs (Time)** | A sequence of points in time, $t = 0, 1, \dots$, at which the agent can take actions. | The subscript $t$ in expressions like $s_t$ and $a_t$. |
| 2 | **State Space $\mathcal{S}$** | A set of possible situations. A state, $s_t \in \mathcal{S}$, must summarize all past information relevant for future outcomes (the **Markov Property**). | $s_t \in \mathcal{S}$ |
| 3 | **Action Space $\mathcal{A}$** | A set of feasible actions, $a_t \in \mathcal{A}$, available to the agent in a given state. | $a_t \in \mathcal{A}(s_t)$ |
| 4 | **Transition Dynamics $P(\cdot\mid s,a)$** | A rule, either deterministic or probabilistic, that governs how the environment moves to the next state, $s_{t+1}$, after the agent takes action $a_t$ in state $s_t$. | $P(s_{t+1}=s'\mid s_t=s, a_t=a)$ |
| 5 | **Objective (Evaluation Criterion)** | A function that specifies why some sequences of actions are better than others. This is typically a cumulative reward or cost. | Reward function $R(s,a,s')$ and the expected return $\mathbb{E}\left[\sum_{t}\gamma^t R_{t+1}\right]$. |

A clean formulation of a sequential decision process requires:
1.  **State Sufficiency (Markov Property):** The current state summarizes all relevant history.
2.  **Stochastic Transition Model:** A known or simulatable rule for how the environment changes.
3.  **Well-defined Objective:** A clear metric for success, such as cumulative reward, cost, or risk.

Success is often measured by the ability to compute a policy with strong performance guarantees or one that can scale to high-dimensional problems.

---

## **3. The Role of the Goal**

A model that only describes how the world changes (`states + actions + transitions`) is insufficient for making choices. To decide on an action, an agent needs a criterion to judge what "good" performance looks like. Therefore, the specification of a goal is what transforms a simple transitional model into a decision process.

This goal is formally captured by an **objective function**. The most common objective is to maximize the expected cumulative reward. However, defining this function correctly is a major challenge in RL. Small misspecifications can lead to **reward hacking** due to Goodhart's Law, where the agent achieves the literal goal in unintended and undesirable ways. Real-world systems often need to balance efficiency, safety, fairness, and cost, and turning these into a single scalar reward remains an active area of research.

In many human-in-the-loop scenarios, goals are specified implicitly through demonstrations or preference rankings, which motivates frameworks like Inverse RL or Reinforcement Learning from Human Feedback (RLHF).

---

### Formalizing the Policy and Value Functions

To formalize how an agent behaves and evaluates its situation, we introduce three key concepts:

1.  **Policy ($\pi$):** A policy is the agent's strategy or "brain." It's a mapping from states to a probability distribution over actions.

    $$
    \pi(a|s) \doteq P(A_t=a \mid S_t=s)
    $$

    This function tells us the probability of taking action $a$ when in state $s$.

2.  **State-Value Function ($V^\pi(s)$):** This function measures the "goodness" of a state $s$ when following a specific policy $\pi$. It is the expected return starting from state $s$ and then following policy $\pi$ thereafter.

    $$
    V^\pi(s) \doteq \mathbb E_\pi [G_t \mid S_t=s] = \mathbb E_\pi \left[ \sum_{k=0}^{\infty} \gamma^k R_{t+k+1} \mid S_t=s \right]
    $$

3.  **Action-Value Function ($Q^\pi(s,a)$):** This function measures the "goodness" of taking a specific action $a$ in a state $s$, and then following policy $\pi$. It is the expected return after taking action $a$ in state $s$ and subsequently following policy $\pi$.

    $$
    Q^\pi (s,a) \doteq \mathbb E_\pi [G_t \mid S_t=s, A_t=a] = \mathbb E_\pi \left[ \sum_{k=0}^{\infty} \gamma^k R_{t+k+1} \mid S_t=s, A_t=a \right]
    $$

    These value functions are what RL algorithms aim to learn, as they allow the agent to compare actions and find the optimal way to behave.

---

Different frameworks formalize the goal in various ways:

| Framework | Typical Objective ("Goal") |
|---|---|
| **Markov Decision Process (MDP)** | Maximize expected discounted cumulative reward: $\mathbb{E}\left[\sum_{t=0}^\infty \gamma^t r_{t+1}\right]$ |
| **Stochastic Shortest Path** | Minimize expected cost or steps to reach a designated goal state. |
| **Constrained / Risk-Sensitive MDP** | Maximize reward subject to constraints on cost, risk, or safety. |
| **Partially-Observable MDP (POMDP)** | Same as MDP, but the objective is defined over belief states (probability distributions over hidden states). |
| **Preference-Based RL** | Learn a policy that satisfies human preference rankings, where the goal is inferred from feedback rather than a numeric reward. |

Approaches to specifying the objective include:

| Approach | Essence | Strengths | Pitfalls |
|---|---|---|---|
| **Hand-Designed Reward/Cost** | Manually engineer a numeric function $R(s,a)$. | Simple, analytically convenient. | Prone to misspecification and brittleness. |
| **Reward Shaping** | Add intermediate reward terms to guide learning. | Can speed up convergence. | May alter the optimal policy if not designed carefully. |
| **Inverse RL / Preference Learning** | Infer a reward function from expert demonstrations or comparisons. | Better aligns with complex human intent. | Sample-intensive and can be ambiguous. |
| **Multi-Objective Optimization** | Maintain a vector of rewards and find Pareto-optimal policies. | Transparently captures trade-offs. | Computationally harder; requires a final selection mechanism. |

A promising synthesis is to start with a coarse, hand-crafted objective but refine it using human-in-the-loop preference feedback, giving the agent an initial direction while allowing for alignment corrections during deployment.

---

## **4. The Markov Decision Process (MDP)**

A **Sequential Decision Process (SDP)** is the broad conceptual class of problems involving actions over time. A **Markov Decision Process (MDP)** is the most common and powerful mathematical framework used to model a specific subset of these problems.

**Every MDP is a sequential decision process, but not every sequential process is an MDP.**

The crucial distinction is the **Markov Property**: in an MDP, the current state $s_t$ is assumed to contain all information necessary to predict the future. The transition probabilities and rewards depend *only* on the current state and action, not on the entire history of previous states and actions.

**General Sequential Decision Process (SDP):**
* **History:** $H_t=(s_0,a_0,s_1,a_1,\dots ,s_t)$
* **Decision Rule:** An action $a_t$ can depend on the full history: $a_t\sim\delta_t(H_t)$
* **Transition Kernel:** The next state can depend on the full history: $P(s_{t+1}\mid H_t,a_t)$

**Markov Decision Process (MDP):**
An MDP is a specialization of an SDP that adheres to the Markov property. It is formally defined by the 5-tuple $\langle\mathcal{S}, \mathcal{A}, P, R, \gamma\rangle$.
* **Markov Property:** The future is conditionally independent of the past, given the present state and action.
    * **Transition:** $P(s_{t+1}\mid H_t,a_t) = P(s_{t+1}\mid s_t,a_t)$
    * **Reward:** The reward function is also Markovian, $r(s_t, a_t)$.
    * **Decision Rule:** The optimal action depends only on the current state: $a_t\sim\delta_t(s_t)$

**The Bellman Equations: The Heart of the MDP** 

The power of the MDP framework comes from the **Bellman equations**, which provide a recursive decomposition of the value functions. They express the value of a state in terms of the values of its successor states.

* **Bellman Expectation Equation:** This describes the value function for a *given* policy $\pi$.
    * For the state-value function $V^\pi$:
        $$
        V^\pi(s) = \sum_{a} \pi(a \mid s) \sum_{s', r} p(s', r \mid s, a) [r + \gamma V^\pi(s')]
        $$
    * For the action-value function $Q^\pi$:
        $$
        Q^\pi(s,a) = \sum_{s', r} p(s', r \mid s, a) [r + \gamma \sum_{a'} \pi(a' \mid s') Q^\pi(s', a')]
        $$

* **Bellman Optimality Equation:** This describes the value function for the *optimal* policy $\pi^\ast$. The optimal policy selects the action that maximizes the expected return.
    * For the optimal state-value function $V^\ast$:
        $$
        V^\ast (s) = \max_{a} \sum_{s', r} p(s', r \mid  s, a) [r + \gamma V^\ast (s')]
        $$
    * For the optimal action-value function $Q^\ast$:
        $$
        Q^\ast (s,a) = \sum_{s', r} p(s', r \mid s, a) [r + \gamma \max_{a'} Q^\ast (s', a')]
        $$
These equations form the basis for many RL algorithms, such as Value Iteration and Policy Iteration, which are methods for finding the optimal policy $\pi^*$.

*Mathematically*, any SDP can be converted into an MDP by redefining the "state" as the complete history $H_t$. However, the state space explodes exponentially, making this impractical. The power of the MDP formulation comes from situations where a low-dimensional sufficient state naturally exists.
 

### Key Differences between SDP and MDP

| Aspect | SDP (General) | MDP (Special Case) |
|---|---|---|
| **Transition Law** | May depend on the **full history**: $P(s_{t+1}\mid H_t,a_t)$. | Depends **only on current state & action**: $P(s_{t+1}\mid s_t,a_t)$ (Markov) |
| **Decision Rule** | Can be history-dependent: $\delta_t(H_t)$. | Can be state-dependent: $\delta_t(s_t)$. |
| **Sufficient State** | A concise state that summarizes history may not exist (often needs to be augmented by history) | A sufficient state $s_t$ exists by assumption. |
| **Solution Methods** | Generally harder; may require belief-state planning or policy search (i.e., belief-state MDPs, information state). | Amenable to dynamic programming that exploits the Markov structure. |
| **Observability** | Can be fully or partially observable. | Assumes full observability (the agent knows the true state $s_t$) (i.e., classical MDPs); POMDPs extend this. |

---

## **5. When MDP Assumptions Are Violated**

The elegance and efficiency of MDPs come from a set of strong assumptions. When these are violated, standard algorithms can fail, leading to poor performance or unsafe behavior.

| Assumption | Canonical Statement |
|---|---|
| **A1 Markov state sufficiency** | $P(s_{t+1},r_{t+1}\mid s_t,a_t,\text{history}) = P(s_{t+1},r_{t+1}\mid s_t,a_t)$ |
| **A2 Full observability** | The agent knows the true state $s_t$ before acting. |
| **A3 Stationarity** | The transition kernel $P$ and reward function $R$ are time-invariant. |
| **A4 Single-agent, passive world** | The environment contains no other strategic agents. |
| **A5 Markovian, additive objective** | The return is a discounted sum of per-step rewards depending only on $(s_t,a_t,s_{t+1})$. |

If any of these fail, the problem is no longer a vanilla MDP.


| Assumption Broken | Real-World Manifestation | Formal Consequence | Typical Fix |
|---|---|---|---|
| **A1 Markov Property** | Hidden system variables (e.g., battery temperature), delayed effects. | The next-state distribution depends on the unobserved past; Bellman equations become invalid. | Augment the state with memory; use a **Non-Markov Decision Process** or a **POMDP**. |
| **A2 Full Observability** | Noisy sensors, occluded objects, latent medical conditions. | The agent sees an observation $o_t \neq s_t$ and cannot condition its policy on the true state. | Use a **Partially Observable MDP (POMDP)** with belief-state planning. |
| **A3 Stationarity** | Changing traffic patterns, economic seasonality, wear and tear on machinery. | The transition model $P$ and reward function $R$ vary over time; a fixed policy becomes suboptimal. | Use **Non-stationary RL**, meta-learning, or model the changes with a **Hidden-Mode MDP**. |
| **A4 Single-Agent** | Financial markets, multi-robot teams, competitive games. | The environment's evolution depends on the joint actions of multiple agents; it becomes a **stochastic game**. | Use **Multi-Agent RL (MARL)** with game-theoretic solution concepts (e.g., Nash Equilibrium). |
| **A5 Additive Markov Reward** | Goals with complex temporal logic (e.g., "patrol locations A and B, then return to base"). | The reward depends on the entire history, not just the current step. | Use **Non-Markovian Reward Machines** or temporal-logic-based RL. |


### The POMDP Belief State

In a POMDP, the agent doesn't know the true state $s$. Instead, it maintains a **belief state** $b(s)$, which is a probability distribution over all possible states, given the history of actions and observations.
$$
b_t(s) \doteq P(S_t=s \mid o_t, a_{t-1}, o_{t-1}, \dots, a_0, o_0)
$$
After taking action $a_t$ and receiving observation $o_{t+1}$, the agent updates its belief using Bayes' rule. The new belief $b_{t+1}(s')$ is proportional to the probability of seeing that observation from state $s'$, summed over all previous states $s$ that could have led there:
$$
b_{t+1}(s') \propto \sum_{s \in \mathcal{S}} P(o_{t+1}|s', a_t) P(s'|s, a_t) b_t(s)
$$
Planning then happens in the continuous space of beliefs, which is much more complex than the original state space.

### Practitioner's Fault Map

* **Partial Observability & Sensor Noise:** The agent sees an observation $o_t = g(s_t, \epsilon_t)$, not the true state $s_t$. The Markov property fails on observations. Memory-free policies lose optimality. The repair is to formulate a **POMDP** and plan in the space of beliefs.
* **Non-stationary Dynamics:** The rules $P_t, R_t$ change over time. Stationary solutions break. The repair is to augment the state with a time variable, use **Hidden-Mode MDPs**, or apply robust planning.
* **Variable Action Durations:** Actions take a random amount of time to complete. Discrete-time Bellman equations mis-count rewards. The repair is to use **Semi-Markov Decision Processes (SMDPs)** or continuous-time control models.
* **High-Order Dynamics:** The future depends not just on $s_t$ but on a window of past states, $s_{t-k:t}$. Bellman optimality fails unless the state is augmented, but this causes the state space to explode. The repair is state augmentation or using recurrent neural networks (LSTMs) in the policy.
* **Strategic Interaction:** The transition law depends on the actions of other agents, $P(s_{t+1} \mid s_t, a_t, \mathbf{a}^o_t)$. From a single agent's view, the environment is non-stationary. The repair is to model the system as a **Markov Game** or **Dec-POMDP** and use game-theoretic solutions.
 

---

## **6. The Agent-Environment Interaction**

### **6.1 How the State Evolves**
The environment's state changes in response to the agents' actions. The complexity of this evolution depends on the model:
* **Normal-Form Games:** These model a single, static interaction. There is no evolving state; agents act once, receive a reward, and the game ends.
* **Repeated Games:** The same normal-form game is played multiple times. The environment itself does not have a state, but agents' policies can depend on the history of past actions.
* **Stochastic Games (Markov Games):** These introduce an explicit environment state that transitions probabilistically based on the **joint action** of all agents. Agents fully observe the state at each step.
* **Partially Observable Stochastic Games (POSGs):** This is the most general model. The state evolves based on joint actions, but agents receive only partial, noisy **observations** of the true state.

### **6.2 Learning via Repeated Interaction**
Agents learn optimal strategies not from a pre-programmed set of rules, but through a process of "trial and error." This involves a continuous interaction loop:
1.  The agent **observes** the current state (or observation).
2.  It selects an **action** based on its current policy.
3.  The environment **transitions** to a new state and provides a **reward** signal.

This cycle repeats over many "episodes." A critical challenge in this process is the **exploration-exploitation trade-off**: the agent must balance **exploiting** known good actions to maximize immediate reward with **exploring** new actions to discover potentially better long-term strategies.

This process is distinct from other machine learning paradigms. It is not supervised learning because the reward signal indicates the *quality* of an outcome, not the "correct" action. It is not unsupervised learning because the reward guides the agent toward a specific goal.

In multi-agent settings, this becomes more complex as each agent's learning updates the environment for all other agents, creating a non-stationary learning problem.


### **6.3 When the Interaction Ends**
An "episode" of interaction can conclude in several ways:
* **Terminal State:** The process reaches a designated end state (e.g., winning a game, crashing a vehicle).
* **Fixed Horizon:** The interaction stops after a predetermined number of time steps, $T$.
* **Infinite Horizon with Discounting:** For processes that could run forever, a discount factor $\gamma < 1$ is used. This ensures that the cumulative reward remains finite and gives more weight to immediate rewards. It can be interpreted as a constant probability $(1-\gamma)$ of the process terminating at any step.

Many models also use the convention of an **absorbing state**. Once entered, the process remains in this state forever with zero future reward, effectively ending the accumulation of value.

The overall learning process typically spans many such episodes, stopping only when a performance goal is met or a computational budget is exhausted.

---

## **7. What Constitutes a "Solution"?**

A "solution" to a sequential decision process is a **policy**—a rule that tells the agent which action to take in any given situation. The nature of this solution depends on whether there is one agent or multiple agents.

* **Single-Agent (MDP):** The solution is an **optimal policy ($\pi^*$**). This policy maximizes the expected cumulative discounted reward from any starting state. While the optimal *value* (expected return) is unique, there may be multiple different policies that achieve it.

* **Multi-Agent (Stochastic Game):** The concept of a solution becomes more complex, as each agent's optimal action depends on the actions of others. The solution is a **joint policy** that satisfies a chosen **solution concept** from game theory, which defines a stable or desirable outcome. Key solution concepts include:
    * **Nash Equilibrium:** A joint policy where no single agent can improve its own reward by unilaterally changing its strategy, assuming all other agents' strategies remain fixed.
    * **Minimax Equilibrium (for two-player, zero-sum games):** A joint policy where each agent maximizes their return against a worst-case opponent.
    * **$\epsilon$-Nash Equilibrium:** A relaxation where no agent can improve its return by more than a small amount $\epsilon$ by deviating.
    * **Correlated Equilibrium:** A generalization of Nash equilibrium where a central correlating device can suggest actions to agents, leading to potentially better and more stable collective outcomes.


### Formalizing Multi-Agent Solution Concepts

* **Best Response:** An agent $i$'s policy $\pi_i$ is a best response to the other agents' policies $\vec \pi_{-i}$ if it maximizes agent $i$'s utility function $U_i$.

$$
\pi_i \in \text{BR}(\vec \pi_{-i}) \iff U_i(\pi_i, \vec \pi_{-i}) \ge U_i(\pi_i^\prime, \vec \pi_{-i}) \quad \forall \pi_i^\prime \in \Pi_i
$$

* **Nash Equilibrium:** A joint policy $\vec \pi^\ast = (\pi_1^\ast, \dots, \pi_n^\ast)$ is a Nash Equilibrium if every agent's policy is a best response to the policies of the others. No agent can gain by unilaterally deviating.

$$
\forall i, \quad U_i(\pi_i^\ast, \vec \pi_{-i}^\ast) \ge U_i(\pi_i, \vec \pi_{-i}^\ast) \quad \forall \pi_i \in \Pi_i
$$

* **Pareto Optimality:** A joint policy $\vec{\pi}$ is Pareto optimal if there is no other policy $\vec{\pi}^\prime$ that improves at least one agent's utility without harming any other agent.

* **Social Welfare:** This measures the collective utility of all agents. A welfare-maximizing solution is one that maximizes this sum:

$$
W(\vec \pi) = \sum_{i=1}^n U_i(\vec \pi)
$$

These formal definitions provide the language to analyze and solve the complex strategic interactions in multi-agent systems.


Solutions can also be evaluated using other criteria beyond stability:

* **Pareto Optimality:** A joint policy where it's impossible to make one agent better off without making at least one other agent worse off. A solution can be a Nash equilibrium but not be Pareto optimal.
* **Social Welfare and Fairness:** Solutions can be judged by their collective outcomes. **Welfare-optimal** policies maximize the sum of all agents' returns ($\sum U_i(\pi)$). **Fairness-optimal** policies maximize the product of returns ($\prod U_i(\pi)$), which promotes equity.
* **No-Regret:** This concept focuses on the learning process itself. An agent achieves "no-regret" if its performance over time is asymptotically as good as the best fixed strategy in hindsight. Learning algorithms that guarantee no-regret often converge to a set of correlated equilibria.




