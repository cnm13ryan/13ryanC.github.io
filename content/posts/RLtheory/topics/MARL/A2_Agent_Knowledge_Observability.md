---
date: "2025-07-20"
title: "A2. Agent Knowledge and Observability in Multi-Agent Systems"
summary: "A2. Agent Knowledge and Observability in Multi-Agent Systems"
lastmod: "2025-07-20"
category: "Notes"
series: ["RL Topics", "MARL"]
author: "Bryan Chan"
hero: /assets/images/hero3.png
image: /assets/images/card3.png
---

# 2 Agent Knowledge and Observability in Multi-Agent Systems

In the study of multi-agent systems (MAS), a central challenge is understanding what each agent knows, what it can see, and how it reasons about others. This exploration is broken down into four key questions, now supplemented with formal definitions to provide greater clarity.

---

## 1. What Do Agents Know About the World and Each Other?

> Given limited perceptions and interactions, what can the agents genuinely be sure of—both about objective reality and about what the other agents believe or know?

This question asks us to define an agent's informational state, which includes its knowledge and beliefs about:
1. The external environment ("the world").
2. The informational states of other agents (often called meta-knowledge or higher-order beliefs)


### Why This Is Important
Most real-world problems—from self-driving cars and trading bots to cooperative robotics—involve knowledge of both the environment and other agents. Mistakes in acquiring or deducing this knowledge can have catastrophic results. 

Even two perfectly rational agents with common priors may disagree on posterior beliefs if their observations diverge, raising paradoxes central to game theory (e.g., Aumann's Agreement Theorem). The question is also vital for designing secure communication and authentication protocols.

A single agent with perfect observability is simple but unrealistic. Adding just one more agent introduces nested, potentially infinite reasoning (“I know that you know that I know…”). This creates several core tensions:
* **Observability vs. Cost:** Rich sensing narrows uncertainty but is expensive or impossible.
* **Privacy vs. Coordination:** Sharing knowledge improves teamwork but may expose strategic vulnerabilities.
* **Logical Omniscience vs. Bounded Rationality:** Classical logic assumes agents know all consequences of their information—an inhuman idealization.
 

### The Spectrum of Knowledge: From Complete to Incomplete

The assumptions underpinning an agent's knowledge exist on a spectrum, which is best understood by formalizing the "game" it is participating in. This spectrum ranges from the idealized world of complete information, characteristic of classical game theory, to the more realistic paradigm of incomplete information found in multi-agent reinforcement learning.


#### **Complete Information: The Classical Game Theory Paradigm**

Traditional game theory operates under the foundational assumption of **complete information**, where the entire structure of the game is common knowledge. This means every agent possesses full and accurate information about the rules, the other agents, their potential actions, and the consequences of those actions.

This idealization allows for precise analytical solutions. The game's structure is formally defined:

* **Normal-Form Game**: A static interaction is described by $G = \langle N, A, R \rangle$.
    * $N = \lbrace 1, ..., n\rbrace$: The set of agents.
    * $A = A_1 \times \dots \times A_n$: The joint action space, which is the Cartesian product of the individual action sets $A_i$. An element $a \in A$ is an **action profile** $a = (a_1, \dots, a_n)$, where $a_i$ is the action chosen by agent $i$.
    * $R = (R_1, \dots, R_n)$: The set of reward functions. For each agent $i$, $R_i(a)$ specifies the numerical reward for that agent given the joint action $a$.

* **Stochastic Game** (Markov Game): For dynamic environments, the model is $SG = \langle N, S, A, P, R \rangle$.
    * $S$: A set of environment states.
    * $P$: The state transition function, $P(s' | s, a)$, which defines the probability of transitioning from state $s$ to state $s'$ after the agents take the joint action $a$. It must satisfy $\sum_{s' \in S} P(s' | s, a) = 1$.
    * $R_i(s, a)$: The reward function for agent $i$ now depends on the state $s$ as well as the joint action $a$.

With complete knowledge, agents can predict each other's behaviour. This leads to powerful solution concepts like the **Nash Equilibrium**. A joint strategy profile $\pi^\ast = (\pi_1^\ast, \dots, \pi_n^\ast)$ is a Nash Equilibrium if no single agent can improve its own expected reward by unilaterally changing its strategy. Formally, for every agent $i$ and for any alternative strategy $\pi_i$:

$$
\mathbb E_{s \sim d, a \sim \pi^\ast}[R_i(s,a)] \ge \mathbb E_{s \sim d, a \sim (\pi_i, \pi_{-i}^\ast)}[R_i(s,a)]
$$

where $\pi_{-i}^\ast$ denotes the equilibrium strategies of all agents except $i$, and the expectation is over the game's outcomes.

---

#### **Incomplete Information: The MARL Paradigm**

MARL addresses more realistic scenarios of **incomplete information**, where agents have only a partial, local view of the system. The mathematical model that captures this is the **Decentralized Partially Observable Markov Decision Process (Dec-POMDP)**. It is defined by the tuple $\langle N, S, A, P, R, \Omega, O \rangle$.

Compared to a Stochastic Game, a Dec-POMDP introduces:
* $\Omega = (\Omega_1, \dots, \Omega_n)$: A set of individual observations for each agent.
* $O$: The observation function. After a transition to state $s'$, the environment issues a joint observation $(o_1, \dots, o_n)$, where agent $i$'s observation $o_i \in \Omega_i$ is drawn from the probability distribution $O(o | s', a)$.

In this setting, an agent $i$ does not know the true state $s$. Instead, it only receives its private observation $o_i$ and its own reward $r_i$. It does not know the other agents' rewards, observations, or the underlying state transition function $P$. To cope with this uncertainty, an agent must maintain a **belief state**, $b(s)$, which is a probability distribution over the possible true states $S$, updated based on its history of actions and observations.

The agent's goal is to learn a **policy** $\pi_i$ that maps its history (or its derived belief state) to an action. The search for an optimal policy is fundamentally harder because the agent cannot directly observe the full state required to make a perfectly informed decision.

---

#### **The Learning Objective: Maximizing Value**

The objective for each agent is to find a policy $\pi_i$ that maximizes its long-term cumulative reward. This is captured by the **value function**.

* **State-Value Function ($V$-function)**: The expected return starting from state $s$ and following a joint policy $\vec{\pi} = (\pi_1, \dots, \pi_n)$.
    $$
    V_i^{\vec{\pi}}(s) = \mathbb{E}\left[\sum_{t=0}^{\infty} \gamma^t r_{i,t+1} \mid S_t=s, \vec{\pi}, P\right]
    $$
    The expectation is over all possible future trajectories $\tau = (s_0, a_0, s_1, a_1, \dots)$, where the probability of a specific trajectory is determined by both the environment dynamics $P(s_{t+1}|s_t, a_t)$ and the collective actions sampled from the joint policy $\vec{\pi}(a_t|s_t) = \prod_j \pi_j(a_{j,t}|s_t)$. Since the agent knows neither $P$ nor $\pi_j$ for $j \neq i$, this cannot be computed directly.

* **Action-Value Function ($Q$-function)**: More commonly used in learning algorithms, this function gives the expected return after taking a specific joint action $a$ in a state $s$.
    $$
    Q_i^{\vec{\pi}}(s, a) = \mathbb{E}\left[\sum_{t=0}^{\infty} \gamma^t r_{i,t+1} \mid S_t=s, A_t=a, \vec{\pi}, P\right]
    $$
    An agent would ideally choose its action $a_i$ to maximize this Q-value. However, this still requires knowing the concurrent actions of others, $a_{-i}$, which is a central challenge in MARL.

---

### Formalizing Knowledge with Epistemic Logic

**Epistemic logic** provides a formal language to reason about states of knowledge. The semantics are given by a **Kripke Model**, $M = \langle W, \lbrace \sim_i\rbrace_{i \in N}, V \rangle$:

* $W$ is a set of possible worlds (e.g., all possible configurations of a game state).
* $\sim_i \subseteq W \times W$ is an indistinguishability relation for agent $i$. If $w \sim_i w'$, it means agent $i$ cannot tell the difference between world $w$ and world $w'$, given its information.
* $V$ is a valuation function that assigns truth values to basic propositions in each world.

Within this model, the statement "$i$ knows $\phi$" (written $K_i \phi$) is true in world $w$ if and only if $\phi$ is true in *all* worlds $w'$ that $i$ considers possible:
$$(M, w) \models K_i \phi \iff \forall w' \in W, (w \sim_i w' \implies (M, w') \models \phi)$$
This mathematical structure provides the foundation for defining concepts like "common knowledge" (where the indistinguishability relations are shared and iterated) and an agent's private, partial information in a Dec-POMDP.


| Strand | Core Idea | Canonical Tools / Results | Limitations |
|---|---|---|---|
| *Epistemic logic* | Possible-world models with accessibility relations encode what each agent counts as possible. | Kripke frames, S5 axioms, Common Knowledge operator (CK) | Logical omniscience and binary certainty; no graded belief |
| *Probabilistic epistemics* | Agents hold probability distributions over worlds/world–agent pairs. | Bayesian games, Interactive POMDPs, Harsanyi type spaces | Infinite regress of types; computational intractability |
| *Distributed-systems knowledge*| “Knowledge in distributed systems” analyses timing, message order, and fault models. | Knowledge-based protocols, coordinated attack impossibility, clock synchrony | Assumes discrete message-passing; rarely continuous dynamics |
| *Empirical AI / ToM* | Learned models approximate other agents’ belief states. | Deep theory-of-mind networks, recursive reasoning RL | Hard to verify; opaque representations |

### Necessary Conditions and Foundational Works
Any viable account of agent knowledge requires a finite but expressive representation of nested beliefs, compatibility with noisy perception, and computationally tractable update rules.

Key foundational works include:
* Robert Aumann. *Agreeing to Disagree*. Annals of Statistics, 1976.
* Joseph Y. Halpern & Yoram Moses. *Knowledge and Common Knowledge in a Distributed Environment*. JACM, 1990.
* Ronald Fagin, et al. *Reasoning About Knowledge*. MIT Press, 1995.

---

## 2. What Can Agents Observe Directly vs. What Must They Infer?

> For every participant in a system, which facts come “for free” through sensing or direct disclosure, and which facts must be pieced together from partial evidence, prediction, or negotiation?

This question distinguishes between **direct observation**--information acquired innately through an agent's sensors--and **reasoned inference**, the proceess of constructing a more complete model of the world from incomplete data. 

The ability to correctly navigate this distinction is important for designing robust and secure autonomous systems.

An agent that misattributes an inference as a direct, infallible observation can cause catastrophic failures (e.g., an autonomous vehicle assuming an intersection is clear based on the absence of immediate sensory data) or critical security breaches (e.g., a financial trading algorithm trusting spoofed market data as genuine).

This challenge is rooted in an inherent tension between an agent's need for situational awareness and the practical limitations of information acquisition. The core sources of this tension can be broken down as follows:

The core tension arises from:
1.  **Cost vs. Completeness:** 
    * Acquiring perfect information is often physically impossible or economically prohibitive. 
    * Adding more, or higher-fidelity, sensors (like LiDAR, radar, and high-resolution cameras) increases the material cost, computational load, and energy consumption of an agent. 
    * Furthermore, pervasive sensing raises significant privacy and data security risks, which may be unacceptable or illegal in many contexts. 
    * Systems must therefore operate with an information deficit, making inference a necessity.

2.  **Local vs. Global Knowledge:** 
    * Most agents operate with inherently local sensory data; they can only perceive their immediate surroundings. 
    * However, optimal decision-making often requires global awareness—understanding the broader system state, including the locations, intentions, and status of other agents. 
    * For example, a robot in a warehouse only sees its current aisle but needs to infer overall inventory levels, the positions of other robots, and human traffic patterns to execute its tasks efficiently and safely. 
    * This gap between local perception and the need for global context forces agents to infer the global state from local cues.

3.  **Truthfulness vs. Strategic Opacity:** 
    * In competitive or mixed-motive environments (e.g., auctions, negotiations, or military engagements), other agents have strategic incentives to be opaque. 
    * They may deliberately withhold, obscure, or even falsify information to gain an advantage. 
    * This forces an agent to treat incoming information not as ground truth but as a signal that must be critically evaluated. 
    * The agent must infer the true state of affairs by reasoning about the incentives and likely strategies of its counterparts, a process fraught with uncertainty.


This distinction between observation and inference is formalized within the mathematical frameworks of game theory and reinforcement learning, particularly in models designed for dynamic, multi-agent environments.
 
---

### **Full Observability: Stochastic Games**

The simplest interactive model, which assumes away the problem of partial information, is the **stochastic game**, also known as a Markov game. It extends the single-agent Markov Decision Process (MDP) to a multi-agent setting.

A stochastic game is formally defined by a tuple $G = \langle N, S, \mathbf{A}, T, \mathbf{R} \rangle$, where:
* $N$: A finite set of $n$ agents.
* $S$: A finite set of environment states. In this model, the state $s \in S$ is **fully and directly observable** to all agents at every timestep.
* $\mathbf{A} = A_1 \times \dots \times A_n$: A set of joint actions, where $A_i$ is the action set for agent $i$.
* $T: S \times \mathbf{A} \times S \to [0, 1]$: The state transition function. This function encodes the environment's dynamics. The expression $T(s, \mathbf{a}, s')$ gives the probability that the environment will transition from its current state $s$ to a new state $s'$ if the agents execute the joint action $\mathbf{a} = (a_1, \dots, a_n)$.
* $\mathbf{R} = R_1, \dots, R_n$: A set of reward functions. The function $R_i: S \times \mathbf{A} \to \mathbb{R}$ determines the immediate numerical reward agent $i$ receives after the joint action $\mathbf{a}$ is taken in state $s$.

#### **The Agent's Objective**
An agent's goal is not simply to get a high immediate reward, but to maximize its total expected reward over an entire episode or lifetime. This is formalized as the **expected discounted return**. 

Agent $i$ follows a **policy** $\pi_i$, which is a strategy that maps states to actions (or a distribution over actions), $\pi_i: S \to \Delta(A_i)$. The objective is to find an optimal policy $\pi_i^\ast$ that maximizes the value function $V_i^{\pi}$:
$$
V_i^{\pi}(s) = \mathbb E_{\pi} \left[ \sum_{t=0}^{\infty} \gamma^t R_i(S_t, \mathbf A_t) \mid S_0=s \right]
$$
where:
* $\pi = (\pi_1, \dots, \pi_n)$ is the joint policy of all agents.
* $\gamma \in [0, 1)$ is the **discount factor**, which prioritizes immediate rewards over future ones. A $\gamma$ close to 0 makes the agent "myopic," while a $\gamma$ close to 1 makes it "far-sighted."
* The expectation $\mathbb{E}_{\pi}$ is taken over the sequence of states and actions induced by all agents following their respective policies.

Even with full observability, agents typically do not know the functions $T$ and $\mathbf{R}$ and must infer them through exploration.


### **Partial Observability: The Realistic Default**

The most realistic and complex scenario is modeled by the **Partially Observable Stochastic Game** (POSG). A POSG is formally a tuple $G = \langle N, S, \mathbf{A}, T, \mathbf{R}, \mathbf{\Omega}, O \rangle$, with two additional components:

* $\mathbf{\Omega} = \Omega_1 \times \dots \times \Omega_n$: The joint observation space. Each $\Omega_i$ is the set of all possible private observations for agent $i$.
* $O$: The observation function, $O: S \times \mathbf{A} \to \Delta(\mathbf{\Omega})$. Here, $\Delta(\mathbf{\Omega})$ denotes the set of probability distributions over $\mathbf{\Omega}$. The function gives the probability of receiving a joint observation $\mathbf{o} \in \mathbf{\Omega}$ given the system just transitioned to state $s$ after joint action $\mathbf{a}$ was taken.

In a POSG, an agent must maintain a **belief state**, $b_i$, which is a probability distribution over the true states $S$. 

This belief is a **sufficient statistic** for the agent's past experiences; it summarizes the entire action-observation history $h_i^t = (a_i^0, o_i^1, \dots, a_i^{t-1}, o_i^t)$ into a single probability distribution that is sufficient for optimal decision-making.

The agent's policy now maps beliefs to actions, $\pi_i: \mathcal{B} \to \Delta(A_i)$, where $\mathcal{B}$ is the space of all possible belief states.

#### **Mathematical Detail of the Belief Update**
The belief is updated recursively using a **Bayesian filter**. The formula for updating belief $b_i$ to $b_i^\prime$ after taking action $a_i$ and receiving observation $o_i$ is:
$$
b_i^\prime (s') = \eta P(o_i \mid s', a_i) \sum_{s \in S} P(s' \mid s, \mathbf a) b_i(s)
$$

Let's dissect this update step-by-step:

1.  **Prediction:** The agent first predicts the next state distribution before accounting for the new observation. This is the summation term:
    $$
    \hat b_i(s') = \sum_{s \in S} T(s, \mathbf a, s') b_i(s)
    $$
    This is an application of the Law of Total Probability. It calculates the predicted belief $\hat{b}_i(s')$ by summing over every possible previous state $s$, weighting its transition probability $T(s, \mathbf a, s')$ by the agent's prior belief $b_i(s)$ that it was in that state.
    * **Challenge**: Note that the transition $T$ depends on the joint action $\mathbf{a}$, but agent $i$ only knows its own action $a_i$. To compute this, the agent must model or infer the other agents' actions, $\mathbf{a}_{-i}$.

2.  **Correction (Update):** Next, the agent corrects its prediction using the new evidence—its observation $o_i$. This is done by multiplying the predicted belief by the observation probability:
    $$
    b'_{i, \text{unnormalized}}(s') = P(o_i \mid s', a_i) \times \hat{b}_i(s')
    $$
    The term $P(o_i \mid s', a_i)$ comes from the observation function $O$ and represents the likelihood of seeing $o_i$ if the world were truly in state $s'$. This step re-weights the predicted belief, increasing the probability of states that are consistent with the observation and decreasing it for states that are not.

3.  **Normalization:** The resulting belief $b_{i, \text{unnormalized}}^\prime$ is not yet a valid probability distribution (it doesn't sum to 1). It is normalized by dividing by the probability of the evidence, $P(o_i \mid h_i^t, a_i)$. This gives the normalization constant $\eta$:
    $$
    \eta = \frac{1}{\sum_{s' \in S} P(o_i \mid s', a_i) \hat b_i(s')}
    $$
    The final, complete update is $b_i^\prime (s') = \eta \times b_{i, \text{unnormalized}}^\prime (s')$, which is the application of Bayes' rule. This recursive update allows an agent to continuously refine its understanding of the hidden state as it interacts with its world.

---

## 3. Who Knows That Others Know What?

> How is information distributed among agents, and to what depth do they recognise each other’s knowledge?

This fundamental question addresses the architecture of belief and knowledge within a multi-agent system. The way information is shared—or withheld—critically determines the potential for cooperation, coordination, and conflict. Understanding this distribution requires moving beyond what an individual agent knows, to what it knows about others' knowledge, and what it knows about what others know about its knowledge, ad infinitum. This recursive nature of knowledge can be stratified into distinct levels.

* **Asymmetric Knowledge:** This is the most common state, where an agent possesses private information that others do not. In a card game, for example, a player knows the cards in their hand, but their opponents do not. This information imbalance is a primary driver of strategic action.

* **Mutual Knowledge:** A fact is considered mutual knowledge if every agent in a given group knows it. If a public announcement is made to a room of agents—for instance, "the meeting is at 3 PM"—then every agent knows the meeting time. However, an agent does not necessarily know that every other agent also heard the announcement.

* **Common Knowledge:** This represents the deepest level of shared information. A fact is common knowledge if it is mutually known, and it is also mutually known that it is mutually known, and so on, in an infinite recursion. It is the state of complete informational transparency where "everyone knows, everyone knows that everyone knows, and so on."

### Why This Is Important

The distinction between these levels is not merely academic; it has profound practical consequences across various domains.

* **Coordination & Conventions:** Social conventions and public safety systems rely on common knowledge. Consider a traffic light turning green. For you to proceed safely, it is not enough for you to know the rule (green means go) and for others to know the rule (mutual knowledge). You must also know that they know the rule, and that they know that you know the rule. This shared, recursive certainty prevents hesitation and enables smooth, decentralized coordination.

* **Strategic Interaction:**  In financial markets, asymmetric knowledge creates opportunities for profit. An insider with private information about a company's future earnings can exploit this advantage. The goal of regulations against insider trading is to prevent such asymmetries and move the market closer to a state where material information is common knowledge, ensuring fairness.

* **Security & Trust:** Distributed computational systems, such as blockchains or databases, must achieve consensus on the state of a ledger. This requires protocols where messages are broadcast and acknowledged in a way that creates a sufficiently strong, albeit technically finite, approximation of common knowledge. Each node must be certain that other nodes have received the same messages to prevent the system from fracturing.

### Formal Epistemic Logic Framework

These concepts can be defined with mathematical precision using epistemic logic. While the logical syntax provides the language for expressing knowledge, the meaning of these expressions is grounded in a formal mathematical structure known as a Kripke model.

#### The Mathematical Semantics of Knowledge (Kripke Models)

To move past jargon, we can formally define knowledge using an **Epistemic Model** (or Kripke structure), denoted as a tuple $M = (W, \lbrace \sim_i\rbrace_{i \in G}, V)$. This structure provides a concrete way to reason about "possible worlds."

* $W$ is a non-empty set of **possible worlds** (or states). A world represents a complete description of how things might be.
* $\lbrace \sim_i\rbrace_{i \in G}$ is a set of **accessibility relations**, one for each agent $i$ in the group $G$. The relation $\sim_i \subseteq W \times W$ captures the uncertainty of agent $i$. We write $w \sim_i w'$ to mean "in world $w$, agent $i$ considers world $w'$ to be possible." If an agent cannot distinguish between $w$ and $w'$, then everything it knows in $w$ must also be true in $w'$. For knowledge, these relations are typically **equivalence relations** (reflexive, symmetric, and transitive).
* $V$ is a **valuation function** that assigns truth values to primitive propositions in each world. For a proposition $p$, $V(w, p)$ is either true or false.

With this structure, the statement **"agent $i$ knows $\phi$"**, denoted $K_i \phi$, is given a precise meaning:
$$(M, w) \models K_i \phi \iff \forall w' \in W, (w \sim_i w' \implies (M, w') \models \phi)$$
In plain terms: "Agent $i$ knows $\phi$ in world $w$" is true if and only if "$\phi$ is true in all worlds $w'$ that agent $i$ considers possible from $w$."

#### Defining Shared Knowledge Mathematically

Using this formal model, we can now define the levels of shared knowledge with greater clarity:

1.  **Mutual Knowledge ($E_G \phi$):** This is the case where every agent in group $G$ knows $\phi$.
    $$(M, w) \models E_G \phi \iff \forall i \in G, (M, w) \models K_i \phi$$
    This simply means the condition for $K_i \phi$ holds for all agents in the group.

2.  **Common Knowledge ($C_G \phi$):** The "infinite recursion" of common knowledge has a direct and elegant mathematical interpretation. First, let's define the iterative sequence:
    * $E_G^1 \phi \equiv E_G \phi$ (Everyone knows $\phi$)
    * $E_G^2 \phi \equiv E_G(E_G \phi)$ (Everyone knows that everyone knows $\phi$)
    * $E_G^k \phi \equiv E_G(E_G^{k-1} \phi)$
    * Common knowledge is the infinite conjunction: $C_G \phi \equiv \bigwedge_{k=1}^{\infty} E_G^k \phi$.

Operationally, this is captured by reachability in the Kripke model. 

Let $R_G = \bigcup_{i \in G} \sim_i$ be the union of all individual accessibility relations. Let $R_G^*$ be the **reflexive, transitive closure** of $R_G$. This new relation $w \ R_G^\ast \ w'$ means that world $w'$ is reachable from $w$ by traversing any number of accessibility links belonging to any agent in $G$.

Common knowledge is then defined as truth across all reachable worlds:
$$
(M, w) \models C_G \phi \iff \forall w' \in W, (w \ R_G^\ast \ w' \implies (M, w') \models \phi)
$$
This provides a clear, operational definition: **$\phi$ is common knowledge** if it is true not only in the current world, but in every world that is reachable through any chain of reasoning about what any agent considers possible.

### Concluding Remarks

While common knowledge is a cornerstone of classical game theory, used to justify equilibria in games of complete information, its direct application in Multi-Agent Reinforcement Learning (MARL) is less prominent. In MARL, the environment is often partially observable and agents must learn foundational facts through interaction and trial-and-error, making the problem of achieving even mutual knowledge a significant challenge. The infinite, higher-order beliefs of common knowledge are rarely modelled explicitly. Instead, the focus is on learning policies that lead to successful coordination, which can be seen as an emergent, implicit form of shared 

---

## 4. What Level of Observability Should Agents Have?

> How much of the world should an intelligent agent be allowed (or required) to “see,” and how much of the agent should the rest of us be able to “see”?

This is a design question that governs the interaction between an agent and its environment, as well as between the agent and its human overseers. 

Answers to this question involves balancing critical trade-offs between performance, computational cost, and safety. This issue can be decomposed into two distinct, yet related, domains:

1. The agent's observability of the world
2. Our observability of the agent

### Perceptual Observability: The Agent's Worldview

An agent's perceptual observability hings on its ability to perceive its environment, aka the extent to which an agent can access to the "true" state of its environment.

This is a foundational choice in its design, formally captured by the mathematical frameworks used to model its decision-making process, which defines the trade-space for perceptual observability, betweeen complete and partial observability.

The two primary models, Stochastic Games (SGs) and Partially Observable Stochastic Games (POSGs), represent the poles of this spectrum.

#### **Full Observability (Stochastic Games):** 

Choosing a **Stochastic Game (SG)**, also known as a Markov Game, as the underlying model implies that every agent has perfect and complete information about the state of the world.

**Formalism:** A multi-agent SG is defined as a tuple $\langle S, \lbrace A_i\rbrace_{i=1..N}, T, \lbrace R_i\rbrace_{i=1..N} \rangle$, where:
* $S$ is the set of all possible world states.
* $A_i$ is the set of actions for agent $i$.
* $T: S \times A \to \Delta(S)$ is the transition function, where $P(s' | s, \vec{a})$ gives the probability of moving to state $s'$ from state $s$ after joint action $\vec{a} = \langle a_1, ..., a_N \rangle$.
* $R_i: S \times A \to \mathbb{R}$ is the reward function for agent $i$.

The crucial assumption is that the true state $s \in S$ is directly provided to each agent. This satisfies the **Markov Property**, $P(s_{t+1} \mid s_t, \vec a_t) = P(s_{t+1} \mid s_t, \vec a_t, ..., s_0, \vec a_0)$, meaning the current state contains all information needed for optimal decision-making. This simplifies an agent's **policy** $\pi_i$, which becomes a direct mapping from states to actions, $\pi_i: S \to A_i$. However, assuming full observability is often physically impossible or prohibitively expensive.

#### **Partial Observability (POSGs):** 

Choosing a **Partially Observable Stochastic Game (POSG)** is a more realistic approach. Here, an agent receives only a private observation—a piece of probabilistic evidence about the state.

**Formalism:** A POSG extends the SG tuple with observation components: $\langle S, \lbrace A_i\rbrace, T, \lbrace R_i\rbrace, \lbrace \Omega_i\rbrace, O \rangle$.
* $\Omega_i$ is the set of possible observations for agent $i$.
* $O$ is the observation function, where $P(\vec{o} | s', \vec{a})$ gives the probability of the agents receiving the joint observation $\vec{o} = \langle o_1, ..., o_N \rangle$ after transitioning to state $s'$.

This uncertainty forces the agent to perform **belief state tracking**. It maintains a **belief state** $b_i$, a probability distribution over all possible world states ($b_i \in \Delta(S)$). After taking action $a_i$ and receiving observation $o_i'$, the agent updates its belief from $b_i$ to $b_i'$ using a Bayesian filter:
$$
b_i'(s') = \eta P(o_i' | s') \sum_{s \in S} P(s' | s) b_i(s)
$$
where $\eta$ is a normalising constant. This continuous update is computationally expensive. The agent's policy must now map from this high-dimensional belief space to actions, $\pi_i: \Delta(S) \to A_i$, which is a significantly harder problem to solve.


---

### Operational Observability: Our View of the Agent


The other side of this coin is **operational observability**—often referred to as transparency or, more broadly, **AI Observability**. This concerns our ability to inspect an agent's internal state and decision-making calculus. This includes its **policy** $\pi_i$ (its strategy), its **value function** $V_i^{\pi}$ (its prediction of future rewards), or its **belief state** $b_i$.

**Formalism:** An agent's goal is to learn a policy $\pi_i$ that maximizes its **value function** $V_i^{\pi}$, which is the expected sum of discounted future rewards:

$$
V_i^{\pi}(s_0) = \mathbb E_{\pi} \left[ \sum_{t=0}^{\infty} \gamma^t R_i(s_t, \vec a_t) \mid s_0 \right]
$$

where $\gamma \in [0, 1)$ is a discount factor that prioritizes sooner rewards. Transparency means having access to the parameters that define $\pi_i$ and $V_i^{\pi}$ to understand *how* the agent arrives at its decisions.

Greater transparency is a prerequisite for robust debugging, safety verification, and legal accountability. In high-stakes environments, the ability to audit an agent's decision logic is essential for trust and compliance, a central goal of **Explainable AI (XAI)**. However, maximizing transparency can expose proprietary logic and create overwhelming data management challenges.

Ultimately, the choice of observability level is determined by balancing the agent's computational complexity against the physical costs of sensing and the pressing need for external oversight.

| School of thought | Slogan | Rationale | Objections |
| :--- | :--- | :--- | :--- |
| **Max-information** | “See everything; log everything.” | Maximises agent performance by providing complete data. Enables comprehensive debugging, auditing, and post-hoc analysis. | Often physically impossible or financially prohibitive. Creates significant privacy risks, conflicting with regulations like GDPR. Generates massive, costly-to-manage datasets and increases the system's attack surface. |
| **Minimal-need** | “Only observe what you strictly need.” | Aligns with the principle of data minimization in privacy law. Reduces computational load and data storage costs. Constrains the agent's capabilities, which can reduce the attack surface for adversarial manipulation. | Makes debugging extremely difficult as root causes may be unobserved. Creates the risk of "unknown unknowns" or critical blind spots, where the agent is unaware of rare but crucial environmental factors, leading to catastrophic failure. |
| **Adaptive/elastic**| “Dial observability up or down on demand.” | Offers a dynamic balance between the above extremes. Low-cost, privacy-preserving operation is the default, but full-observability "flight recorder" mode can be triggered for debugging, anomaly detection, or periodic audits. | Adds significant architectural and computational complexity to the system. Defining robust triggers for changing observability levels is non-trivial. Creates potential for "observer effects" or gaming, where agents learn to alter their behavior when they detect they are under closer scrutiny. |


