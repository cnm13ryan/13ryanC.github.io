---
date: "2025-07-10"
title: "(4.1) Planning with a Known Model (Online Planning, MCTS)" 
summary: "Planning with a Known Model (Online Planning, MCTS)"
lastmod: "2025-07-10"
category: "Notes"
series: ["RL Theory"]
author: "Bryan Chan"
hero: /assets/images/hero3.png
image: /assets/images/card3.png
---

# Online Planning: A Formal Treatment

This section pivots from the *offline* planning setting to *online* planning. In the offline model, where the MDP is described by explicit transition and reward tables, no algorithm can find a nearly optimal policy with a computational cost less than $\Omega(|S|^2|A|)$. This "curse of dimensionality" renders tabular methods intractable for problems with large state spaces.

The online planning model circumvents this limitation by changing two fundamental assumptions:

1. **Goal**: The planner no longer needs to compute a complete policy $\pi: S \to A$. Instead, it only needs to produce a single action $a_t$ for the *current* state $s_t$.
2. **Information Access**: The planner does not read from transition tables. It interacts with a **simulator** (or generative model) of the environment.

---

## 1.1 Formalizing the Online Planning Problem

We build our definitions upon the measure-theoretic framework of an MDP. An MDP is a tuple $(S, A, \kappa, \gamma)$, where the state space $S$ and action space $A$ are **standard Borel spaces**, and $\kappa$ is the unified transition kernel.

**Definition: MDP Simulator**

An MDP simulator is an oracle that, when queried with a state-action pair $(s, a)$, returns a single random outcome $(s', r')$ sampled from the joint probability measure $\kappa(ds'dr' | s, a)$.

This replaces the explicit representation of the probability measure $p(ds'|s, a)$ and reward function $r(s, a)$ with a mechanism to generate samples.

---

**Definition: Online Planner**

An online planner is an algorithm that takes a state $s$ and access to an MDP simulator, queries the simulator a finite number of times, and returns an action $a$.

When deployed in an environment, the planner and the MDP simulator together induce a **planner-induced policy**, $\pi$. The objective is for this policy to be nearly optimal.

---

**Definition: $\varepsilon$-Optimal Online Planner**

An online planner is **$\varepsilon$-optimal** if, for any MDP $M$, the planner-induced policy $\pi$ is $\varepsilon$-optimal in $M$. That is, its value function $v_\pi$ satisfies $v_\pi \geq v^\ast - \varepsilon \mathbf{1}$ for all states, where $v^\ast$ is the optimal state-value function. This is also referred to as being $\delta$-sound, where $\delta = \varepsilon$.

The primary performance metric is the planner's **query cost**: the maximum number of simulator queries it makes for any given state.

### A Concrete View: The Oracle Model for Finite MDPs

To make these concepts more concrete, we can specialize the definitions for the common case of finite MDPs, where the problem can be framed in terms of a "black-box oracle". In this view, we identify the state and action spaces with subsets of the natural numbers.

**Definition (MDP simulator):** A simulator implementing an MDP  $M = (S,\mathcal{A},P,r)$  is a "black- box oracle" that when queried with a state action pair  $(s,a)\in S\times \mathcal{A}$  returns the reward  $r_a(s)$  and a random state  $S^{\prime}\sim P_{a}(s)$.

**Definition (Online Planner):** An online planner takes as input the number of actions  $\mathbf{A}$, a state  $s \in \mathbb{N}$, and an MDP simulator "access point". After querying this simulator finitely many times, the planner must return an action from  $[\mathbf{A}]$.

A planner is **well-formed** if, for any MDP, it returns an action after a finite number of queries and never queries the simulator with an invalid state-action pair. The policy induced by a well-formed planner is always well-defined.

**Definition ($\delta$-sound Online Planner):** An online planner is **$\delta$-sound** if it is well-formed and for any MDP $M$, the induced policy $\pi$ is $\delta$-optimal in $M$. This is equivalent to the $\varepsilon$-optimality defined previously.

The following vignette summarizes the online planning problem from this perspective:

<table>
  <tr>
    <td><strong>Model:</strong></td>
    <td>Any finite MDP M</td>
  </tr>
  <tr>
    <td><strong>Oracle:</strong></td>
    <td>Black-box simulator of M</td>
  </tr>
  <tr>
    <td><strong>Local input:</strong></td>
    <td>State s</td>
  </tr>
  <tr>
    <td><strong>Local output:</strong></td>
    <td>Action A</td>
  </tr>
  <tr>
    <td><strong>Outcome:</strong></td>
    <td>Policy π</td>
  </tr>
  <tr>
    <td><strong>Postcondition:</strong></td>
    <td>$v^\pi_M \geq v^*_M - \delta \mathbf{1}$</td>
  </tr>
</table>


## 1.2 Online Planning with Action-Value Iteration

A natural approach to online planning is to adapt the iterative algorithms used for offline solutions. To select an action in state $s$, we need to estimate the optimal action-values $q^\ast(s, a)$. This suggests using the **Bellman optimality operator for action-value functions**, $T_q^*$.

The optimal value function $v^\ast$ is the fixed point of the Bellman optimality operator for state-value functions, $T^\ast$, and similarly, the optimal action-value function $q^\ast$ is the fixed point of the Bellman optimality operator for action-value functions, $T_q^\ast$. The operator $T_q^\ast$ is defined as:

$$(T_q^* q)(s, a) := r(s,a) + \gamma \int_S p(ds'|s,a) \sup_{a' \in A} q(s', a')$$

This is identical to the $\tilde{T}$ operator, where $(Mq)(s) = \sup_{a'\in A} q(s',a')$ and the integral corresponds to the expectation $\langle P_a(s), Mq \rangle$.

An online planner can approximate $q^\ast (s, a)$ by recursively computing $q_k(s, a) = ((T_q^\ast)^k \mathbf{0})(s, a)$ for a sufficient horizon $k$. The greedy policy extracted from $v_k$ is guaranteed to be $\varepsilon$-optimal if $k$ is large enough.

The recursive computation for $q_k(s, \cdot)$ can be expressed as:

```python
def q(k, s, simulator):
  if k == 0:
    return array_of_zeros
  
  # For each action, estimate the expected value of the next state
  action_values = []
  for a in Actions:
    # Use simulator to estimate the expectation
    # In the deterministic case, one call is enough.
    next_state_val = E_s'~p(·|s,a) [ max(q(k-1, s', simulator)) ] 
    action_values.append( r(s,a) + gamma * next_state_val )
  
  return action_values
```

In the general stochastic case, estimating the expectation requires multiple simulator calls. However, for a **deterministic MDP**, the transition kernel $p(ds'|s,a)$ is a Dirac delta measure $\delta_{g(s,a)}(ds')$, where $g: S \times A \to S$ is the deterministic transition function. The expectation becomes a simple function evaluation:

$$\int_S v(s') p(ds'|s,a) = v(g(s,a))$$

The recursive calculation for a deterministic MDP simplifies to:

```python
def q_deterministic(k, s, simulator):
  if k == 0: return array_of_zeros
  
  action_values = []
  for a in Actions:
    # Simulator returns the single next state g(s,a)
    next_s = simulator(s, a).next_state 
    val = r(s,a) + gamma * max(q_deterministic(k-1, next_s, simulator))
    action_values.append(val)

  return action_values
```

The runtime of this procedure is $O(A^k)$, which is **independent of the size of the state space $|S|$**. This demonstrates that, at least for deterministic environments, online planning can break the curse of dimensionality imposed by the tabular representation.

---

## 1.3 Lower Bound on Online Planning

While $O(A^k)$ is a significant improvement, we must ask if it's possible to do better. A lower bound shows that this dependency on the action-space size and horizon is fundamental.

**Theorem: Online Planning Lower Bound**

For any $\varepsilon$-optimal online planner with $\varepsilon < 1$ and rewards in $[0,1]$, there exists an MDP on which the planner must make $\Omega(A^k)$ queries at some state, where $A$ is the number of actions and $k$ is the effective horizon:

$$k = \left\lceil \frac{\ln(1 / (\varepsilon(1 - \gamma)))}{\ln(1 / \gamma)}\right\rceil$$

**Proof Sketch:**
The proof uses a "needle-in-the-haystack" argument. We construct a deterministic MDP whose state space is a full $A$-ary tree of depth $k$. The dynamics follow the tree structure. All rewards are zero except at one of the $A^k$ leaf nodes, where the reward is 1.

* The optimal policy must follow the path to this high-reward leaf. The value of this policy at the root is $\gamma^k / (1 - \gamma)$.
* Any other policy has a value of 0.
* For the planner to be $\varepsilon$-optimal, it must distinguish the optimal path from the others. Since $\gamma^k / (1 - \gamma) \geq \varepsilon$ by the definition of $k$, the planner must find the correct path.
* Finding this unique path is equivalent to finding the one leaf node with a non-zero reward, which requires exploring the tree. In the worst case, this necessitates $\Omega(A^k)$ queries to the simulator to traverse all paths to the leaves.

This result establishes that the $A^k$ complexity is near-optimal for online planning in a worst-case scenario.

---

## 1.4 Online Planning in Stochastic MDPs: The Sampling Approach

The core idea of online planning is to amortize the planning cost by having a planner produce a single action for the current state. When called repeatedly, this process induces a near-optimal policy. A key benefit of this approach is that the planning cost can be independent of the state space size, particularly for deterministic MDPs. This is achieved using a recursive implementation of value iteration based on action-value functions and the corresponding Bellman optimality operator for action-values, denoted here as $T$:

$$
Tq(s,a) = r_a(s) + \gamma \langle P_a(s),Mq\rangle .
$$

Note: The notation for the Bellman optimality operator for action-value functions now simplifies from $T_q^\ast$ (used in Section 1.2) to $T$ for the remainder of this discussion.

Lower bounds show that no procedure can significantly improve upon the runtime of this recursive method in the worst case. These foundational ideas can also be extended to stochastic MDPs.

### Sampling May Save the Day?

Assume now that the MDP is stochastic. The integral in the Bellman operator, $\langle P_a(s),Mq\rangle = \int_S Mq(s') p(ds'|s,a)$, involves an expectation over the next-state distribution. We can approximate this expectation by sampling, which allows the approximation error to be independent of the cardinality of the state space $S$. This is the key to avoiding the curse of dimensionality.

To formalize this, we introduce the concept of an **empirical measure**. 

For a given state-action pair $(s,a)$, we draw $m$ i.i.d. samples $S'_1, \dots, S'_m$ from the true transition kernel $p(\cdot \mid s, a)$. 

The empirical measure $\hat p_m(\cdot \mid s, a)$ is the random probability measure defined as:

$$
\hat p_m(ds' \mid s, a) := \frac{1}{m} \sum_{i=1}^{m} \delta_{S'_i}(ds')
$$

where $\delta_{S'_i}$ is the Dirac delta measure concentrated at the sample point $S'_i$.
 

To quantify the size of these errors, we recall Hoeffding's inequality:

**Lemma (Hoeffding's Inequality):** 

Given $m$ independent, identically distributed (i.i.d.) random variables that take values in the $[0,1]$ interval, for any $0 \leq \zeta < 1$ , with probability at least $1 - \zeta$ it holds that

$$
\left|\frac{1}{m}\sum_{i = 1}^{m}X_{i} - \mathbb{E}[X_{1}]\right|\leq \sqrt{\frac{\log\frac{2}{\zeta}}{2m}}.
$$

Letting $S_{1}^{\prime},\ldots ,S_{m}^{\prime}\stackrel {\mathrm{i.i.d.}}{\sim}p(\cdot|s,a)$ for some state- action pair $(s,a)$ and $v:S\to [0,v_{\max}]$, by this result, for any $0\leq \zeta < 1$ , with probability $1 - \zeta$

$$
\left|\frac{1}{m}\sum_{i = 1}^{m}v(S_i^{\prime}) - \langle P_a(s),v\rangle \right|\leq v_{\max}\sqrt{\frac{\log\frac{2}{\zeta}}{2m}}. \tag{1}
$$

This suggests the following approach: For each state action pair $(s,a)$ draw $S_{1}^{\prime},\ldots ,S_{m}^{\prime}\stackrel {\mathrm{i.i.d.}}{\sim}P_{a}(s)$ and store it in a list $C(s,a)$. Then, whenever for some function $v$ we need the value of $\langle P_a(s),v\rangle$ , just use the sample average:

$$
\frac{1}{m}\sum_{s'\in C(s,a)}v(s').
$$

Plugging this approximation into our previous pseudocode gives the following new code:

```
1. define q(k,s): 
2. if k = 0 return [0 for a in A] # base case 
3. return [r(s,a) + gamma/m * sum([max(q(k-1,s')) for s' in C(s,a)]) for a in A] 
4. end
```

The total runtime of this function is now $O((mA)^{k + 1})$. What is important is that this will give us a compute time independent of the size of the state space as long as we can show that $m$ can be set independently of S while meeting our target for the suboptimality of the induced policy.

This pseudocode sweeps under the rug on who creates the lists $C(s,a)$ and when? A simple and effective approach is to use "lazy evaluation" (or memoization): Create $C(s,a)$ at the first time it is needed (and do not create it otherwise).

### Good Action-Value Approximations Suffice

As a first step towards understanding the strength and weaknesses of this approach, let us define $\hat{T}:\mathbb{R}^{S\times \mathcal{A}}\rightarrow \mathbb{R}^{S\times \mathcal{A}}$ by

$$
(\hat{T} q)(s,a) = r_a(s) + \frac{\gamma}{m}\sum_{s'\in C(s,a)}\max_{a'\in \mathcal{A}}q(s',a').
$$

With the help of this definition, when called with state $s = s_0$ , the planner computes

$$
A = \arg \max_{a\in \mathcal{A}}(\hat{T}^{H}\mathbf{0})(s_0,a),
$$

Let us now turn to the question of whether the policy $\hat{\pi}$ induced by this planners is a good one. We start with a lemma that parallels our earlier result that bounded the suboptimality of a policy that is greedy w.r.t. a function over the states as a function of how well the function approximates the optimal value function. To state the lemma, we need the analog of optimal value functions but with action values.

To analyze the performance of this approach, a key lemma is needed to bound the suboptimality of a policy that is greedy with respect to an action-value function $q$. This requires defining the optimal action-value function, $q^\ast$.
 

Define
$$
q^\ast (s,a) = r_{a}(s) + \gamma \langle P_{a}(s),v^\ast \rangle .
$$

We call this function $q^\ast$ the optimal action- value function (in our MDP). The function $q^\ast$ is easily seen to satisfy $Mq^\ast = v^\ast$ and thus also $q^\ast = Tq^\ast$. 

The promised lemma is as follows:

**Lemma (Policy error bound - I.):** 

Let $\pi$ be a memoryless policy and choose a function $q:\mathcal{S}\times \mathcal{A}\rightarrow \mathbb{R}$ and $\epsilon \geq 0$. Then, the following hold:

1.  If $\pi$ is $\epsilon$-optimizing in the sense that $\sum_{a}\pi (a \mid s) q^\ast (s,a) \geq v^\ast (s) - \epsilon$ holds for every state $s\in \mathcal{S}$ then $\pi$ is $\epsilon /(1 - \gamma)$ suboptimal: $v^{\pi}\geq v^\ast - \frac{\epsilon}{1 - \gamma}\mathbf{1}$.

2.  If $\pi$ is greedy with respect to $q$ then $\pi$ is $2\epsilon$-optimizing with $\epsilon = \Vert q - q^\ast \Vert_{\infty}$ and thus
$$
v^{\pi}\geq v^\ast - \frac{2 \Vert q - q^\ast \Vert_{\infty}}{1 - \gamma}\mathbf{1}.
$$

**Suboptimality of almost $\epsilon$-optimizing policies**

The best we can hope for is that with each call, $Q_{H}(s_0,\cdot)$ is a good approximation to $q^\ast (s_0,\cdot)$ outside of some "failure event" $\mathcal{F}$ whose probability we will control separately. 

Let us say the probability of $\mathcal{F}$ is at most $\zeta$:

$$
\mathbb{P}_{s_0}(\mathcal{F})\leq \zeta
$$

We will choose $\mathcal{F}$ so that on $\mathcal{F}^c$, the complementer of $\mathcal{F}$ (a "good" event), it holds that
$$
\delta_{H} = \Vert Q_{H}(s_{0},\cdot) - q^{*}(s_{0},\cdot)\Vert_{\infty}\leq \epsilon . \tag{2}
$$
Then, on $\mathcal{F}^c$ the action $A$ returned by the planner is $2\epsilon$ optimizing at state $s_0$.

With probability at least $1 - \zeta$, $\hat{\pi}$ chooses $2\epsilon$-optimizing actions: The policy is almost $2\epsilon$-optimizing. The next lemma makes this precise:

**Lemma (Policy error bound II):** 

Let $\zeta \in [0,1]$, $\pi$ be a memoryless policy that selects $\epsilon$-optimizing actions with probability at least $1 - \zeta$ in each state. Then,

$$
v^{\pi} \geq v^\ast - \frac{\epsilon + 2 \zeta \Vert q^ast \Vert_{\infty}}{1 - \gamma} \mathbf{1}.
$$

### Error Control and Bounding

What remains is to show that with high probability, the error $\delta_{H}$, defined in (2) is small. 

Intuitively, $\hat{T} \approx T$. For any fixed $q \in \mathbb{R}^{S \times A}$ function over the state-action pairs such that $\Vert q \Vert_{\infty} \leq \frac{1}{1 - \gamma}$ and for any fixed $(s, a) \in \mathcal{S} \times \mathcal{A}$, by Eq. (1) and the choice of the sets $\mathcal{C}(s, a)$, with probability $1 - \zeta$,

$$
\vert \hat{T} q(s,a) - T q(s,a) \vert \leq \frac{\gamma}{1 - \gamma}\sqrt{\frac{\log\frac{2}{\zeta}}{2m}} := \Delta (\zeta ,m) \tag{3}
$$

A naive application of the union bound over all state-action pairs would reintroduce a dependency on the size of the state space $|S|$. The key to avoiding this dependence is to realize that the recursive planner only explores a limited, reachable subset of states.

For $h \geq 0$, define $\mathcal{S}_h = \lbrace s \in \mathcal{S} \mid \mathrm{dist}(s_0,s)\leq h \rbrace$ as the set of states accessible from $s_0$ by at most $h$ steps in the graph induced by the sampled transitions. 

In the calculation of $Q_H(s_0, \cdot)$, the recursion only calls states $s \in \mathcal S_{H - 1}$. The size of this set can be bounded by $\vert \mathcal S_{H - 1} \vert \leq (mA)^H$, which is independent of the size of the state space.

By recursively analyzing the error and applying the union bound only over the sequence of states actually visited by the planner (a sequence of length $n \le (mA)^H$), it can be shown that the error is controlled without depending on $|S|$.

Putting everything together, we get that for any $0\leq \zeta \leq 1$, the policy $\hat{\pi}$ induced by the planner is $\epsilon (m,H,\zeta)$-optimal with

$$
\epsilon (m,H,\zeta) := \frac{2}{(1 - \gamma)^2} \left[ \gamma^H +\frac{1}{1 - \gamma} \sqrt{\frac{\log\left(\frac{2nA}{\zeta}\right)}{2m}} +\zeta \right].
$$

To obtain a planner that induces a $\delta$-optimal policy, we can set $H$, $\zeta$, and $m$ appropriately. This leads to the final result:

**Theorem:** Assume that the immediate rewards belong to the $[0,1]$ interval. There is an online planner such that for any $\delta \geq 0$, in any discounted MDP with discount factor $\gamma$, the planner induces a $\delta$-optimal policy and uses at most $O((m^* \mathrm{A})^H)$ elementary arithmetic and logic operations per its calls, where $m^* (\delta ,\mathrm{A})$ and $H$ are chosen to meet the target error $\delta$.

Overall, we see that the runtime did increase compared to the deterministic case, but we managed to get a runtime that is independent of the cardinality of the state space. The exponential dependence on the effective horizon is, as we have seen, unavoidable in the worst case.

---

## 1.5 Notes on Simulator Access Models

The interaction between a planner and a simulator can be categorized by the access model provided.

* **Global Access**: The planner knows the full state space $S$ and can query the simulator for any $(s, a)$ pair. This is the most informative but least practical model for large $S$.
* **Local Access**: The planner can only query the simulator for states it has previously seen (either the initial state or states returned by the simulator). This requires the simulator to support "checkpointing" to reset to a previously visited state.
* **Online Access**: The simulator maintains an internal state. The planner can either reset this state to the initial one or provide an action to transition the internal state forward. This is the most restrictive and realistic model, as it mirrors an agent's actual interaction with an environment.

The algorithms discussed here can be adapted to these different models, with online access being the most general and challenging setting.

---

## 5.6 Further Notes and Connections

### Sparse lookahead trees

The idea of the algorithm that we analyzed comes from a paper by Kearns, Mansour and Ng from 2002. In their paper they consider the version of the algorithm which creates a fresh "new" random set $\mathcal{C}(s,a)$ in every recursive call. This makes it harder to see their algorithm as approximating the Bellman operator, but in effect, the two approaches are by and large the same. Much work has been devoted to improving these basic ideas and eventually these ideas led to various Monte-Carlo tree search algorithms, including UCT.

### Measure concentration

Hoeffding's inequality is a special case of what is known as measure concentration. This phrase refers to that the empirical measure induced by a sample is a good approximation to the whole measure. The price of being more stringent (i.e. reducing the failure probability $\zeta$) is mild, scaling with $\sqrt{\log(1 / \zeta)}$, which is known as a subgaussian deviation.

### A model-centered view and random operators

A key idea of this approach is that $\hat{T}$ is a good (random) approximation to $T$, hence, it can be used in place of $T$. One can also tell this story by saying that the data underlying $\hat{T}$ gives a random approximation to the MDP. The transition probabilities of this random approximating MDP would be defined using

$$
\hat{P} (s,a,s^{\prime}) = \frac{1}{m}\sum_{s^{\prime \prime}\in C(s,a)}\mathbb{I} \lbrace s^{\prime \prime} = s^{\prime} \rbrace
$$

A bigger point is that for a model to be a "good" approximation to the "true MDP", it suffices that the Bellman optimality operator that it induces is a "close" approximation to the Bellman optimality operator of the true MDP.

### Imperfect simulation model?

We can rarely expect simulators to be perfect. Luckily, if the simulator induced an MDP whose Bellman optimality operator is in a way close to the Bellman optimality operator of the true MDP, we expect the outcome of planning to be still a good policy in the true MDP. 

It can be shown that if $\hat{T}$ is a $\gamma$-max-norm contraction and $\hat{q}^\ast$ is its fixed point then

$$
\Vert \hat{q}^\ast - q^\ast \Vert_{\infty}\leq \frac{\Vert \hat{T}q^\ast - Tq^\ast \Vert_{\infty}}{1 - \gamma},
$$

which, combined with our first lemma on the policy error bound gives that the policy that is greedy with respect to $\hat{q}^\ast$ is $\dfrac{2 \left\Vert \hat{T}q^\ast - Tq^\ast \right\Vert_\infty}{(1 - \gamma)^{2}}$ optimal in the MDP underlying $T$.

### From local to online access

The algorithm analyzed here requires local access simulators. This is better than requiring global access, but worse than requiring online access. It remains an open question whether a similar result can be achieved using only online access.
