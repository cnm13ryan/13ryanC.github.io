---
date: "2025-07-09"
title: "(1.2) Optimality: Bellman Operators, Contraction Properties, and the Fundamental Theorem"
summary: "Optimality: Bellman Operators, Contraction Properties, and the Fundamental Theorem"
lastmod: "2025-07-09"
category: "Notes"
series: ["RL Theory"]
author: "Bryan Chan"
hero: /assets/images/hero3.png
image: /assets/images/card3.png
---

# 1. Optimality: Bellman Operators, Contraction Properties, and the Fundamental Theorem

This section builds upon the measure-theoretic definition of an MDP to establish the core principles of optimality. We will introduce the Bellman operators as the central tool for analyzing value functions, prove their contraction properties which guarantee the existence and uniqueness of solutions, and conclude with the fundamental theorem that connects optimal value functions to optimal policies.

---

### 1.1. Optimality Notions and Objective Functions

The objective in a discounted, infinite-horizon MDP is to find a policy that maximizes the expected discounted return from any starting state. This concept is formalized through optimal value functions.

**Definition 1.1.1 (Optimal Value Functions)**

The **optimal state-value function**, denoted $v^*: S \to \mathbb{R}$, yields the highest possible expected return achievable from any state $s \in S$. It is defined as the supremum over all possible policies $\pi \in \Pi$:

$$
v^\ast (s) := \sup_{\pi \in \Pi} v_\pi(s)
$$

Similarly, the **optimal action-value function**, $q^*: S \times A \to \mathbb{R}$, yields the highest possible expected return achievable by taking action $a$ in state $s$ and following an optimal policy thereafter:

$$
q^\ast (s, a) := \sup_{\pi \in \Pi} q_\pi(s, a)
$$


These two functions are intrinsically linked. The optimal value of a state is the value of the best action in that state. This relationship is given by:

$$
v^\ast (s) = \sup_{a \in A} q^\ast (s, a)
$$

The existence of a measurable policy that satisfies this condition is non-trivial and represents a cornerstone of the theory. The expression $\arg\max$ is not guaranteed to produce a measurable function from state $s$ to action $a$. We need to prove that a function $\mu: S \to A$ satisfying the greedy condition is itself measurable.

This is guaranteed by **measurable selection theorems**. 

One such result, the **Jankov-von Neumann Theorem**, states that if $S$ is a Standard Borel space, $A$ is a Polish space, and the function to be maximized (here, the $q$-value) is jointly measurable in $(s,a)$, then a measurable selector function $\mu(s)$ exists. 

This is why the Standard Borel assumption from the foundations is critical; without it, we could not guarantee the existence of a greedy policy.

> **Remark: Intuition for Measurable Selection**
> Why does the $\arg\max$ operator cause problems, and how do these theorems fix it?
>
> * **The Problem**: Imagine projecting the 3D graph of the function $q(s,a)$ down onto the $(s,a)$-plane. The $\arg\max$ for each $s$ corresponds to picking the 'highest' point(s) in the $a$ dimension. For a "wild" function, the set of these highest points can form a bizarre, non-measurable shape when projected onto the $a$-axis, making the selector function non-measurable.
> * **The Solution**: The proof for Standard Borel spaces leverages their topological structure. Because the action space $A$ is Polish, it has a countable basis of open sets. The core idea is to perform a search for the maximum over this countable basis. One can define a sequence of sets corresponding to actions that are $\epsilon$-optimal, for $\epsilon = 1/n$. The topological properties ensure these sets are measurable. By taking a countable intersection of these sets as $\epsilon \to 0$, we can "trap" a single maximizing action for each state $s$ in a way that preserves measurability. We are essentially building a measurable "sieve" that isolates a valid choice.
 
 

**Definition 1.1.2 (Optimal Policy)**

A policy $\pi^\ast$ is said to be **optimal** if its value function is equal to the optimal state-value function for all states. That is, for all $s \in S$:

$$
v_{\pi^\ast}(s) = v^\ast (s)
$$

A foundational result in dynamic programming is that for any discounted MDP defined on Standard Borel spaces, at least one such stationary optimal policy exists.

---

### 1.2. Bellman Operators: $T^\pi, T^*$

The recursive structure of the Bellman equations allows us to define operators that map the space of value functions onto itself. These operators are the primary object of analysis. We consider the space $B(S)$ of all bounded, measurable functions from the state space $S$ to $\mathbb{R}$.

**Definition 1.2.1 (Bellman Expectation Operator $T^\pi$)**

For any stationary policy $\pi$, the **Bellman expectation operator** $T^\pi: B(S) \to B(S)$ maps a candidate value function $v$ to its expected value after one step under policy $\pi$. Using the expected reward function $r(s,a)$ and transition kernel $p(ds'|s,a)$ derived from the unified kernel $\kappa$, it is defined as:

$$
(T^\pi v)(s) := \int_A \pi(da|s) \left( r(s,a) + \gamma \int_S v(s') p(ds'|s,a) \right)
$$

The state-value function $v_\pi$ is the **unique fixed point** of this operator, satisfying the Bellman expectation equation: $T^\pi v_\pi = v_\pi$.

---

**Definition 1.2.1a (Bellman Operators for Action-Value Functions)**

Analogous operators exist for the space of bounded, measurable action-value functions, $B(S \times A)$.

* **Expectation Operator $T_q^\pi$**:
    $$ (T_q^\pi q)(s, a) := r(s,a) + \gamma \int_S p(ds'|s,a) \int_A \pi(da'|s') q(s', a') $$
    The action-value function $q_\pi$ is the unique fixed point of this operator.

* **Optimality Operator $T_q^\ast$**:
    $$ (T_q^\ast q)(s, a) := r(s,a) + \gamma \int_S p(ds'|s,a) \sup_{a' \in A} q(s', a') $$
    The optimal action-value function $q^*$ is the unique fixed point of this operator.


**Definition 1.2.2 (Bellman Optimality Operator $T^*$**)

The **Bellman optimality operator** $T^*: B(S) \to B(S)$ maps a candidate value function $v$ to the best possible value achievable after one step. It is defined by taking the supremum over all actions:

$$
(T^\ast v)(s) := \sup_{a \in A} \left( r(s,a) + \gamma \int_S v(s') p(ds'|s,a) \right)
$$

The optimal state-value function $v^\ast$ is the **unique fixed point** of this operator. 

This identity is the celebrated **Bellman Optimality Equation**:

$$
v^\ast (s) = (T^\ast v^\ast)(s) = \sup_{a \in A} \left( r(s,a) + \gamma \int_S v^\ast (s') p(ds'|s,a) \right)
$$

This equation states that the value of a state under an optimal policy must equal the expected return for the best action from that state, followed by acting optimally thereafter.

---

### 1.3. Key Analytical Properties (Monotonicity, Contraction, Error Bounds)

To prove that the Bellman operators have unique fixed points, we analyze their properties on the **Banach space** $(B(S), \Vert \cdot \Vert_\infty)$, which is the space of bounded measurable functions on $S$ equipped with the supremum norm, $\Vert v \Vert_\infty := \sup_{s \in S} |v(s)|$.

**Property 1.3.1 (Monotonicity)**

Both operators $T^\pi$ and $T^*$ are **monotone**. For any two functions $v_1, v_2 \in B(S)$ such that $v_1(s) \le v_2(s)$ for all $s \in S$, it holds that:

* $(T^\pi v_1)(s) \le (T^\pi v_2)(s)$ for all $s \in S$.

* $(T^\ast v_1)(s) \le (T^\ast v_2)(s)$ for all $s \in S$.

This follows directly from the non-negativity of the probability measures $\pi(da|s)$ and $p(ds'|s,a)$.

**Property 1.3.2 ($\gamma$-Contraction)**

Both operators are **contraction mappings** with respect to the supremum norm for $\gamma \in [0, 1)$. For any $v_1, v_2 \in B(S)$:

$$
\Vert T^\ast v_1 - T^\ast v_2 \Vert_\infty \le \gamma \Vert v_1 - v_2 \Vert_\infty
$$

A similar inequality holds for $T^\pi$.

The proof for the action-value operators $T_q^\pi$ and $T_q^*$ is analogous and also yields a $\gamma$-contraction.
 

*Proof Sketch for $T^\ast$*:

Let $v_d(s) = v_1(s) - v_2(s)$. For any $s \in S$:

Let $a_1 \in A$ be an action that is $\epsilon$-optimal for $(T^\ast v_1)(s)$.

By definition, $(T^\ast v_1)(s) \le r(s, a_1) + \gamma \int_S v_1(s') p(ds' \mid s, a_1) + \epsilon$.

The optimality operator $T^\ast v_2$ applied at $s$ is the supremum over *all* actions, so its value must be greater than or equal to the value obtained by taking the specific action $a_1$.

Therefore, $(T^\ast v_2)(s) \ge r(s, a_1) + \gamma \int_S v_2(s') p(ds' \mid s, a_1)$.

Subtracting the second line from the first cancels the common reward term and yields the inequality:

$(T^\ast v_1)(s) - (T^\ast v_2)(s) \le \gamma \int_S (v_1(s') - v_2(s')) p(ds' \mid s,a_1) + \epsilon$

We can bound the difference $v_1 - v_2$ by its supremum norm:

$\le \gamma \int_S \Vert v_1 - v_2 \Vert_\infty \ p(ds' \mid s,a_1) + \epsilon = \gamma \Vert v_1 - v_2 \Vert_\infty + \epsilon$

Since this holds for any $\epsilon > 0$, we can take the limit as $\epsilon \to 0$. 

By swapping $v_1$ and $v_2$, we obtain the same bound for $|(T^\ast v_1)(s) - (T^\ast v_2)(s)|$. Taking the supremum over $s$ gives the result.

**The Banach Fixed-Point Theorem**

This theorem states that every contraction mapping on a complete metric space has a unique fixed point. Since $(B(S), \Vert \cdot \Vert_\infty)$ is a complete metric space (a Banach space) and both $T^\pi$ and $T^\ast$ are $\gamma$-contractions, we conclude:

1.  There exists a **unique** bounded measurable function $v_\pi$ such that $T^\pi v_\pi = v_\pi$.

2.  There exists a **unique** bounded measurable function $v^\ast$ such that $T^\ast v^\ast = v^\ast$.

The **completeness** of the Banach space $B(S)$ is not a mere technicality. 

The contraction property guarantees that the sequence of functions generated by Value Iteration, $v_{k+1} = T^\ast v_k$, is a **Cauchy sequence**. 

Completeness is the axiom that ensures every Cauchy sequence in the space converges to a limit point that is also *within that same space*. 

Without completeness, the sequence could converge to a limit function that is either unbounded or not measurable, rendering it useless as a value function. Completeness thus guarantees that $v^\ast$ exists and is a proper member of $B(S)$.

**Proof Sketch: Completeness of $B(S)$** 

Let $\lbrace v_n \rbrace_{n \in \mathbb N}$ be a Cauchy sequence in $(B(S), \Vert \cdot \Vert_\infty)$.

1.  **Pointwise Convergence**: For any fixed state $s \in S$, the sequence of real numbers $\{v_n(s)\}$ is a Cauchy sequence in $\mathbb{R}$ because $|v_n(s) - v_m(s)| \le \Vert v_n - v_m \Vert_\infty$. Since $\mathbb{R}$ is complete, this sequence converges to a limit. Let's define a function $v(s) := \lim_{n \to \infty} v_n(s)$. This defines the limit function for all $s$.

2.  **Uniform Convergence**: The convergence $v_n \to v$ is uniform. A Cauchy sequence implies that for any $\epsilon > 0$, there exists an $N$ such that for $m, n > N$, $\Vert v_n - v_m \Vert_\infty < \epsilon$. Taking the limit as $m \to \infty$, we get $\Vert v_n - v \Vert_\infty \le \epsilon$.

3.  **The Limit is in B(S)**: We must show $v$ is bounded and measurable.
    * **Boundedness**: Since the sequence converges, it is bounded. There exists some $M$ such that $\Vert v_n \Vert_\infty < M$ for all $n$. Thus $|v(s)| \le |v(s) - v_n(s)| + |v_n(s)| \le \epsilon + M$, so $v$ is bounded.
    * **Measurability**: The limit of a sequence of measurable functions is itself measurable. This is a standard result in measure theory.

Since the limit function $v$ is in $B(S)$, the space is complete.

Furthermore, iterating the operator from any starting function $v_0 \in B(S)$ will converge to this fixed point. This proves the convergence of **Value Iteration** ($v_{k+1} = T^\ast v_k$) to $v^\ast$. 

The convergence is geometric, with the error bound:

$$
\Vert v_k - v^\ast \Vert_\infty \le \frac{\gamma^k}{1-\gamma} \Vert v_1 - v_0 \Vert_\infty
$$

---

### 1.4. Greedy Policies and the Fundamental Theorem of Dynamic Programming

With the existence and uniqueness of $v^\ast$ established, we now connect it back to finding an optimal policy $\pi^\ast$.

**Definition 1.4.1 (Greedy Policy)**

A policy $\pi$ is said to be **greedy** with respect to a value function $v \in B(S)$ if it selects actions that maximize the one-step lookahead value. That is, for each state $s$, the policy $\pi(\cdot|s)$ assigns all of its probability mass to the set of actions that achieve the supremum in the Bellman optimality operator:

$$
\text{supp}(\pi(\cdot|s)) \subseteq \arg\sup_{a \in A} \left( r(s,a) + \gamma \int_S v(s') p(ds'|s,a) \right)
$$

By its definition, a greedy policy is inherently **memoryless (or stationary)**. 

The action-selection mechanism depends only on the current state $s$ and the time-independent value function $v$; it does not rely on the time step or past history. This property is crucial, as it confirms that the search for an optimal policy can be restricted to this simpler class of strategies.
 

The existence of a measurable deterministic policy $\mu: S \to A$ that satisfies this condition is guaranteed by **measurable selection theorems**, a result that depends critically on the state and action spaces being Standard Borel.

The need for a specific theorem here is subtle but fundamental. 

While the value function itself, defined by a supremum, $v(s) = \sup_{a \in A} q(s, a)$, is guaranteed to be measurable if $q$ is, the function that *selects* the maximizing action, $\mu(s) = \arg\sup_{a \in A} q(s, a)$, is **not**. 

The $\arg\max$ operator can produce non-measurable functions when the supremum is taken over an uncountable set (like a continuous action space). 

The measurable selection theorem guarantees the existence of at least one measurable function $\mu(s)$ whose output is always in the $\arg\max$ set. 

Without it, we could define an optimal value function $v^*$ but have no guarantee that a well-defined, measurable policy exists that can achieve it.

**The Fundamental Theorem of Dynamic Programming**

This theorem provides the central connection between optimal policies and the optimal value function. It has two parts:

1.  A policy $\pi$ is optimal **if and only if** it is greedy with respect to its own value function $v_\pi$. This means $T^\pi v_\pi = T^\ast v_\pi$, which implies that $v_\pi$ is a fixed point of $T^\ast$. Since $T^\ast$ has a unique fixed point $v^\ast$, it must be that $v_\pi = v^\ast$.

2.  Any policy $\pi^\ast$ that is greedy with respect to the optimal value function $v^\ast$ is an optimal policy.

*Proof Sketch for Part 1 ("only if" direction)*:

We prove by contradiction that if a policy $\pi$ is optimal ($v_\pi = v^*$), it must be greedy with respect to $v_\pi$. 

Assume $\pi$ is optimal but is *not* greedy at some state $s_0$. 

This means the action(s) chosen by $\pi(\cdot|s_0)$ are strictly suboptimal in the one-step lookahead:

$$
v_\pi(s_0) = (T^\pi v_\pi)(s_0) < (T^\ast v_\pi)(s_0)
$$

Let's define a new policy $\pi'$ which is identical to $\pi$ everywhere except at $s_0$. At $s_0$, $\pi'$ takes a greedy action $a' = \arg\sup_a q_\pi(s_0, a)$. 

For any state $s \ne s_0$, $v_{\pi'}(s) = v_\pi(s)$. But for state $s_0$:

$$ 
v_{\pi'}(s_0) = q_\pi(s_0, a') = \sup_a q_\pi(s_0, a) = (T^\ast v_\pi)(s_0) > v_\pi(s_0)
$$

Since $v_{\pi'}(s_0) > v_\pi(s_0)$ and $v_{\pi'}(s) \ge v_\pi(s)$ everywhere, the policy $\pi'$ is strictly better than $\pi$. 

This contradicts our initial assumption that $\pi$ was optimal. 

Therefore, an optimal policy must be greedy with respect to its own value function. 

Since $v_\pi = v^\ast$, it must be greedy with respect to $v^\ast$.

In essence, the entire problem of reinforcement learning is reduced to two steps: 

1. Find the unique solution to the Bellman Optimality Equation, $v^\ast$; 

2. Derive an optimal policy by simply acting greedily with respect to $v^\ast$.

---

### 1.5. Existence of an Optimal Stationary Policy

The previous sections provide all the components to prove the central existence theorem of dynamic programming.

**Theorem**: For any discounted MDP with Standard Borel state and action spaces and bounded rewards, there exists a stationary, deterministic policy $\pi^\ast$ that is optimal.

*Proof Sketch*:

1.  **Existence of Optimal Value Function**: 

From the **Banach Fixed-Point Theorem**, we know that the Bellman optimality operator $T^\ast$ is a $\gamma$-contraction on the complete metric space $B(S)$. 

Therefore, it has a unique fixed point, which we call the optimal value function, $v^\ast$.

2.  **Existence of a Greedy Policy**: 

We define the optimal action-value function $q^\ast (s, a) = r(s,a) + \gamma \int_S v^\ast (s') p(ds'|s,a)$. 

Because the spaces are Standard Borel and the component functions are measurable, $q^\ast$ is jointly measurable. 

By the **Jankov-von Neumann measurable selection theorem**, there exists a measurable function $\pi^\ast : S \to A$ such that $\pi^\ast (s) \in \arg\sup_{a \in A} q^\ast (s, a)$ for all $s \in S$. 

This $\pi^*$ is a deterministic, stationary, greedy policy.

3.  **Optimality of the Greedy Policy**: 

By definition, since $\pi^\ast$ is greedy with respect to $v^\ast$, it satisfies $(T^{\pi^\ast} v^\ast)(s) = (T^\ast v^\ast)(s)$ for all $s$. 

Since $v^\ast$ is the fixed point of $T^\ast$, we have $T^\ast v^\ast = v^\ast$. 

Therefore, $T^{\pi^\ast} v^\ast = v^\ast$. 

This equation shows that $v^\ast$ is a fixed point of the operator $T^{\pi^\ast}$. 

However, the Bellman expectation operator $T^{\pi^\ast}$ is also a $\gamma$-contraction and has a unique fixed point, which is by definition $v_{\pi^\ast}$. 

By uniqueness of the fixed point, we must have $v_{\pi^\ast} = v^\ast$. This proves the policy $\pi^\ast$ is optimal.



