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

#### 1.1.0 Objective, Notation, and Approximate Optimality

The primary goal in a discounted, infinite-horizon MDP is to find a policy that maximizes the expected discounted return. Let's establish the notation for this objective.

We use $\mathbb{E}_s^\pi$ to denote the expectation when starting in state $s$ and following policy $\pi$. The total discounted return, $R$, is the sum of discounted rewards over an infinite horizon:

$$R = \sum_{t = 0}^{\infty}\gamma^{t}r_{t}$$

where $r_t$ is the reward received at timestep $t$.

The **state-value function** for a policy $\pi$, denoted $v_\pi: S \to \mathbb{R}$, is the expected return from starting in state $s$ and subsequently following policy $\pi$:

$$v_\pi(s) = \mathbb{E}_s^{\pi}[R]$$

An **optimal policy**, $\pi^\ast$, is a policy that achieves the maximum possible value in every state. This maximum value is captured by the **optimal state-value function**, $v^\ast: S \to \mathbb{R}$, defined as:

$$
v^\ast (s) = \sup_{\pi} v_\pi(s), \quad \forall s \in S
$$

By definition, $v_\pi(s) \le v^\ast (s)$ for all states $s$ and any policy $\pi$. We use the shorthand $v_\pi \le v^\ast$ to express this relationship for all states. In general, for two functions $f, g$ on the same domain, $f \le g$ implies $f(z) \le g(z)$ for all elements $z$ in the domain.

In practice, finding a perfectly optimal policy may not be necessary or feasible. Instead, we often seek an **$\epsilon$-optimal policy**.

**Definition: $\epsilon$-Optimal Policy**

Let $\epsilon > 0$. A policy $\pi$ is said to be **$\epsilon$-optimal** if its value function is within $\epsilon$ of the optimal value function for all states. Using vector notation where $\mathbf{1}$ is a vector of ones, this is expressed as:

$$v_\pi \ge v^\ast - \epsilon \mathbf{1}$$

Finding an $\epsilon$-optimal policy is often a more tractable goal in reinforcement learning.


**Definition 1.1.1 (Optimal Value Functions)**

The **optimal state-value function**, denoted $v^\ast: S \to \mathbb{R}$, yields the highest possible expected return achievable from any state $s \in S$. It is defined as the supremum over all possible policies $\pi \in \Pi$:

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

The set of policies $\Pi$ here consists of all **probability kernels** from the state space to the action space, as formally defined in **Definition 2.2.0** of the formalism document.
 
The existence of a measurable policy that satisfies this condition is non-trivial and represents a cornerstone of the theory. The expression $\arg\max$ is not guaranteed to produce a measurable function from state $s$ to action $a$. We need to prove that a function $\mu: S \to A$ satisfying the greedy condition is itself measurable.

This is guaranteed by **measurable selection theorems**. 

One such result, the **Jankov-von Neumann Theorem**, states that if $S$ is a **Standard Borel space (Definition 1.2.4)**, $A$ is a **Polish space (Definition 1.2.3)**, and the function to be maximized (here, the $q$-value) is jointly measurable in $(s,a)$, then a measurable selector function $\mu(s)$ exists.

This is precisely why the Standard Borel assumption from the foundations is critical; without it, as explained in **Section 1.2.5 ("Why Standard Borel?")**, we could not guarantee the existence of a greedy policy.

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

### 1.2.0 The Linear Algebra Perspective: Classifying the Operators

> **Linear‑algebra backdrop.** All basic notions—vector spaces, bases, linear maps, kernel, image, and the rank–nullity theorem—are summarised once‑and‑for‑all in the *Formalism* document (§1.2.6 “Finite‑Dimensional Vector‑Space Primer”). Here we simply *use* those facts to analyse the Bellman and Markov operators. 

* **Application: The Markov Operator is Linear.**
* Definition 1.6.3 in *Formalism* shows that $(Pg)(x)=\int g(y)\kappa(x,dy)$ is linear because integration is linear; kernel, image, and rank arguments invoked later rely directly on the facts in §1.2.6.

**Affine and Non-Linear Operators**

Not all operators are linear. The Bellman operators fall into two other important categories.

* **Affine Transformation**: A transformation $A(v) = L(v) + w_0$, where $L$ is linear and $w_0$ is a fixed vector (a translation), is **affine**.
    * **Application: The Bellman Expectation Operator ($T^\pi$) is Affine.**
        The operator $T^\pi v = r_\pi + \gamma P_\pi v$ (in its finite-space form) or its integral equivalent consists of a linear part ($\gamma P_\pi v$) and a fixed translation ($r_\pi$). The presence of the non-zero reward function makes it **affine**, not linear.

* **Non-Linear Operator**: An operator that does not satisfy the additivity and homogeneity axioms is non-linear.
    * **Application: The Bellman Optimality Operator ($T^\ast$) is Non-Linear.**
        The operator $(T^\ast v)(s) := \sup_{a \in A} \{...\}$ contains a supremum (`sup`). This operation is **non-linear**, as $\sup(f+g)$ is not generally equal to $\sup(f) + \sup(g)$.

This classification is crucial: the analysis of the *linear* Markov operator is distinct from the analysis of the *affine* expectation operator and the *non-linear* optimality operator, whose properties must be established using different tools, such as the Contraction Mapping Theorem.

---

### 1.2. Bellman Operators: $T^\pi, T^\ast$

This function space, as discussed in the context of the **Markov Operator (Definition 1.6.3)**, forms a Banach space when equipped with the supremum norm. The Bellman operators act upon this space.

**Definition 1.2.1 (Bellman Expectation Operator $T^\pi$)**

For any stationary policy $\pi$, the **Bellman expectation operator** $T^\pi: B(S) \to B(S)$ maps a candidate value function $v$ to its expected value after one step under policy $\pi$. Using the expected reward function $r(s,a)$ and state transition kernel $p(ds'|s,a)$—derived from the unified kernel $\kappa$ as per **Section 2.2 of the formalism**—it is defined as:

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


**Definition 1.2.2 (Bellman Optimality Operator $T^\ast$**)

The **Bellman optimality operator** $T^\ast: B(S) \to B(S)$ maps a candidate value function $v$ to the best possible value achievable after one step. It is defined by taking the supremum over all actions:

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

#### 1.2.3 Special Case: Finite State-Space Formulation

While the integral form is general, it is instructive to see how the Bellman operators are expressed in the common case of a finite state space $S$ and action space $A$. Here, value functions can be represented as vectors in $\mathbb{R}^{|S|}$, and the operators become matrix-vector operations.

Let's define the components for a fixed policy $\pi$:

* The **expected reward vector**, $r_\pi \in \mathbb R^{|S|}$, is defined by 

$$
r_\pi = \sum_{s \in S} \biggl( \sum_{a \in A}  \pi (a \mid s) \hspace{0.1cm} r(s, a) \biggr)
$$

* The **policy transition matrix**, $P_\pi \in \mathbb R^{|S| \times |S|}$, is defined by 

$$
P_\pi = \sum_{(s, s') \in S \times S} \biggl( \sum_{a \in A} \pi (a \mid s) p (s' \mid s,a) \biggr)
$$


With this notation, the **Bellman expectation operator** ($T^\pi$ in the general case) can be written as a simple linear operation:

$$
T^\pi v = r_\pi + \gamma P_\pi v
$$

where $v \in \mathbb R^{|S|}$ is a value vector. This operator performs a one-step lookahead based on the policy $\pi$.

The **Bellman optimality operator** ($T^\ast$ in the general case, sometimes denoted simply as $T$) is defined with a maximization:

$$
(T^\ast v)(s) = \max_{a \in A} \left( r(s,a) + \gamma \sum_{s' \in S} p(s' \mid s,a) v(s') \right)
$$

This form makes the subsequent analysis of contraction properties more direct. 

#### 1.2.4 Spectral Decomposition: Jordan and Rational Canonical Forms

For certain analyses—e.g. *non‑diagonalizable* transition matrices, transient
mixing rates, or perturbation bounds—we need structural information beyond
eigenvalues alone.

> **Jordan Canonical Form (Hoffman–Kunze §7.5).**  
> Over an algebraically‑closed field $F$ (e.g. $\mathbb C$) every square
> matrix $A$ is similar to a block‑diagonal matrix  
> $J=\operatorname{diag}\bigl(J_{k_1}(\lambda_1),\dots,J_{k_m}(\lambda_m)\bigr)$
> where each block $J_{k}(\lambda)$ has $\lambda$ on the diagonal and ones
> on the super‑diagonal.  Generalized eigenvectors extend an ordinary
> eigen‑basis to a full basis of chains.  
> *Consequence.*  The $t$-step transition matrix satisfies  
> $A^{t}=PJP^{-1}= \sum_{j=0}^{k_{\max}-1} t^{j}N_j$,  
> where the $N_j$ are matrices depending only on $A$.  Thus
> $\Vert A^{t} - A_\infty \Vert_\infty$= $\tilde O(t^{k_{\max}-1}\rho^{t})$ whenever the
> spectral radius $\rho<1$.  This sharpens geometric convergence bounds.  
> (Used in §1.3.3 below.)

> **Rational Canonical (Frobenius) Form (Hoffman–Kunze §7.6).**  
> When the field is *not* algebraically closed—e.g. $\mathbb Q$ in exact
> arithmetic—the same invariants are captured by companion blocks derived from
> the invariant factors of $F[x]$-module $V$.  This guarantees that all
> power‑series arguments (resolvents, Green’s functions) remain valid without
> extending the field.

In practice: for a stochastic matrix $P_\pi$ whose minimal polynomial has a
repeated root $\lambda=1$, the leading Jordan block size quantifies *how many
moments* of the initial distribution influence asymptotic bias—important in
finite‑horizon regret bounds.

As stated in the **Banach Fixed-Point Theorem**, these operators are $\gamma$-contractions under the maximum norm, $\Vert \cdot \Vert_\infty$:

1.  $\Vert T^\pi u - T^\pi v \Vert_{\infty}\leq \gamma \Vert u - v \Vert_{\infty}$
2.  $\Vert T^\ast u - T^\ast v \Vert_{\infty} \leq \gamma \Vert u - v \Vert_{\infty}$

This guarantees that iterative application converges to the unique fixed point. For any initial value vector $u \in \mathbb{R}^{|S|}$:

* $\lim_{k\to \infty} (T^\pi)^k u = v_\pi$, where $v_\pi$ is the unique fixed point of $T^\pi$.

* $\lim_{k\to \infty} (T^\ast)^k u = v^\ast$, where $v^\ast$ is the unique fixed point of $T^\ast$.

---

### 1.3. Key Analytical Properties (Monotonicity, Contraction, Error Bounds)

To prove that the Bellman operators have unique fixed points, we analyze their properties on the **Banach space** $(B(S), \Vert \cdot \Vert_\infty)$, which is the space of bounded measurable functions on $S$ equipped with the supremum norm, $\Vert v \Vert_\infty := \sup_{s \in S} |v(s)|$.

**Property 1.3.1 (Monotonicity)**

Both operators $T^\pi$ and $T^\ast$ are **monotone**. For any two functions $v_1, v_2 \in B(S)$ such that $v_1(s) \le v_2(s)$ for all $s \in S$, it holds that:

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

This theorem states that every contraction mapping on a complete metric space has a unique fixed point. Since $(B(S), \Vert \cdot \Vert_\infty)$ is a complete metric space (a Banach space), and both $T^\pi$ and $T^\ast$ are $\gamma$-contractions, we conclude:

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


##### 1.3.3 Spectral‑Gap Refinement via Jordan Blocks

When $P_\pi$ is *diagonalizable*, the error decays exactly as $\rho(P_\pi)^k$. If not, the Jordan analysis of §1.2.4 yields the sharper bound  

$$
  \Vert P_\pi^{k} - \Pi \Vert_\infty \le Ck^{q-1}\rho^{k},
$$

where $q$ is the size of the largest Jordan block associated with the eigenvalue of maximal magnitude $\rho$.  This polynomial prefactor is often negligible in discounted RL because $k\asymp(1-\gamma)^{-1}$. See Hoffman–Kunze §7.5 for proof details.

##### 1.3.4 Bilinear Forms and Variance Bounds

Define the **Dirichlet form**
$\mathcal E_P(f)=\tfrac12\langle f,(I-P)f\rangle_\mu$ on
$\bigl(L^2(S,\mu),\langle\cdot,\cdot\rangle_\mu\bigr)$.
It is the quadratic form associated with the symmetric bilinear form
$B(f,g)=\langle f,(I-P)g\rangle_\mu$.  By Hoffman–Kunze §11.3 the rank of the
matrix representation of $B$ equals the dimension of the orthogonal
complement of constant functions—interpreted as the number of “mixing
directions.”  This underpins variance estimates for Monte‑Carlo value
estimates (see forthcoming §2.4).

---

### 1.4. Greedy Policies and the Fundamental Theorem of Dynamic Programming

With the existence and uniqueness of $v^\ast$ established, we now connect it back to finding an optimal policy $\pi^\ast$.

**Definition 1.4.1 (Greedy Policy)**

A policy $\pi$ is said to be **greedy** with respect to a value function $v \in B(S)$ if it selects actions that maximize the one-step lookahead value. That is, for each state $s$, the policy $\pi(\cdot|s)$ assigns all of its probability mass to the set of actions that achieve the supremum in the Bellman optimality operator:

$$
\text{supp}(\pi(\cdot|s)) \subseteq \arg\sup_{a \in A} \left( r(s,a) + \gamma \int_S v(s') p(ds'|s,a) \right)
$$

By its definition, a greedy policy is inherently **memoryless (or stationary)**. The action-selection mechanism depends only on the current state $s$ and the time-independent value function $v$; it does not rely on the time step or past history.

The existence of a measurable deterministic policy $\mu: S \to A$ that satisfies this condition is guaranteed by the **measurable selection theorems** discussed in **Section 1.1.1**, which rely on the Standard Borel structure of the state and action spaces. For the finite case, since we are maximizing over a finite set of actions, a maximizing action always exists.

A key property for finite MDPs is that greediness can be characterized by an equality of operators.

**Proposition (Characterizing Greediness in Finite MDPs)**

A memoryless policy $\pi$ is greedy with respect to a value function $v \in \mathbb{R}^{|S|}$ if and only if:

$$T^\pi v = T^\ast v$$

---

**The Fundamental Theorem of Dynamic Programming (Finite MDPs)**

This theorem provides the central connection between optimal policies and the optimal value function for finite MDPs.

**Theorem:** The following hold true in any finite MDP:
1.  Any policy $\pi$ that is greedy with respect to $v^\ast$ is optimal (i.e., $v_\pi = v^\ast$).
2.  The optimal value function is the unique fixed point of the Bellman optimality operator, satisfying the **Bellman Optimality Equation**: $v^\ast = T^\ast v^\ast$.

*Proof:*

The proof proceeds in two parts. First, we prove the theorem by only considering the set of memoryless policies, denoted ML. We define $\tilde{v}^\ast = \sup_{\pi \in \text{ML}} v_\pi$. We will show that a policy greedy with respect to $\tilde{v}^\ast$ is optimal within this class, and that $\tilde{v}^\ast = T^\ast \tilde{v}^\ast$. In the second part, we show that this is sufficient because $\tilde{v}^\ast = v^\ast$.

**Part 1: Optimality within Memoryless Policies**

The proof relies on first establishing that $\tilde{v}^\ast \le T^\ast \tilde{v}^\ast$. 

By definition, $v_\pi \le \tilde{v}^\ast$ for all $\pi \in \text{ML}$. Applying the monotone operator $T^\pi$ (as established in Property 1.3.1) to both sides gives $T^\pi v_\pi \le T^\pi \tilde{v}^\ast$. Since $v_\pi$ is the fixed point of $T^\pi$, this means $v_\pi \le T^\pi \tilde{v}^\ast$. Taking the supremum over all memoryless policies $\pi$ on both sides yields:
$$\tilde{v}^\ast = \sup_{\pi \in \text{ML}} v_\pi \le \sup_{\pi \in \text{ML}} (T^\pi \tilde{v}^\ast) = (T^\ast \tilde{v}^\ast) \quad (1)$$

Now, let $\pi$ be any policy that is greedy with respect to $\tilde{v}^\ast$. By the proposition above, this means $T^\pi \tilde{v}^\ast = T^\ast \tilde{v}^\ast$. Combining this with inequality (1), we get:
$$T^\pi \tilde{v}^\ast \ge \tilde{v}^\ast \quad (2)$$
Because $T^\pi$ is monotone, we can apply it repeatedly to both sides of inequality (2):
$$(T^\pi)^2 \tilde{v}^\ast \ge T^\pi \tilde{v}^\ast \ge \tilde{v}^\ast$$
Continuing this for $k$ steps, we have $(T^\pi)^k \tilde{v}^\ast \ge \tilde{v}^\ast$. As shown in Section 1.2.3, the sequence of functions generated by iterating the operator, $(T^\pi)^k \tilde{v}^\ast$, converges to the unique fixed point $v_\pi$. Taking the limit as $k \to \infty$, we get:
$$v_\pi \ge \tilde{v}^\ast$$
By definition, we also know $v_\pi \le \tilde{v}^\ast$, because $\tilde{v}^\ast$ is the supremum over all memoryless policies. Therefore, we must have $v_\pi = \tilde{v}^\ast$.

Finally, we show that $\tilde{v}^\ast$ satisfies the Bellman equation. Since $\pi$ is greedy with respect to $\tilde{v}^\ast$ and $v_\pi = \tilde{v}^\ast$:
$$T^\ast \tilde{v}^\ast = T^\pi \tilde{v}^\ast = T^\pi v_\pi = v_\pi = \tilde{v}^\ast$$
This establishes both parts of the theorem for the restricted class of memoryless policies.

**Part 2: Equivalence of Memoryless and General Policies**

It remains to be shown that $\tilde{v}^\ast = v^\ast$. The set of memoryless policies is a subset of all policies ($\text{ML} \subset \Pi$), so it is clear that $\tilde{v}^\ast \le v^\ast$. We must show $v^\ast \le \tilde{v}^\ast$.

This relies on a known result that for any policy $\pi$ (even non-stationary or history-dependent) and any starting state $s$, there exists a memoryless policy, let's call it $\pi_{ML}$, that induces the same discounted state-visitation distribution. Therefore, their value functions are identical: $v_\pi(s) = v_{\pi_{ML}}(s)$.

For any arbitrary policy $\pi \in \Pi$ and any state $s$:
$$v_\pi(s) = v_{\pi_{ML}}(s) \le \sup_{\pi' \in \text{ML}} v_{\pi'}(s) = \tilde{v}^\ast(s)$$
Since this holds for any policy $\pi$, we can take the supremum over all policies on the left side:
$$v^\ast(s) = \sup_{\pi \in \Pi} v_\pi(s) \le \tilde{v}^\ast(s)$$
As this holds for all states $s$, we have $v^\ast \le \tilde{v}^\ast$. Combined with $\tilde{v}^\ast \le v^\ast$, this proves $v^\ast = \tilde{v}^\ast$, completing the proof.

---

#### 1.4.2 Implications of the Fundamental Theorem

The Fundamental Theorem is powerful because it reduces the problem of finding an optimal policy—a search over a potentially vast space of functions—to the problem of solving the Bellman Optimality Equation.

* **Efficient Policy Calculation**: If we can find the optimal value function $v^\ast$, we can find an optimal policy simply by "greedifying" it. That is, for each state, we choose the action that maximizes the one-step lookahead:
    $$
    \pi^\ast(s) = \arg\max_{a \in A} \left( r(s,a) + \gamma \sum_{s' \in S} p(s'|s,a) v^\ast(s') \right)
    $$
    For a finite MDP, this greedy policy can be found in $O(|S|^2 |A|)$ time.

* **Dynamic Programming**: This reframing is the foundation of **dynamic programming** algorithms like Value Iteration and Policy Iteration. These methods provide a way to compute $v^\ast$ in time polynomial in $|S|$, $|A|$, and $1/(1-\gamma)$, which is vastly more efficient than the naive approach of enumerating all policies (the number of which can grow exponentially with $|S|$).

The fact that a stationary policy, which only depends on the current state, can be optimal over all history-dependent policies is a deep consequence of the **Markov property**. This property ensures that the future is conditionally independent of the past given the present state, allowing for this tremendous simplification.
 

---

### 1.5. Existence of an Optimal Stationary Policy

The previous sections provide all the components to prove the central existence theorem of dynamic programming.

**Theorem**: For any discounted MDP with Standard Borel state and action spaces and bounded rewards, there exists a stationary, deterministic policy $\pi^\ast$ that is optimal.

*Proof Sketch*:

1.  **Existence of Optimal Value Function**: 

From the **Banach Fixed-Point Theorem**, we know that the Bellman optimality operator $T^\ast$ is a $\gamma$-contraction on the complete metric space $B(S)$. 

As defined in the formalism **(e.g., Definitions 1.5.1 and 1.6.3)**, this is the space of bounded, measurable functions on which the system's operators act. The completeness of this space, proven in **Section 1.3**, guarantees that $T^\ast$ has a unique fixed point, which we call the optimal value function, $v^\ast$.

2.  **Existence of a Greedy Policy**: 

We define the optimal action-value function $q^\ast (s, a) = r(s,a) + \gamma \int_S v^\ast (s') p(ds'|s,a)$. 

Because the state and action spaces are **Standard Borel** and the component functions ($r$, $p$, $v^\ast$) are measurable, $q^\ast$ is jointly measurable.

By the **Jankov-von Neumann measurable selection theorem**, the requirement for Standard Borel spaces **(Definition 1.2.4)** and Polish spaces **(Definition 1.2.3)** ensures that there exists a measurable function $\pi^\ast : S \to A$ such that $\pi^\ast (s) \in \arg\sup_{a \in A} q^\ast (s, a)$ for all $s \in S$.
 

This $\pi^*$ is a deterministic, stationary, greedy policy.

3.  **Optimality of the Greedy Policy**: 

By definition of a greedy policy, since $\pi^\ast$ is greedy with respect to $v^\ast$, it satisfies $(T^{\pi^\ast} v^\ast)(s) = (T^\ast v^\ast)(s)$ for all $s$.
 

Since $v^\ast$ is the fixed point of $T^\ast$, we have $T^\ast v^\ast = v^\ast$. 

Therefore, $T^{\pi^\ast} v^\ast = v^\ast$. 

This equation shows that $v^\ast$ is a fixed point of the operator $T^{\pi^\ast}$. 

However, the Bellman expectation operator $T^{\pi^\ast}$ is also a $\gamma$-contraction and thus has its own unique fixed point, which is by definition $v_{\pi^\ast}$.

By uniqueness of the fixed point, we must have $v_{\pi^\ast} = v^\ast$. This proves the policy $\pi^\ast$ is optimal.

---

## Appendix A Linear‑Algebra Toolbox (Hoffman & Kunze Cheat‑Sheet)

| Concept | Formal statement | Typical use in RL |
|---------|------------------|-------------------|
| **Minimal vs. characteristic polynomial** | Unique monic polynomial of least degree annihilating $T$; divides the characteristic polynomial (H&K §6.3). | Detects when power‑series (e.g. Neumann) expansions terminate exactly. |
| **Jordan canonical form** | Similarity $A=PJP^{-1}$ where $J$ is block‑diagonal in Jordan blocks (H&K §7.5). | Transient dynamics, sensitivity analysis. |
| **Rational canonical form** | Block companion matrix determined by invariant factors (H&K §7.6). | Exact arithmetic over $\mathbb Q$. |
| **Spectral theorem** | Normal $T$ ⇒ orthonormal eigenbasis (H&K §9.4). | Analysis on $L^2$ with reversible chains. |
| **Sylvester’s law of inertia** | Signature of a quadratic form is invariant (H&K §12.2). | Proving norm equivalence & stability. |
| **Dual space $V^\*$** | $V^\*= \operatorname{Hom}_F(V,F)$ (H&K Ch. 13). | Interprets signed measures as elements dual to bounded functions. |

(For a quick reference, keep this table next to §1.2 when reading proofs.) 

