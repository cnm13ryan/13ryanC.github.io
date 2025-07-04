---
date: "2025-07-04"
title: "(Part 2.1) Dynamic Programming: Value Iteration"
summary: "Two classic planning algorithms for solving known MDPs. Present VI/ PI side by side; prove convergence rates, complexity"
category: "Tutorial"
series: ["RL Theory"]
author: "Bryan Chan"
hero: /assets/images/hero4.png
image: /assets/images/card4.png
---

### 0 Motivation for Value Iteration — Why start with a fixed‑point update?

Optimal sequential decision‑making in a finite Markov decision process (MDP) ultimately reduces to finding the unique fixed point of the **Bellman‑optimality operator**

$$
(Tv)(s)=\max_{a\in A}\Bigl[r_a(s)+\gamma\,P_a(s)^{\top}v\Bigr],\qquad s\in S.
\tag{0.1}
$$

Two facts make an **iterative fixed‑point scheme** especially attractive:

1. **Contraction & uniqueness.** $T$ is a $\gamma$-contraction on $(\mathbb R^{|S|},\lVert \cdot \rVert_\infty)$, hence it admits a *single* fixed point $v^*$ and simple Picard iterations $v_{k+1}=Tv_k$ converge geometrically .

2. **No inner linear solves.** Unlike policy‑iteration (which alternates greedy improvement with a *policy‑evaluation* linear solve) or the primal/dual LP formulations (which solve a global linear or quadratic programme), each Bellman update is a *local* maximisation followed by a sparse matrix–vector product, costing

$$
O \bigl(|S|^2|A|\bigr)\quad\text{arithmetic ops.}
\tag{0.2}
$$

   For table‑input MDPs this is within a logarithmic factor of the Chen–Wang lower bound for any planning algorithm .

---

#### 0.1 From Dynamic Programming to Value Iteration

The **Fundamental Theorem of Dynamic Programming** (next section) guarantees that a memory‑less deterministic policy $\pi^\*$ achieving $v^\*$ exists and satisfies the Bellman optimality equation.  Acting greedily with respect to $v^\*$ is therefore sufficient for optimality.

Value iteration provides the *simplest constructive proof*: starting from any $v_0\in\mathbb R^{|S|}$,

$$
\lVert v_k-v^\ast \rVert_\infty\le\gamma^k \lVert v_0-v^\ast \rVert_\infty,\tag{0.3}
$$

so an $\varepsilon$-accurate approximation is reached after

$$
k \ge H_{\gamma,\varepsilon} := \frac{\ln \bigl(1/(\varepsilon(1-\gamma))\bigr)}{1-\gamma}\tag{0.4}
$$

iterations, the **effective discount‑horizon** .

---

#### 0.2 Role in Approximate Planning

Stopping after $k=H_{\gamma,\varepsilon}$ updates and taking a *single* greedy step yields an $\varepsilon$-optimal policy with total runtime

$$
\tilde O \Bigl(\tfrac{|S|^2|A|}{1-\gamma}\ln\tfrac1\varepsilon\Bigr),\tag{0.5}
$$

establishing value iteration as the canonical baseline for sample‑free approximate planning .

---

#### 0.3 When is Value Iteration Preferable?

| Criterion              | Value Iteration        | Policy Iteration       | LP Methods           |
| ---------------------- | ---------------------- | ---------------------- | -------------------- |
| **Per‑iteration cost** | O(S²A)                 | O(S³) solve            | Depends on solver    |
| **Convergence rate**   | Geometric              | Typically quadratic    | One shot             |
| **Parallelism**        | High (state‑wise max) | Lower                  | Solver dependent     |
| **Memory**             | O(S)                   | O(S²)                  | Large                |

Hence value iteration is the *method of choice* when memory or per‑iteration compute dominate, or when only an approximate policy is needed quickly (e.g. as a roll‑out baseline or warm‑start for simulation‑based learning).

---

### 1 Fundamental Theorem of Dynamic Programming for Finite MDPs

#### 1.1 Formal set‑up and notation

Let

$$
M=\bigl(S,A,P,r,\gamma\bigr),\qquad 0\le\gamma<1,
$$

be a *finite*, $\gamma$-discounted Markov decision process (MDP) with

* $S=\{1,\dots,|S|\}$ finite state set;
* $A=\{1,\dots,|A|\}$ finite action set;
* $P_a(s,s')=\Pr\{s_{t+1}=s'\mid s_t=s,\;a_t=a\}$ transition matrix for each $a\in A$;
* $r_a(s)\in[0,1]$ bounded reward;
* $\gamma$ discount factor.

A **(stationary) policy** is a mapping $\pi:S\to\Delta(A)$ with $\pi(a\mid s)$ the probability of choosing $a$ in $s$.
Define the **state‑value function**

$$
v^\pi(s)=\mathbb E^\pi_s \Bigl[\sum_{t=0}^{\infty}\gamma^t r_{a_t}(s_t)\Bigr],
\tag{1.1}
$$

and the **state–action value**

$$
q^\pi(s,a)=r_a(s)+\gamma\sum_{s'}P_a(s,s')\,v^\pi(s').
\tag{1.2}
$$

For any policy $\pi$ let

$$
(T_\pi v)(s)=\sum_{a}\pi(a\!\mid\! s)\bigl[r_a(s)+\gamma P_a(s)^\top v\bigr],\qquad
(Tv)(s)=\max_a\bigl[r_a(s)+\gamma P_a(s)^\top v\bigr].
\tag{1.3}
$$

$T_\pi$ and $T$ are, respectively, the **policy** and **optimality Bellman operators**. Both are $\gamma$-contractions on $(\mathbb R^{|S|},\|\cdot\|_\infty)$.&#x20;

---

#### 1.2 Statement of the theorem

> **Theorem 1 (Fundamental Theorem of Dynamic Programming).**
> For every finite discounted MDP $M$:
>
> 1. There exists a deterministic stationary policy $\pi^\*$ such that
>
>    $$
>    v^{\pi^\*}=Tv^{\pi^\*}=:\,v^\*,\tag{1.4}
>    $$
>
>    and $v^\*\ge v^\pi$ for *all* policies $\pi$.
> 2. Any policy that is **greedy** with respect to $v^\*$,
>
>    $$
>      \pi_{\text{greedy}}(s)\in\arg\max_{a\in A}\bigl[r_a(s)+\gamma P_a(s)^\top v^\*\bigr],\tag{1.5}
>    $$
>
>    is optimal.
> 3. $v^\*$ is the **unique** fixed point of $T$.

*(cf. Szepesvári, Lec 2, Thm. “Fundamental Theorem”) *

---

#### 1.3 Proof sketch

1. **Restriction to memoryless policies.**
   For any (possibly history‑dependent) policy π and start‑state distribution μ, the *discounted occupancy measure*

   $$
   \nu^{\pi}_{\mu}(s,a)=\sum_{t=0}^{\infty}\gamma^{t}\Pr_\mu^\pi\{s_t=s,a_t=a\}
   \tag{1.6}
   $$

   satisfies $\langle\nu^{\pi}_{\mu},r\rangle=v^{\pi}_\mu$.
   Given $\nu^{\pi}_{\mu}$ define $\tilde\pi(a\mid s)=\nu^{\pi}_{\mu}(s,a)\big/\!\sum_{a'}\nu^{\pi}_{\mu}(s,a')$.
   Then $v^{\tilde\pi}_\mu=v^{\pi}_\mu$. Hence it suffices to consider **stationary** policies.&#x20;

2. **Existence and uniqueness of $v^\*$.**
   $T$ is a γ‑contraction; by the Banach fixed‑point theorem it admits a *unique* fixed point $v^\*$. Iterating $T$ from any $v_0$ converges geometrically:

   $$
   \|T^{k}v_0-v^\*\|_\infty\le\gamma^{k}\|v_0-v^\*\|_\infty.
   \tag{1.7}
   $$

3. **Optimality of a greedy policy.**
   Let $\pi_g$ satisfy (1.5). Then $T_{\pi_g}v^\*=Tv^\*=v^\*$. Since $T_{\pi_g}$ is a contraction with unique fixed point $v^{\pi_g}$, we have $v^{\pi_g}=v^\*$. Consequently $v^{\pi_g}\ge v^\pi$ for all π because $v^\*$ dominates all other value functions (monotonicity of $T$).

4. **Policy determinism.**
   Equation (1.5) permits tie‑breaking; choose an action deterministically at every state. Thus an *optimal deterministic* policy exists.

*Details of steps 1–4 appear in Lec 2, pp. 2–4.* 

---

#### 1.4 Interpretation and implications

* **Why it matters.** The theorem justifies *planning* via value functions: computing $v^\*$ alone suffices—its greedy policy is optimal.
* **Contraction view.** Because $T$ is a strict contraction, *value iteration* (Section 6) converges with predictable geometric rate; the same property ensures stable approximate methods.
* **Determinism is enough.** The optimal control law needs no memory and no randomness. This dramatically shrinks the search space from $|A|^{|S|}$ (deterministic) versus an uncountable set of history‑dependent stochastic policies, underpinning dynamic‑programming algorithms.
* **Foundation for error analysis.** Later sections (policy‑error bounds, approximate planning) build on (1.5): sub‑optimality is measured relative to the greedy step with respect to an approximate value function.

---

#### 1.5 Connections to upcoming sections

* The **Banach fixed‑point theorem** (Section 8) supplies the contraction result used in the proof.
* **Policy‑error bounds** (Section 3) quantify how close a greedy policy based on an approximate value is to $\pi^\*$.
* The **linear‑programming formulation** (Section 9) offers an alternative proof of existence/uniqueness of $v^\*$ via duality.
* The **geometry of value functions** (Section 7) visualises the set $\{v^\pi\}$ and shows $v^\*$ occupies a dominating vertex of a polytope.

---

### 2 Effective Horizon and the Single‑Step Bellman Update

In this section we formalise the idea that, for a discounted MDP with factor $\gamma\in[0,1)$, only the first
$H_{\gamma,\varepsilon}=\dfrac{\ln\!\bigl(1/(\varepsilon(1-\gamma))\bigr)}{1-\gamma}$
time‑steps are relevant when we care about an absolute error $\varepsilon$.
We begin by restating the Bellman update as a contraction, prove its fixed‑point error bound, and then derive the **effective horizon**.

---

#### 2.1 The Bellman (optimality) operator

For any $v\in\mathbb R^{|S|}$ define

$$
(Tv)(s)=\max_{a\in A}\Bigl[r_a(s)+\gamma\,P_a(s)^{\!\top}v\Bigr],\qquad s\in S. \tag{2.1}
$$

*Lemma 2.1 (γ‑contraction).* $T$ is a $\gamma$-contraction on $(\mathbb R^{|S|},\|\cdot\|_\infty)$:

$$
\|Tv-Tu\|_\infty\le\gamma\,\|v-u\|_\infty.\tag{2.2}
$$

*Proof.* For each state $s$ choose actions $a^\ast,b^\ast$ maximising the right– and left–hand sides respectively:

$$
(Tv)(s)-(Tu)(s)=r_{a^\ast}(s)-r_{b^\ast}(s)+\gamma P_{a^\ast}(s)^{\!\top}v-\gamma P_{b^\ast}(s)^{\!\top}u
\le\gamma P_{a^\ast}(s)^{\!\top}(v-u)\le\gamma\|v-u\|_\infty.
$$

Swap $v,u$ and take maxima to complete the inequality.

By Banach’s fixed‑point theorem (Appendix A, §8) the sequence $v_{k+1}=Tv_k$ converges **geometrically** to the unique fixed point $v^\*=Tv^\*$ proven in §1:

$$
\|v_k-v^\*\|_\infty\le\gamma^k\|v_0-v^\*\|_\infty.\tag{2.3}
$$

---

#### 2.2 Discounted tail mass and the effective horizon

Assume rewards are bounded: $0\le r_a(s)\le R_{\max}$ (w\.l.o.g. let $R_{\max}=1$).
Then for **any** policy $\pi$,

$$
0\;\le\;v^\pi(s)\;\le\;\frac{1}{1-\gamma},\qquad s\in S.\tag{2.4}
$$

Fix $v_0\equiv 0$.  Combine (2.3) with (2.4):

$$
\|v_k-v^\*\|_\infty\;\le\;\gamma^k\frac1{1-\gamma}.\tag{2.5}
$$

Set the right‑hand side to $\varepsilon$ and solve for $k$:

$$
k\;\ge\;H_{\gamma,\varepsilon}\;:=\;\frac{\ln\!\bigl(1/(\varepsilon(1-\gamma))\bigr)}{1-\gamma}. \tag{2.6}
$$

*Interpretation.* Because the **discounted tail mass**
$\sum_{t\ge k}\gamma^t=\gamma^k/(1-\gamma)$
equals the bound in (2.5), iterating for $H_{\gamma,\varepsilon}$ steps ensures that all rewards beyond that horizon contribute at most $\varepsilon$ to every state value.  Thus $H_{\gamma,\varepsilon}$ is the **effective horizon**.

---

#### 2.3 Stopping rule for Value Iteration

Algorithmically, start with $v_0=0$ and update $v_{k+1}=Tv_k$ until

$$
\|v_{k+1}-v_{k}\|_\infty\le\delta:=\frac{\varepsilon(1-\gamma)}{2\gamma}. \tag{2.7}
$$

Because $\|v_{k+1}-v_k\|_\infty\ge(1-\gamma)\|v_{k+1}-v^\*\|_\infty$ , the criterion guarantees (2.5) with margin $\varepsilon$.  At that point a single **greedification** step (§3) yields an $\varepsilon$-optimal policy.

---

#### 2.4 Finite‑horizon truncation equivalence

Define the finite‑horizon return

$$
v_H^\pi(s)=\mathbb E_s^\pi\!\Bigl[\sum_{t=0}^{H-1}\gamma^tr_{a_t}(s_t)\Bigr].\tag{2.8}
$$

*Lemma 2.2.* For every $\pi$ and $H\ge 0$,

$$
0\le v^\pi(s)-v_H^\pi(s)\le\frac{\gamma^{H}}{1-\gamma}.\tag{2.9}
$$

Hence choosing $H=H_{\gamma,\varepsilon}$ (2.6) ensures $\|v^\pi-v_H^\pi\|_\infty\le\varepsilon$.
Consequently, **discounted** planning to accuracy $\varepsilon$ is equivalent to **undiscounted finite‑horizon** planning with horizon $H_{\gamma,\varepsilon}$.

*Proof.* Tail of geometric series identical to derivation of (2.5).

---

#### 2.5 Summary & Practical implications

* **Geometric convergence** of value iteration: error shrinks by factor $\gamma$ per sweep.
* **Runtime** for $\varepsilon$-optimal values (table MDP, cost $O(S^2A)$ per sweep):

$$
T_{\text{VI}}(\varepsilon)=O\!\Bigl(S^2A\;\frac{\ln(1/(\varepsilon(1-\gamma)))}{1-\gamma}\Bigr).\tag{2.10}
$$

* The **effective horizon** $H_{\gamma,\varepsilon}$ provides a principled truncation for simulation or Monte‑Carlo rollouts, and underlies sample‑complexity bounds in model‑free RL.
* Near $\gamma\!\uparrow\!1$ the horizon explodes as $1/(1-\gamma)$; planning (and learning) accordingly becomes harder (§11).

---

**Next section (§3)**: *Policy‑error bound* – we quantify how far the greedy policy derived from an approximate value function can be from optimal and connect this to the stopping rule above.

---

\### References (section‑specific)
\[Szepesvári Lec 2] Proof of γ‑contraction and Banach argument 
\[Szepesvári Lec 3] Effective‑horizon derivation, tail bound, practical stopping rule 

---

### 3 Policy‑Error (Greedy) Bound

We now quantify how far the **greedy policy extracted from an *approximate* value function** can be from the truly optimal policy.
This result is crucial for turning value iteration into an ε‑optimal *planning* algorithm (§10) and for analysing approximate dynamic‑programming schemes.

---

#### 3.1 Definitions and notation

* **Greedy operator**

  $$
  \Gamma(v)(s)\;\in\;\arg\max_{a\in A}\Bigl[r_a(s)+\gamma P_a(s)^{\!\top}v\Bigr],\qquad s\in S.
  \tag{3.1}
  $$

  Any tie‑breaking rule is acceptable; the resulting policy may be deterministic or stochastic.

* **Policy error** for a candidate policy π:

  $$
  \Delta_\pi\;:=\;v^{\pi_\*}-v^{\pi}\;\in\;\mathbb R^{|S|},\qquad
  \|\Delta_\pi\|_\infty=\max_{s\in S}\bigl[v^{\pi_\*}(s)-v^{\pi}(s)\bigr].
  \tag{3.2}
  $$

* **Approximation error** of a value function v:

  $$
  \varepsilon\;:=\;\|v-v^\*\|_\infty.
  \tag{3.3}
  $$

Throughout this section rewards are bounded in $[0,1]$; hence $0\le v^\*\le (1-\gamma)^{-1}$ (§2).

---

#### 3.2 Theorem and proof 

> **Theorem 5 (Policy‑Error/Greedy bound).**
> Let v be any real‑valued function on S and let
> $\pi_g=\Gamma(v)$ be greedy w\.r.t. v.  Then
>
> $$
> v^{\pi_g}\;\ge\;v^\*-\frac{2\gamma}{1-\gamma}\,\|v-v^\*\|_\infty\;\mathbf 1.
> \tag{3.4}
> $$

*Proof.*
Define the residual vector

$$
\delta:=v^\*-v^{\pi_g}.
\tag{3.5}
$$

---

1. **Greediness relates Bellman operators**
   Using (3.1) and monotonicity of T and $T_{\pi_g}$ we have

   $$
   v^\*=Tv^\*\;\le\;T(v+\varepsilon\mathbf1)
          \;=\;T_{\pi_g}(v+\varepsilon\mathbf1)
          \;=\;T_{\pi_g}v+\gamma\varepsilon\mathbf1.
   \tag{3.6}
   $$

   Similarly $T_{\pi_g}v\le T_{\pi_g}(v^\*+\varepsilon\mathbf1)                          =T_{\pi_g}v^\*+\gamma\varepsilon\mathbf1$.
   Subtracting $T_{\pi_g}v^{\pi_g}=v^{\pi_g}$ from (3.6) yields

   $$
   \delta\;\le\;\gamma P_{\pi_g}\delta+2\gamma\varepsilon\mathbf1.
   \tag{3.7}
   $$

2. **Solve the linear inequality**
   Premultiply (3.7) by $(I-\gamma P_{\pi_g})^{-1} =\sum_{k\ge0}\gamma^k P_{\pi_g}^k$ (which is non‑negative):

   $$
   \delta\;\le\;2\gamma\varepsilon\sum_{k\ge0}\gamma^{k}P_{\pi_g}^{k}\mathbf1
          =  \frac{2\gamma\varepsilon}{1-\gamma}\,\mathbf1.
   \tag{3.8}
   $$

3. **Take the max‑norm**
   Equation (3.8) implies component‑wise dominance; hence

   $$
   \|\delta\|_\infty\;\le\;\frac{2\gamma}{1-\gamma}\,\varepsilon,
   $$

   which is exactly (3.4). ∎

---

#### 3.3 Tightness of the bound

*Singh–Yee (1994)* construct a 2‑state, 2‑action deterministic MDP where equality holds in (3.4) for every v satisfying $\|v-v^\*\|_\infty=\varepsilon$ (see lecture note example, p. 3) .
Thus the factor $2\gamma/(1-\gamma)$ is the *best possible* in the worst case.

---

\### 3.4 Consequences for approximate planning

1. **Stop‑and‑greedy rule**
   Combine (3.4) with the value‑error bound (2.5):
   if we iterate until

   $$
   \|v_k-v^\*\|_\infty\;\le\;\frac{\varepsilon(1-\gamma)}{2\gamma},
   \tag{3.9}
   $$

   then the greedy policy $\pi_k=\Gamma(v_k)$ is ε‑optimal:

   $$
   v^{\pi_k}\;\ge\;v^\*-\varepsilon\mathbf1.
   \tag{3.10}
   $$

2. **Iteration complexity**
   Using (2.6) with the tighter threshold in (3.9) gives

   $$
   k\;\ge\;\left\lceil\frac{\ln\!\bigl(\tfrac{2\gamma}{\varepsilon(1-\gamma)}\bigr)}
                           {1-\gamma}\right\rceil.
   \tag{3.11}
   $$

3. **Lipschitz continuity of the greedy operator**
   Equation (3.4) implies

   $$
   d\!\bigl(\Gamma(v),\Gamma(v')\bigr)
   :=\|v^{\Gamma(v)}-v^{\Gamma(v')}\|_\infty
   \;\le\;\frac{2\gamma}{1-\gamma}\,\|v-v'\|_\infty,
   \tag{3.12}
   $$

   i.e. the mapping $v\mapsto v^{\Gamma(v)}$ is *Lipschitz‑continuous* with constant $2\gamma/(1-\gamma)$. This property underlies many error‑propagation analyses in approximate policy iteration.

---

#### 3.5 Relative‑error variant

For tasks where only a *relative* accuracy $0<\delta_{\mathrm{rel}}<1$ is meaningful, require

$$
\|v_k-v^\*\|_\infty\;\le\;\frac{\delta_{\mathrm{rel}}(1-\gamma)}{2\gamma}\,\|v^\*\|_\infty.
\tag{3.13}
$$

Because $0\le v^\*\le(1-\gamma)^{-1}\mathbf1$, condition (3.13) is strictly weaker than (3.9) whenever $v^\*$ is small in magnitude (e.g. low‑reward problems). Iteration complexity becomes $O\!\bigl(\frac{\ln(1/\delta_{\mathrm{rel}})}{1-\gamma}\bigr)$.

---

#### 3.6 Interpretation

* **Error contraction for policies.** Even a crude value estimate propagates only *linear* error into the policy, tempered by $\frac{2\gamma}{1-\gamma}$.
* **High‑discount regimes.** As $\gamma\uparrow1$ the multiplicative factor explodes, explaining the empirical difficulty of planning with near‑undiscounted horizons unless value functions are extremely accurate.
* **Algorithm design.** Inequality (3.4) justifies the widespread “evaluate‑until‑near‑convergence then greedify” strategy and underlies sample‑complexity proofs for model‑free RL algorithms that alternate policy evaluation with behaviour improvement.

---

#### 3.7 Summary checklist

| Item                             | Formula | Source                     |
| -------------------------------- | ------- | -------------------------- |
| Greedy policy definition         | (3.1)   | §3.1                       |
| Policy‑error bound               | (3.4)   | Lec 3, p. 1                |
| Tight two‑state example          | —       | Lec 3, p. 3                |
| Stop‑and‑greedy threshold        | (3.9)   | Derived here               |
| Iteration count for ε‑optimality | (3.11)  | Combining (3.9) with (2.6) |

---

In the next section (§4) we dive deeper into the **fixed‑point interpretation** of the Bellman operators and formalise convergence via Banach’s theorem, linking the analytical bounds we have just derived with classic contraction‑mapping theory.

---

### 4 Fixed‑Point Iteration and Contraction Analysis

#### 4.1 Metric, contraction and Banach’s theorem

Let $(X,\lVert\cdot\rVert)$ be a normed linear space.
A mapping $F:X\to X$ is a **$c$-contraction** if there exists $0\le c<1$ such that

$$
\lVert F(x)-F(y)\rVert\le c\,\lVert x-y\rVert,\qquad \forall x,y\in X.\tag{4.1}
$$

> **Banach Fixed‑Point Theorem.**
> If $(X,\lVert\cdot\rVert)$ is *complete* and $F$ is a $c$-contraction, then
> (i) $F$ has a unique fixed point $x^\*\in X$ with $F(x^\*)=x^\*$;
> (ii) for any $x_0\in X$, the Picard iterates $x_{k+1}:=F(x_k)$ satisfy
>
> $$
> \lVert x_k-x^\*\rVert\le c^{\,k}\lVert x_0-x^\*\rVert,\qquad k\ge0.
> \tag{4.2}
> \] :contentReference[oaicite:0]{index=0}  
> $$

---

#### 4.2 Bellman operators as contractions

For a **fixed policy** $\pi$ the evaluation operator

$$
(T_\pi v)(s)=r_\pi(s)+\gamma P_\pi(s)^{\!\top}v,\qquad s\in S
\tag{4.3}
$$

is linear.  Because each row of $P_\pi$ is stochastic, for the max‑norm

$$
\lVert T_\pi u-T_\pi v\rVert_\infty
     =\gamma\max_{s}\lvert P_\pi(s)^{\!\top}(u-v)\rvert
     \le\gamma\lVert u-v\rVert_\infty.\tag{4.4}
$$

Thus $T_\pi$ is a $\gamma$-contraction on $(\mathbb R^{|S|},\lVert\cdot\rVert_\infty)$.
Identical reasoning (with a maximisation over $a$) shows the **optimality operator**

$$
(Tv)(s)=\max_{a\in A}\!\bigl[r_a(s)+\gamma P_a(s)^{\!\top}v\bigr]\tag{4.5}
$$

is also a $\gamma$-contraction.

---

#### 4.3 Existence & uniqueness of value functions

*Policy evaluation.* Applying Banach to $(T_\pi,\lVert\cdot\rVert_\infty)$ yields:

> **Proposition 4.1 (Unique value of a policy).**
> For every stationary policy $\pi$ there exists a **unique** $v^\pi\in\mathbb R^{|S|}$ satisfying $T_\pi v^\pi=v^\pi$; moreover $\lVert T_\pi^{\,k}u-v^\pi\rVert_\infty\le\gamma^{k}\lVert u-v^\pi\rVert_\infty$ for any initial $u$.&#x20;

*Optimal value.*  The same argument with $T$ (Eq. 4.5) proves uniqueness of the optimal value $v^\*$.

---

#### 4.4 Geometric rate and stopping rules

With rewards bounded in $[0,1]$ we have $\lVert v^\*\rVert_\infty\le(1-\gamma)^{-1}$.
Starting Picard iteration from $v_0\equiv0$,

$$
\lVert v_k-v^\*\rVert_\infty\le\gamma^{k}/(1-\gamma).\tag{4.6}
$$

Hence $v_k$ is $\varepsilon$-accurate once

$$
k\;\ge\;H_{\gamma,\varepsilon}:=\frac{\ln\!\bigl(1/(\varepsilon(1-\gamma))\bigr)}{1-\gamma}\tag{4.7}
$$

—the *effective horizon* already derived in § 2.

**Residual‑based test.**
Because $\lVert v_{k+1}-v_k\rVert_\infty\ge(1-\gamma)\lVert v_{k+1}-v^\*\rVert_\infty$ ,
stopping when

$$
\lVert v_{k+1}-v_k\rVert_\infty\le\frac{\varepsilon(1-\gamma)}{2\gamma}\tag{4.8}
$$

implies the greedy policy from $v_{k+1}$ is ε‑optimal (cf. § 3).

---

#### 4.5 Fixed‑point view of policy evaluation

Writing $A_\pi:=I-\gamma P_\pi$ (nonsingular since $\rho(\gamma P_\pi)<1$), we have

$$
v^\pi=A_\pi^{-1}r_\pi=\sum_{t=0}^{\infty}\gamma^{t}P_\pi^{t}r_\pi.\tag{4.9}
$$

Equation (4.9) is precisely the Neumann series for $(I-\gamma P_\pi)^{-1}$ and gives **discounted occupancy measures** as weights—linking fixed‑point iteration to the linear‑algebraic solution used in § 9.&#x20;

---

#### 4.6 Variations and practical refinements

| Variant                         | Iteration rule                 | Convergence constant                | Notes                                                                   |
| ------------------------------- | ------------------------------ | ----------------------------------- | ----------------------------------------------------------------------- |
| **Synchronous VI**              | $v_{k+1}=Tv_k$                 | $\gamma$                            | Eq. (4.6)                                                               |
| **Policy evaluation (exact)**   | $v_{k+1}=T_\pi v_k$            | $\gamma$                            | For fixed $\pi$                                                         |
| **Gauss–Seidel / Asynchronous** | update one state at a time     | $\gamma$ (same)                     | Requires state‑wise sweep ordering; proof uses non‑expansiveness of $T$ |
| **Successive over‑relaxation**  | $v_{k+1}=v_k+\omega(Tv_k-v_k)$ | $\gamma(1-\omega)$ optimum $\omega$ | Accelerates when $\gamma$ close to 1                                    |

All share the same fixed‑point $v^\*$; tuning $\omega$ (or using multi‑grid ideas) improves the multiplicative constant but **cannot beat the $\gamma^{k}$ factor**, a fundamental limitation of contraction mappings.&#x20;

---

#### 4.7 Interpretation and links

* **Banach is the backbone** of dynamic programming: every Bellman‑style algorithm is a disguise for Picard iteration in a properly chosen metric.
* The **effective horizon** $H_{\gamma,\varepsilon}$ emerges naturally from the contraction factor—tying analytic error to computational effort.
* Fixed‑point theory justifies *function‑approximation* schemes (e.g., TD(0)) whenever the projected operator remains a contraction—a topic returned to in § 10 when discussing approximate planning.

---

### 5 Finite‑Horizon Interpretation of Value Iteration

Although our objective is the infinite‑horizon discounted return, value iteration can be viewed as repeatedly solving **finite‑horizon** control problems whose length $H$ grows as the algorithm proceeds. This perspective clarifies why stopping after $H_{\gamma,\varepsilon}$ sweeps suffices and links discounted planning to its undiscounted, finite‑horizon counterpart.

---

#### 5.1 Finite‑Horizon Truncated Return

For any policy $\pi$ and integer horizon $H\ge 1$ define the truncated value

$$
v_{H}^{\pi}(s)\;:=\;\mathbb E^{\pi}\!\Bigl[\sum_{t=0}^{H-1}\gamma^{t} r_{a_t}(s_t)\;\Bigm|\;s_0=s\Bigr].
\tag{5.1}
$$

Because rewards satisfy $0\le r_a(s)\le 1$, the **tail mass** beyond step $H$ is bounded:

$$
0\;\le\;v^{\pi}(s)-v_{H}^{\pi}(s)\;\le\;\frac{\gamma^{H}}{1-\gamma},\qquad s\in S.
\tag{5.2}
$$

*Proof.*  The residual series is a geometric sum:
$\sum_{t=H}^{\infty}\gamma^{t} \le \gamma^{H}/(1-\gamma)$.&#x20;

---

#### 5.2 Effective Horizon and $\varepsilon$-Accuracy

Set $H=H_{\gamma,\varepsilon}$ (cf. § 2, Eq. (2.6)):

$$
H_{\gamma,\varepsilon}\;=\;\frac{\ln\!\bigl(1/(\varepsilon(1-\gamma))\bigr)}{1-\gamma}.
\tag{5.3}
$$

Then (5.2) implies

$$
\|v^{\pi}-v_{H_{\gamma,\varepsilon}}^{\pi}\|_{\infty}\;\le\;\varepsilon.
\tag{5.4}
$$

Hence *any* planning or learning algorithm that is exact for horizon $H_{\gamma,\varepsilon}$ is automatically $\varepsilon$-optimal for the infinite‑horizon discounted problem.

---

#### 5.3 Value Iteration as Successive Finite‑Horizon Solutions

Observe that one Bellman update prepends a **single discounted step** to the horizon‑$H$ solution:

$$
v_{H+1}^{\pi}(s)=r_{\pi}(s)+\gamma\,P_{\pi}(s)^{\!\top}v_{H}^{\pi}.
\tag{5.5}
$$

Applying the max over actions gives

$$
v_{H+1}^{\max}=T v_{H}^{\max},\qquad v_{0}^{\max}\equiv 0,
\tag{5.6}
$$

where $v_{H}^{\max}:=\max_{\pi} v_{H}^{\pi}$.  Thus **value iteration with $k$ sweeps computes the optimal $k$-step truncated value**; sweeping to $H_{\gamma,\varepsilon}$ delivers (5.4).

---

#### 5.4 Algorithmic Consequences

*Stopping rule.*  Combining (5.4) with the greedy policy error bound (3.4) gives the *stop‑and‑greedy* criterion used in § 3:

$$
\|v_k-v_{k-1}\|_\infty\;\le\;\frac{\varepsilon(1-\gamma)}{2\gamma}\;\Longrightarrow\;v^{\Gamma(v_k)}\ge v^\ast-\varepsilon\mathbf1.
\tag{5.7}
$$

*Runtime.*  Each sweep costs $O(S^2A)$ operations; using $k=H_{\gamma,\varepsilon}$ yields

$$
T_{\text{VI}}(\varepsilon)=O\!\Bigl(S^{2}A\,\frac{\ln\!\bigl(1/(\varepsilon(1-\gamma))\bigr)}{1-\gamma}\Bigr),
\tag{5.8}
$$

matching Eq. (2.10).

*Monte‑Carlo roll‑outs.*  In simulation‑based evaluation one may truncate trajectories after $H_{\gamma,\varepsilon}$ steps with bias ≤ $\varepsilon$, reducing variance without sacrificing correctness.&#x20;

---

#### 5.5 Relation to Undiscounted Finite‑Horizon MDPs

Define an *augmented* MDP $\tilde M_H$ whose state space is $S\times\{0,\dots,H\}$ and which deterministically transitions $(s,h)\mapsto(s',h+1)$ with zero discount ($\gamma=1$) until an absorbing layer $h=H$.  Then for any policy $\pi$:

$$
v_{H}^{\pi}(s)\;=\; \tilde v^{\pi}\bigl((s,0)\bigr),
\tag{5.9}
$$

so discounted planning to accuracy $\varepsilon$ is *equivalent* to undiscounted finite‑horizon planning with $H=H_{\gamma,\varepsilon}$.

---

#### 5.6 Summary

* Finite‑horizon error (5.2) links horizon length to discounting.
* The **effective horizon** $H_{\gamma,\varepsilon}$ (5.3) governs both sample and computation complexity.
* Value iteration can be interpreted as solving a sequence of optimal control problems with horizons $1,2,\dots$.
* Stopping after $H_{\gamma,\varepsilon}$ sweeps plus one greedy step guarantees an $\varepsilon$-optimal policy with runtime (5.8).

Section 6 will leverage these insights to present the standard **Value‑Iteration Algorithm** with rigorous convergence and stopping criteria.

---

### 6 Algorithmic Description of **Value Iteration**

#### 6.1 Pseudocode

| **Algorithm 1 — Value Iteration**                                    |                                                                                       |
| -------------------------------------------------------------------- | ------------------------------------------------------------------------------------- |
| **Input:** finite MDP $M=(S,A,P,r,\gamma)$; accuracy $\varepsilon>0$ |                                                                                       |
| 1                                                                    | $v\gets 0$                   ▷ initialise pessimistically                             |
| 2                                                                    | **repeat**                                                                            |
| 3                                                                    |  $v'\;\gets\;Tv$             ▷ Bellman update (2.1)                                   |
| 4                                                                    |  $\text{res}\;\gets\;\lVert v'-v\rVert_\infty$                                        |
| 5                                                                    |  $v\;\gets\;v'$                                                                       |
| 6                                                                    | **until** $\text{res}\le\frac{\varepsilon(1-\gamma)}{2\gamma}$  ▷ residual test (4.8) |
| 7                                                                    | **return** $\pi_{\text{greedy}}=\Gamma(v)$     ▷ extract ε‑optimal policy             |

*Comments* 

* Line 3 performs one **synchronous** Bellman sweep; on a table MDP this costs $O(S^2A)$ arithmetic operations because each state–action backup touches every next‑state probability.
* The stopping criterion on line 6 is the “stop‑and‑greedy” threshold derived from the policy‑error bound (3.9).
* Line 7 greedifies once; this is $O(SA)$. A deterministic optimal policy exists by the Fundamental Theorem (§1).

---

#### 6.2 Convergence Guarantee

> **Theorem 6 (ε‑Optimality & Complexity of Algorithm 1).**
> Assume $0\le r_a(s)\le 1$.  Let
>
> $$
> K(\varepsilon)=\Bigl\lceil \tfrac{\ln\!\bigl(\frac{2\gamma}{\varepsilon(1-\gamma)}\bigr)}{1-\gamma}\Bigr\rceil.
> \tag{6.1}
> $$
>
> After at most $K(\varepsilon)$ iterations of the loop (lines 2‑6), Algorithm 1 returns a policy $\pi$ satisfying
>
> $$
> v^{\pi}\;\ge\;v^\*-\,\varepsilon\mathbf1.
> \tag{6.2}
> $$
>
> The total arithmetic complexity is
>
> $$
> T_{\text{VI}}(\varepsilon)=\tilde O\!\Bigl(S^{2}A\,\tfrac{\ln(1/\varepsilon)}{1-\gamma}\Bigr).
> \tag{6.3}
> $$

*Proof.* 

1. **Value error.** With $v_0\!=\!0$, inequality (2.5) gives
   $\|v_k-v^\*\|_\infty\le\gamma^k/(1-\gamma)$.
2. **Residual bound.** Residual $\text{res}_k=\|v_{k}-v_{k-1}\|_\infty$ satisfies
   $\text{res}_k\ge(1-\gamma)\|v_k-v^\*\|_\infty$ .
3. **Stop criterion.** If $\text{res}_k\le\varepsilon(1-\gamma)/(2\gamma)$ then
   $\|v_k-v^\*\|_\infty\le \varepsilon/(2\gamma)$.
4. **Policy error.** By the greedy bound (3.4) we obtain (6.2).
5. **Iterations.** Setting $\gamma^{k}/(1-\gamma)=\varepsilon/(2\gamma)$ yields (6.1).
6. **Runtime.** Each sweep costs $O(S^{2}A)$; multiply by $K(\varepsilon)$. ∎

---

#### 6.3 Stopping Rules in Practice

Two equivalent tests are widely used:

* **Residual test** (line 6):                      $\|v'-v\|_\infty\le\delta$ with $\delta=\frac{\varepsilon(1-\gamma)}{2\gamma}$.
* **Bellman error test** (uses current value only):
  stop when $\|Tv-v\|_\infty\le\varepsilon(1-\gamma)/(2)$.
  (Because $\|Tv-v\|_\infty\le\text{res}\le \|Tv-v\|_\infty/(1-\gamma)$ .)

The first is cheaper; the second avoids an extra sweep.

---

#### 6.4 Memory Requirements

Algorithm 1 stores only two value vectors ($v$ and $v'$) and a policy vector for output—$O(S)$ memory. This is minimal among exact planners; policy iteration needs $O(S^2)$ for linear solves.

---

#### 6.5 Variants and Accelerations

| Variant                        | Update rule                                      | Effect on (6.1)                                                                                                         |
| ------------------------------ | ------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------- |
| **Gauss–Seidel**               | update states sequentially using freshest values | same bound, often lower constant                                                                                        |
| **Asynchronous VI**            | update subset of states per sweep                | still $\gamma$-contraction; need longer wall‑time but useful when $P$ sparse                                            |
| **Successive Over‑Relaxation** | $v\leftarrow v+\omega(Tv-v)$, $0<\omega<1$       | effective $\gamma'\!=\!(1-\omega)+\omega\gamma$ ⇒ optimised $\omega$ speeds practical convergence when $\gamma\approx1$ |

No variant can asymptotically beat the geometric rate $\gamma^{k}$ implied by Banach’s theorem (§4).

---

#### 6.6 Take‑aways

* **Simplicity.** One-line Bellman backup; minimal memory footprint.
* **Predictable accuracy.** Formula (6.1) ties iteration count to $\varepsilon$ and $\gamma$.
* **Baseline for planners.** Runtime is within a logarithmic factor of the Chen–Wang lower bound (see §11–12).
* **Limitation.** For $\gamma\uparrow1$ effective horizon $H_{\gamma,\varepsilon}=Θ\bigl(\frac{1}{1-\gamma}\bigr)$ makes VI slow; policy‑iteration variants can mitigate this.

Section 7 will leave the algorithmic realm and explore the **geometry** of value functions—showing how Dadashi et al. (2019) interpret value iteration as straight lines inside a high‑dimensional polytope.

---

### 7 Geometry of Value Functions — the Value‑Function Polytope

We now leave algorithmics and turn to the *shape* of the set of value functions that can be realised by stationary policies.
The key reference is Dadashi et al. (2019), who prove that this set is a (potentially non‑convex and self‑intersecting) polytope and uncover a series of striking structural properties. We summarise and formalise their results, relate them to the objects defined in §§1‑6, and explain how they illuminate the behaviour of value‑iteration–type algorithms.

#### 7.1 Policy simplex and the value‑function map

Recall the mapping

$$
f_V:\;\pi \;\mapsto\; V^{\pi}\;=\;(I-\gamma P_{\pi})^{-1}r_{\pi}\quad\text{from Definition 1}.  
$$

* **Domain.** The policy space $\mathcal{P} \doteq \bigl(\Delta(A)\bigr)^{|S|}$ is the Cartesian product of |S| simplices; it is a compact, convex polytope of dimension $d_{\mathcal P}=|S|(|A|-1)$.
* **Image.** We denote by

$$
\mathcal V \;=\;f_V(\mathcal P)=\{V^{\pi}\mid \pi\in\mathcal P\}\subset\mathbb R^{|S|}
$$

the **value‑function polytope**.

Dadashi et al. show that $f_V$ is smooth and its image $\mathcal V$ is connected and compact (Lemma 2 in the paper) .

---

#### 7.2 The Value‑Function Polytope Theorem

> **Theorem 7.1 (Dadashi et al., 2019).**
> $\mathcal V$ is a *general polytope*: a finite union of convex polytopes, each of dimension at most $|S|$. It may be non‑convex and may self‑intersect.&#x20;

*Proof sketch.*
Partition $\mathcal P$ by which action maximises the Bellman backup in each state; each region is the interior of a simpler polytope on which $f_V$ is affine because $r_\pi$ and $P_\pi$ depend linearly on $\pi$. The image of a convex polytope under an affine map is a convex polytope, hence $\mathcal V$ is a finite union thereof. For small MDPs with duplicated actions these images overlap, producing self‑intersections (cf. Fig. 2 in Dadashi *et al.*).&#x20;

*Visual intuition.* The *diagram on page 1* of Dadashi et al. plots $\mathcal V$ for several 2‑state MDPs; note the “chevron” shape and the fold where two faces intersect.

---

#### 7.3 Line theorem

Fix a *base policy* $\pi$ and a *single state* $s$.
Let

$$
\mathcal P_{\pi}^{(s)}=\{\pi' \in \mathcal P : \pi'( \cdot | s')=\pi(\cdot|s')\;\forall s'\neq s\}
$$

be the set of policies that may differ from $\pi$ **only in state $s$**.

> **Theorem 7.2 (Line Theorem).** $f_V\bigl(\mathcal P_{\pi}^{(s)}\bigr)$ is a *line segment*
> with endpoints at two **$s$-deterministic** policies $\pi^{\text{min}},\pi^{\text{max}}\in\mathcal P_{\pi}^{(s)}$.
> Moreover the segment is *monotone*: either $V^{\pi^{\text{min}}}\le V^{\pi^{\text{max}}}$ or the reverse, component‑wise.&#x20;

*Proof idea.* Restricting $\pi$ on all but one state fixes every column of $(I-\gamma P_\pi)^{-1}$ except that for s, so $V^{\pi}$ varies affinely in a 1‑D subspace (Lemma 3 in Dadashi *et al.*). The extreme points correspond to pure actions in state s, yielding determinism; monotonicity follows from stochastic dominance of occupancy measures.&#x20;

*Consequence.* Line segments provide a *foliation* of $\mathcal V$ by “policy fibres”; value‑iteration (which updates all states synchronously) moves by *parallel translation* across successive fibres (§7.6).

---

#### 7.4 Faces and semi‑deterministic policies

For $d\in\{0,\ldots,|S|\}$ define

$$
\mathcal D_d=\{\pi\in\mathcal P: \text{π is deterministic on at least }d\text{ states}\}.
$$

> **Proposition 7.3 (Dadashi et al.).** A value vector lies on a face of codimension $d$ of $\mathcal V$ *iff* it is generated by some $\pi\in\mathcal D_d$.&#x20;

*Interpretation.* Becoming deterministic in one more state moves the policy to a lower‑dimensional face of the polytope. Deterministic (optimal or not) policies correspond to *vertices*.

---

#### 7.5 Higher‑dimensional sub‑polytopes

For $S_0\subseteq S$ let $\mathcal P_{\pi}^{(S_0)}$ be the set of policies that match π outside $S_0$. Reusing Lemma 3 in Dadashi et al.:

> **Theorem 7.4.** $f_V\!\bigl(\mathcal P_{\pi}^{(S_0)}\bigr)$ is a polytope of dimension ≤|S₀| obtained by the Cartesian product of |S₀| monotone line segments (one per free state).&#x20;

Hence $\mathcal V$ is built recursively from 1‑D fibres.

---

#### 7.6 Geometry‑aware view of Value Iteration

Consider the kth iterate $V_k=T V_{k-1}$ (Section 2). For any state s, the update replaces the s‑th component by the *upper endpoint* of the line segment associated with $\mathcal P_{π_k}^{(s)}$; simultaneously over all states this is equivalent to *projecting orthogonally* onto the product of these |S| upper endpoints.

The effective horizon (§2) bounds how far along each line segment these projections continue before entering an ε‑tube around $V^\*$.

*Measure‑theoretic reading.* Each line corresponds to varying the *state‑action occupancy measure* only at time 0 in state s, holding later behaviour fixed; value iteration greedily re‑allocates that zero‑step mass to the action maximising long‑run return.

---

#### 7.7 Practical implications

| Observation                                                                   | Algorithmic takeaway                                                                                                             |
| ----------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------- |
| **Monotone lines** ⇒ any 1‑state policy change yields predictable value shift | Safe policy‑improvement heuristics can accept updates that stay on the line if the shift is positive                             |
| **Faces ↔ semi‑determinism**                                                  | Progressive *determinisation* (PI) walks down a chain of faces, explaining its empirical speed                                   |
| **Self‑intersections** create *bottlenecks*                                   | Gradient methods may stagnate unless exploration noise (entropy, natural gradient) moves them onto another sheet of the polytope |

---

#### 7.8 Connections to earlier sections

* Sections 2–4 analysed convergence via contraction; Section 7 provides a **geometric counterpart**: each Bellman update moves to an extreme point of every fibre simultaneously.
* The policy‑error bound (§3) can be visualised as a *tubular neighbourhood* of radius $\frac{2\gamma}{1-\gamma}\|V-V^\*\|_\infty$ around the optimal vertex.
* Linear‑programming view (§9) corresponds to optimising a linear functional over $\mathcal V$; strong duality holds because $\mathcal V$ is (piece‑wise) polyhedral.

---

#### 7.9 Summary

The Dadashi‑polytope perspective reveals a rich compositional geometry:

* **Global:** $\mathcal V$ is a finite polytope whose vertices are deterministic policies.
* **Local:** fixing all but k states carves out a k‑dimensional sub‑polytope; k = 1 gives a *monotone line*.
* **Algorithmic:** dynamic‑programming updates correspond to travelling along, or jumping between, these fibres.

In subsequent sections we will return to analysis (Banach, LP formulations) armed with this geometric intuition.

---

#### References for this section

Dadashi, R., Ali Taïga, A., Le Roux, N., Schuurmans, D., & Bellemare, M. G. **“The Value Function Polytope in Reinforcement Learning.”** ICML 2019.&#x20;

Szepesvári, C. *RL Theory – Lectures 2 & 3.* 2020.

---



---

### 8 Banach’s Fixed‑Point (The Contraction‑Mapping) Theorem — Background Box

#### 8.1 Classical statement

> **Theorem 8.1 (Banach, 1922).**
> Let $(X,d)$ be a **complete metric space** and $F:X\!\to\!X$ a **$c$-contraction**, i.e.
>
> $$
> d\bigl(F(x),F(y)\bigr)\;\le\;c\,d(x,y),\qquad\forall\,x,y\in X,\quad 0<c<1. \tag{8.1}
> $$
>
> Then
>
> 1. $F$ admits a **unique fixed point** $x^{\*}\in X$ with $F(x^{\*})=x^{\*}$;
> 2. For every initial $x_0\in X$ the Picard iterates $x_{k+1}:=F(x_k)$ converge **geometrically**:
>
>    $$
>    d(x_k,x^{\*})\;\le\;c^{\,k}\,d(x_0,x^{\*}). \tag{8.2}
>    $$

A standard proof proceeds by showing the Picard sequence is Cauchy (using (8.1)) and invoking completeness of $X$ to ensure convergence; uniqueness follows from (8.1) with $x,y=x^{\*}$.  Detailed steps appear in Appendix A.1 of Szepesvári’s notes .

---

#### 8.2 Application to Markov decision processes

Set

$$
X=\bigl(\mathbb R^{|S|},\|\cdot\|_{\infty}\bigr),\qquad 
F=T\quad\text{or}\quad F=T_\pi .
$$

*Completeness* of $(\mathbb R^{|S|},\|\cdot\|_\infty)$ is immediate;
*contraction* (8.1) holds with $c=\gamma$ for both $T$ and every $T_\pi$ (Lemma “γ‑contraction of Bellman operators” in Lec 2, p. 4) .
Therefore:

* **Unique fixed points**

  * $T$ ⇒ $v^{\*}$ (optimal value)
  * $T_\pi$ ⇒ $v^{\pi}$ (policy value)
* **Geometric error decay** (cf. Eq. (4.6)):

  $$
  \|T^{k}v_0-v^{\*}\|_\infty\le\gamma^{k}\|v_0-v^{\*}\|_\infty,\tag{8.3}
  $$

  and analogously for $T_\pi$.

---

#### 8.3 Quantitative corollaries

* **Effective horizon.**
  Rearranging (8.3) with rewards in $[0,1]$ gives $k\ge H_{\gamma,\varepsilon}$ (2.6) to achieve $\varepsilon$-accuracy.
* **Residual test.**
  For $F=T$,

  $$
  \|v_{k+1}-v_k\|_\infty \;\ge\; (1-\gamma)\|v_{k+1}-v^{\*}\|_\infty, \tag{8.4}
  $$

  yielding the practical stopping rule (4.8).
* **Lipschitz composition.**
  If $G$ is $L$-Lipschitz, $G\!\circ\!F$ is $cL$-Lipschitz; e.g. the greedy‑evaluation mapping $v\mapsto v^{\Gamma(v)}$ has constant $2\gamma/(1-\gamma)$ (3.12).

---

#### 8.4 Generalisations and remarks

1. **Non‑expansive operators.**
   Banach’s guarantee fails when $c=1$; e.g. asynchronous value iteration relies on partial contractions but still converges due to monotonicity (Bertsekas 1994).
2. **Weighted sup‑norms.**
   If certain states matter less, choose weights $w(s)$ and norm $\|x\|_{w,\infty}=\max_{s}|x(s)|/w(s)$; $T$ remains a $cw$-contraction with appropriately scaled $c$.
3. **Stochastic approximation.**
   In temporal‑difference learning the *expected* update is a contraction; stochastic iterates track the fixed point under diminishing step sizes, yielding root‑mean‑square‑error bounds.
4. **Banach vs. Blackwell.**
   For undiscounted finite‑horizon problems contractions vanish ($c=1$), but backward induction exploits *finite nesting* rather than contraction.

---

#### 8.5 Role within this note

* Groundwork for § 4’s fixed‑point iteration, § 6’s convergence proof, and § 11’s runtime.
* Justifies treating planning as solving **one** nonlinear fixed‑point equation instead of $|S|\times|A|$ linear inequalities.
* Provides the mathematical lens through which we interpret approximate dynamic‑programming algorithms: ensure projected or sample‑based operators retain a contraction (possibly in expectation).

---

### 9 Linear‑Programming Interpretation of Planning in MDPs

Linear programming (LP) offers an alternative—and historically important—view of exact planning.  Instead of searching directly for a policy or iterating Bellman updates, we solve for the *value vector* or, dually, the *discounted occupancy measure* that satisfies the optimality conditions.  This section develops:

* the **primal LP** whose feasible set is the epigraph of the Bellman‑optimality operator;
* the **dual LP** whose variables are occupancy measures;
* proofs of **strong duality** and **complementary slackness**;
* the correspondence between LP solutions, Bellman fixed points and deterministic optimal policies;
* computational remarks and ties to preceding sections.

Throughout, let the MDP be $M=(S,A,P,r,\gamma)$ with $0<\gamma<1$ and rewards $r_a(s)\in[0,1]$.  We write $d_0\in\Delta(S)$ for an initial state distribution, assumed fully supported to avoid degeneracy.

---

#### 9.1 Primal formulation – “minimise the start‑state value”

The **primal LP** uses the value vector $v\in\mathbb R^{|S|}$ as decision variable:

$$
\begin{aligned}
\text{minimise} & \quad d_0^\top v \\[2pt]
\text{subject to} & \quad v \;\;\ge\;\; T v. \tag{9.1}
\end{aligned}
$$

*Interpretation.* The constraints $v\ge Tv$ enforce *Bellman dominance*: every feasible $v$ lies **above** the maximal one‑step look‑ahead.  Among all such “upper envelopes”, the objective picks the smallest in the direction of $d_0$.  Because the feasible region is a closed, bounded polyhedron, linear‑programming theory guarantees an optimal basic feasible solution.  Proposition 1 below shows that this solution is exactly $v^\*$.

> **Proposition 9.1 (Optimal value solves the primal LP).**
> If $d_0(s)>0\;\forall s$, the unique optimal solution of (9.1) is $v^\*=Tv^\*$.

*Proof.*  (i) Feasibility: $v^\*=Tv^\*$ satisfies the constraint, so $v^\*$ is in the feasible region.
(ii) Minimality: any feasible $v$ dominates $Tv$ component‑wise; by monotonicity $v\ge T^{k}v$ for every $k$.  Letting $k\to\infty$ (Banach) gives $v\ge v^\*$.  Since $d_0$ is strictly positive, $d_0^\top v>d_0^\top v^\*$ unless $v=v^\*$.∎&#x20;

*Connection to §4.* The primal LP constraints encode the same contraction fixed‑point as Banach’s theorem but in *inequality* form; the LP approach thus generalises to undiscounted and average‑cost criteria where contraction may fail.

---

#### 9.2 Dual formulation – occupancy measures as decision variables

Define an *occupancy measure* $d\in\mathbb R^{|S||A|}$ with components $d(s,a)$ intended to represent the discounted frequency of visiting $(s,a)$ starting from $d_0$.  The **dual LP** is

$$
\begin{aligned}
\text{maximise} & \quad d^\top r \\[2pt]
\text{subject to} & \quad (I-\gamma P^\top)d \;=\;(1-\gamma)d_0, \\[2pt]
                  & \quad d\ge 0, \tag{9.2}
\end{aligned}
$$

where $P$ stacks the $|S|\times|S|$ transition matrices $P_a$ row‑wise.  The linear equations are the **Bellman flow (balance) constraints**: expected inflow equals discounted outflow plus initial mass.&#x20;

---

#### 9.3 Feasible dual variables coincide with policies

Given a feasible $d$ for (9.2) define

$$
\pi_d(a\mid s):=\frac{d(s,a)}{\sum_{b}d(s,b)}\quad\text{if }\sum_{b}d(s,b)>0,\tag{9.3}
$$

with arbitrary tie‑breaking otherwise.  Expanding the balance equation shows $\sum_ad(s,a)=(1-\gamma)\sum_{t=0}^\infty\gamma^t\Pr^\pi\{s_t=s\}$; hence $d$ is exactly the discounted **state–action occupancy measure** of $\pi_d$ under $d_0$.&#x20;

---

#### 9.4 Strong duality and complementary slackness

The primal is feasible and bounded (Prop 9.1) and the dual is feasible (take $d=0$), so **strong duality** holds: optimal values coincide and both attain their optima.  Complementary slackness reads

$$
d^\*(s,a)\bigl[ v^\*(s)- r_a(s)-\gamma P_a(s)^\top v^\*\bigr]=0,\qquad\forall(s,a).\tag{9.4}
$$

Because the bracket is zero exactly for greedy actions, any optimal dual solution places all mass on greedy actions; thus **every optimal dual yields a deterministic optimal policy**.  Conversely, the occupancy measure of any optimal deterministic policy satisfies (9.4).  This recovers the Fundamental Theorem (§1) from LP duality.

---

#### 9.5 Links to fixed‑point iteration and geometry (§7)

| Perspective   | Object        | Optimality condition                             |
| ------------- | ------------- | ------------------------------------------------ |
| Banach (§4)   | value $v$     | $v=Tv$ (equality)                                |
| Primal LP     | value $v$     | $v\ge Tv$ + minimality                           |
| Dual LP       | occupancy $d$ | balance equations (9.2)                          |
| Polytope (§7) | vertex $v$    | $v$ is a dominating vertex; deterministic policy |

The LP view embeds the value‑function **polytope** inside $\mathbb R^{|S|}$ as the feasible region $\{v\ge Tv\}$.  Its vertices correspond to deterministic policies; the primal objective selects the unique *top* vertex $v^\*$.  Dadashi’s geometry thus has an immediate linear‑programming analogue.

---

#### 9.6 Complexity remarks

* Size: primal has $|S|$ vars and $|S|\,|A|$ constraints; dual has $|S|\,|A|$ vars and $|S|$ equality constraints.
* Generic interior‑point solvers run in $O\bigl((|S||A|)^{3}\bigr)$ time⁠—comparable to Gaussian elimination on $(I-\gamma P_\pi)$ but usually more costly than policy iteration’s $O(|S|^3)$ per sweep when $|A|$ is moderate.
* **Approximate LP (ALP)** methods truncate constraints to obtain ε‑optimal policies with polynomial sampling complexity; see de Farias & Van Roy (2003) for bounds.&#x20;
* Sparse or factored MDPs admit structured LPs amenable to decomposition.

---

#### 9.7 Summary

* Planning can be cast as a pair of primal‑dual linear programs.
* The primal seeks the **smallest** value vector dominating its Bellman backup—guaranteed unique.
* The dual optimises expected reward over the polytope of **occupancy measures**; optimal dual variables correspond bijectively to deterministic optimal policies (complementary slackness).
* Strong duality offers a third proof of the Fundamental Theorem and links to the geometric picture of Section 7.
* While exact LP solvers are rarely competitive for large tabular MDPs, approximate and structured LPs underpin many modern RL algorithms (e.g., ALP, dual ascent, Lagrangian actor–critic).

Next, in **Section 10** we leverage the policy‑error bound (§3) and effective horizon (§2) to analyse **value iteration as an approximate planner**, culminating in explicit ε‑runtime guarantees.

---

### 10 Value Iteration as an *Approximate* Planning Algorithm

With the convergence and policy‑error machinery from §§2–4 we can now quantify how many Bellman sweeps are *sufficient*—but **not more than necessary**—to obtain an ε‑optimal policy, and what computational price this entails.

#### 10.1 Stop‑and‑Greedy meta‑algorithm

> **Algorithm 2 — VI‑Stop‑n‑Greedy**
>
> 1. **Input:** accuracy target ε ∈ (0,1), discount γ < 1, rewards in \[0,1].
> 2. Initialise $v_0 \leftarrow 0$.
> 3. **For** $k = 0,1,2,\dots$**:**
>       $v_{k+1} \leftarrow T v_k$                        ▷ Bellman sweep
>       **if** $ \lVert v_{k+1}-v_k\rVert_\infty \le \dfrac{\varepsilon(1-\gamma)}{2\gamma}$ **then**
>           **return** $\pi_{\text{greedy}} = \Gamma(v_{k+1})$.

Lines 2–3 are standard value iteration; line 3b implements the **residual test** derived in §4.4 (Eq. 4.8). Once the residual is small enough we “greedify once” (§3) and terminate.

---

#### 10.2 Correctness guarantee

> **Theorem 10.1 (ε‑Optimality).**
> Let $k^\* =\Bigl\lceil \dfrac{\ln\!\bigl(\tfrac{2\gamma}{\varepsilon(1-\gamma)}\bigr)}{1-\gamma}\Bigr\rceil$.
> After at most $k^\*$ sweeps Algorithm 2 outputs a deterministic stationary policy $\pi$ such that
>
> $$
> v^{\pi}\;\ge\;v^\ast-\varepsilon\,\mathbf 1.
> \tag{10.1}
> $$

*Proof.*
*Value error.* With $v_0=0$ we have $\lVert v_k-v^\*\rVert_\infty\le\gamma^k/(1-\gamma)$ (Eq. 2.5).
*Residual link.*  $\lVert v_{k+1}-v_k\rVert_\infty\ge(1-\gamma)\lVert v_{k+1}-v^\*\rVert_\infty$ .
*Stop rule.*  The residual threshold therefore implies $\lVert v_{k+1}-v^\*\rVert_\infty\le\varepsilon/(2\gamma)$.
*Policy error.*  Applying the greedy bound (Eq. 3.4) yields (10.1).
*Iteration bound.*  Solving $\gamma^{k}/(1-\gamma)=\varepsilon/(2\gamma)$ for $k$ gives $k^\*$. ∎

---

#### 10.3 Arithmetic complexity

Each synchronous sweep computes all $S A$ Q‑values and |S| maximisations, costing

$$
\Theta\!\bigl(S^2A\bigr)
$$

table operations (matrix‑vector product). Hence

$$
T_{\text{VI‑ε}} = \Theta\!\Bigl(S^2A\;\frac{\ln\!\bigl(\tfrac{2\gamma}{\varepsilon(1-\gamma)}\bigr)}{1-\gamma}\Bigr).
\tag{10.2}
$$

Up to the harmless log‑factor $\ln\frac1{1-\gamma}$, this matches the *lower bound* $\Omega(S^2A)$ of Chen & Wang (2017) for table‑input MDPs .

---

#### 10.4 δ–ε trade‑offs and effective horizon revisited

* Absolute accuracy ε translates into
  $k^\*=H_{\gamma,\varepsilon}=\Theta\!\bigl(\frac{\ln(1/\varepsilon)}{1-\gamma}\bigr)$ (cf. Eq. 2.6).
* **Relative** error $0<\delta_{\text{rel}}<1$ is achieved by replacing ε with $\delta_{\text{rel}}(1-\gamma)$ (because $\|v^\*\|_\infty\le(1-\gamma)^{-1}$), leading to
  $k=O\!\bigl(\frac{\ln(1/\delta_{\text{rel}})}{1-\gamma}\bigr)$.

The linear $1/(1-\gamma)$ dependence reflects the **effective planning horizon**: near‑undiscounted tasks ($\gamma\to1$) are intrinsically hard unless stronger structure is available.

---

#### 10.5 Refined stopping criteria

In practice one rarely knows γ exactly, and residuals can oscillate. Two robust alternatives:

| Criterion         | Condition                                                                                | Guarantees                                   |
| ----------------- | ---------------------------------------------------------------------------------------- | -------------------------------------------- |
| **Bellman error** | stop when $\|Tv_k-v_k\|_\infty \le \tfrac{\varepsilon(1-\gamma)}{2}$                     | implies ε‑optimality (uses extra sweep)      |
| **Span seminorm** | maintain upper & lower envelopes and stop when $\text{span}_\infty(v_k) \le \varepsilon$ | policy‑independent bound; useful for large γ |

See Lec 3, pp. 2–3 for derivations .

---

#### 10.6 When is VI‑Stop‑n‑Greedy preferable?

* **Memory budget:** only two |S|-vectors are stored.
* **Parallelism:** state backups are embarrassingly parallel.
* **Anytime property:** $v_k$ itself is an upper bound on $v^\*$.
* **Provably near‑optimal runtime:** within log‑factor of table‑input lower bound.

However, near‑undiscounted domains ($\gamma\!\approx\!1$) or very low ε may favour *policy iteration* (quadratic local rate) or LP‑based methods (one‑shot solve); §11 compares these in detail.

---

#### 10.7 Summary

Value iteration, coupled with a single greedy step, converts *value* convergence into *policy* optimality with a clear ε‑runtime:

$$
\boxed{\text{Sweeps} = \Theta\!\Bigl(\frac{\ln(1/\varepsilon)}{1-\gamma}\Bigr)},\qquad
\boxed{\text{Work} = \Theta\!\Bigl(S^2A\frac{\ln(1/\varepsilon)}{1-\gamma}\Bigr)}.
$$

These bounds are tight up to logarithmic terms and form the benchmark against which all other *exact‑model* planning algorithms are measured.

---

*References for this section*
Szepesvári, **RL Theory Lecture 3** – policy error bound, runtime, lower bound&#x20;
Jiang, **MDP Preliminaries** – LP duality view and occupancy measures&#x20;


---

### 11 Runtime of Approximate Planning with Value Iteration

We now sharpen the computational picture by separating **per‑iteration cost** from **iteration count**, comparing value iteration with competing dynamic‑programming algorithms, and stating **lower bounds** that show our ε‑runtime is essentially optimal for table MDPs.

---

#### 11.1 Cost of a single Bellman sweep

For *dense* tabular input (explicit $P_a(s,s')$ for all $(s,a,s')$):

| Operation                                                 | Complexity         | Explanation                            |
| --------------------------------------------------------- | ------------------ | -------------------------------------- |
| Compute $Q_k(s,a)=r_a(s)+\gamma\sum_{s'}P_a(s,s')v_k(s')$ | $O(S)$ per $(s,a)$ | dot product with entire next‑state row |
| Max over actions $a$                                      | $O(A)$ per $s$     | produce $v_{k+1}(s)=\max_a Q_k(s,a)$   |

Total **arithmetic operations**

$$
C_{\text{sweep}}=Θ(S^2A).
\tag{11.1}
$$

*Bit complexity.* Assuming rewards and transition probabilities are rational numbers with $b$‑bit numerators/denominators, each arithmetic op handles $O(b)$ bits; Equation (11.1) still applies up to poly‑$b$ factors.

*Sparse models.* When each state–action pair has at most $d\ll S$ successors, replace $S$ by $d$ in (11.1).

---

#### 11.2 Iteration count for ε‑optimal policy

From §10 (Theorem 10.1) the number of sweeps required is

$$
k^\*(\varepsilon,\gamma)=\Bigl\lceil \tfrac{\ln\!\bigl(\frac{2\gamma}{\varepsilon(1-\gamma)}\bigr)}{1-\gamma}\Bigr\rceil
 = Θ\!\Bigl(\tfrac{\ln(1/\varepsilon)}{1-\gamma}\Bigr).
\tag{11.2}
$$

---

#### 11.3 Overall runtime

Multiply (11.1) × (11.2):

$$
T_{\text{VI}}(\varepsilon)
   = Θ\!\Bigl(
        S^2A \;
        \tfrac{\ln\!\bigl(1/\varepsilon\bigr)}{1-\gamma}
      \Bigr)
      \quad\text{(dense model).}
\tag{11.3}
$$

The $\tilde{O}$ notation often used suppresses a secondary $\ln\!\frac1{1-\gamma}$ that enters via the ceiling in (11.2).

---

#### 11.4 Lower bound for table‑input planners

> **Theorem 11.1 (Chen & Wang 2017).**
> For any fixed $0<\gamma<1$ and $0<\delta<\frac12$, every deterministic algorithm that, *given explicit tables* for $P$ and $r$, returns a $\delta$-optimal deterministic policy must examine
>
> $$
> \Omega(S^2A)
> $$
>
> table entries in the worst case.

Combined with (11.3), value iteration is optimal **up to a logarithmic factor** in $1/\varepsilon$ (which the lower bound does not track) and the unavoidable $1/(1-\gamma)$ horizon term.

---

#### 11.5 Comparison with alternative planners

| Algorithm                | Per‑iteration cost              | Iterations for ε‑policy                              | Overall runtime       | Remarks                  |                      |                                                            |
| ------------------------ | ------------------------------- | ---------------------------------------------------- | --------------------- | ------------------------ | -------------------- | ---------------------------------------------------------- |
| **Value Iteration**      | $Θ(S^2A)$                       | $Θ\!\bigl(\frac{\ln(1/\varepsilon)}{1-\gamma}\bigr)$ | Eq. (11.3)            | Geometric rate           |                      |                                                            |
| **Policy Iteration**     | Solve $A_\pi v=r_\pi$: $Θ(S^3)$ | ≤                                                    | S                     | sweeps worst‑case        | $Θ(S^4)$ upper bound | Often faster in practice; sub‑cubic linear solves possible |
| **Modified PI (Howard)** | $Θ(S^3)$                        | (O(                                                  | S                     | ^2)) bounds              | $Θ(S^5)$             | Strong empirical speedups                                  |
| **Primal LP (IPM)**      | $Θ\bigl((S A)^{3}\bigr)$        | 1                                                    | Competitive only when | A                        |  ≪ S                 |                                                            |
| **Dual LP / Simplex**    | Variable                        | Exponential worst‑case                               | –                     | Good empirical behaviour |                      |                                                            |

For large $|A|$ or sparse $P$, value iteration’s $S^2A$ cost often dominates Gaussian solves, making it competitive with PI despite slower asymptotic rate.

---

#### 11.6 Memory consumption

| Algorithm        | Storage (dense)                            | Storage (sparse $d$)                      |
| ---------------- | ------------------------------------------ | ----------------------------------------- |
| Value Iteration  | $O(S + S^2A)$ for $P,r$ + $O(S)$ working   | $O(S+dSA)$                                |
| Policy Iteration | same model + $O(S^2)$ LU factors per sweep | ditto                                     |
| Primal LP        | $O(SA)$ constraints stored explicitly      | heavy memory; factorised solvers mitigate |

Working memory of VI is minimal: two value vectors + one residual, all $O(S)$.

---

#### 11.7 Parallel and hardware considerations

* Each state–action backup in (11.1) is independent ⇒ GPU/TPU batched matrix–vector multiply gives near‑peak utilisation.
* For **very large** S, **asynchronous** variants stream transitions from disk and update on‑the‑fly, trading sweeps for I/O passes; contraction still guarantees convergence (§4).
* Recent RL hardware (FP16) suffices since Bellman backups are numerically stable for $\gamma<1$.

---

#### 11.8 Tightness and open gaps

1. **Logarithmic factor.** Whether the $\ln(1/\varepsilon)$ in (11.3) can be removed without resorting to heavy linear solves remains open.
2. **Near‑undiscounted case.** No algorithm circumventing the $1/(1-\gamma)$ term is known for general MDPs.  For structured environments (e.g., deterministic transitions, low tree‑width graphs) sub‑linear horizons are possible.
3. **Randomised algorithms.** Chen–Wang’s lower bound assumes deterministic access.  Randomised query models may shave constants but not asymptotic order.

---

#### 11.9 Summary

Value iteration achieves

$$
T_{\text{VI}}(\varepsilon)=Θ\!\Bigl(S^{2}A\frac{\ln(1/\varepsilon)}{1-\gamma}\Bigr),
$$

which is *information‑theoretically minimal* for explicit, dense MDPs up to the standard $\ln(1/\varepsilon)$ factor and matching the Chen–Wang lower bound in its dependence on $|S|,|A|$.  Memory footprint and parallelism further cement VI as the baseline against which more sophisticated planners (policy iteration, LP, or specialised structure‑exploiting methods) must be judged.

---

*Primary sources*
Szepesvári, **RL Theory – Lecture 3** — runtime analyses and lower‑bound citation.
Jiang, **MDP Preliminaries** — LP complexity background.

---

### 12 Computational‑Complexity Landscape of Exact Planning in Finite MDPs

We revisit the complexity discussion with tighter statements, explicit *problem definitions*, and up‑to‑date results.  Throughout, the input MDP is $M=(S,A,P,r,\gamma)$ with $|S|=S,|A|=A$ and $r_a(s)\in[0,1]$.  All logarithms are base e.

#### 12.1 Decision versus optimisation problems

| Label          | Formal question                                                                           | Output          | Notes                                                |
| -------------- | ----------------------------------------------------------------------------------------- | --------------- | ---------------------------------------------------- |
| **VALUATION**  | Given $M$, state $s_0$ and threshold $\theta$, decide if $v^\*(s_0)\ge\theta$.            | *Yes/No*        | “Value decision”                                     |
| **PLAN‑EXIST** | Same input; decide if there exists a deterministic policy π with $v^{\pi}(s_0)\ge\theta$. | *Yes/No*        | Equivalent to VALUATION by Fundamental Theorem (§ 1) |
| **PLAN‑FIND**  | Produce a deterministic optimal policy.                                                   | table of size S | What planners (VI/PI/LP) actually do                 |

Complexity claims depend **crucially** on *how $P$ is presented* and *how the policy must be output*.  The lecture notes emphasise this point with a “needle‑in‑a‑haystack” construction.&#x20;

#### 12.2 Explicit (“table”) representation

*Input size* is Θ$(S^{2}A)$ numbers.  All three problems above are in deterministic polynomial time:

* **Upper bound.** Run Value Iteration for $k^\*(\varepsilon)=Θ(\frac{\ln(1/\varepsilon)}{1-\gamma})$ sweeps (§ 11) and greedify—time $Θ(S^{2}A\,k^\*)$.
* **Strongly polynomial alternative.** Ye (2011) proved that **policy iteration with Howard pivots is strongly polynomial when γ is fixed**, i.e. the number of arithmetic operations is bounded by poly$(S,A)$ independent of numerical magnitudes.&#x20;

Hence **PLAN‑FIND ∈ P** for table MDPs.  Deciding VALUATION/PLAN‑EXIST reduces to solving a single LP (Section 9) and is also in P.

#### Information‑theoretic lower bound

> **Theorem 12.1 (Szepesvári Lec 3 ↔ Chen‑Wang 2017).**
> Any deterministic algorithm that, given full tables, returns a $\delta$-optimal *deterministic* policy must access
>
> $$
> \Omega(S^{2}A)
> $$
>
> table entries in the worst case, even for fixed $\gamma$ and $\delta$.&#x20;

Because each access reveals at most one entry of $P$ or $r$, the **query complexity** bounds the arithmetic complexity.  Value Iteration’s per‑sweep cost $Θ(S^{2}A)$ is therefore optimal up to logarithmic factors in $1/(1-\gamma)$ and $1/\varepsilon$.

---

#### 12.3 Role of the output format

If the algorithm must *physically output* a state‑action table of size S (deterministic policy), runtime is lower‑bounded by Ω(S) merely to write the answer .  Streaming output does not hurt VI, which already touches every state each sweep.


#### 12.4 Succinct representations blow up hardness

When $P$ and $r$ are given by a **Boolean circuit, Bayesian network, or factored graph** of size poly‑$\log S$:

| Problem                             | Complexity class     | Source                                            |
| ----------------------------------- | -------------------- | ------------------------------------------------- |
| VALUATION (discounted γ fixed)      | **PSPACE‑complete**  | Mundhenk et al. 1997 – paraphrased in Lec 3 p. 5  |
| VALUATION (γ part of input, binary) | **EXPTIME‑complete** | ibid.                                             |
| Finite‑horizon horizon H in binary  | **PSPACE‑complete**  | ibid.                                             |

Intuitively, a succinct encoding can hide an *exponentially large* state graph; reading it explicitly is impossible, and the decision problem inherits circuit‑evaluation hardness.

#### 12.5 Average‑reward and undiscounted variants

* For explicit tables the LP remains polynomial but **no strongly‑polynomial bound** is known for Howard PI; it is conjectured open.
* For succinct inputs, optimisation is **NP‑hard** and in PSPACE.  Discounting is thus a computational *regulariser*.

#### 12.6 Bridging algorithmic and complexity results

| Algorithm (explicit input)      | Worst‑case runtime                               | Matches lower bound?                    |
| ------------------------------- | ------------------------------------------------ | --------------------------------------- |
| **Value Iteration**             | $Θ(S^{2}A \tfrac{\ln(1/\varepsilon)}{1-\gamma})$ | Yes, up to $\ln$ factors (Theorem 12.1) |
| **Howard PI (Ye 2011)**         | poly$(S,A)$ for fixed γ                          | Also optimal; better when ε→0           |
| **Simplex / LP (Ye, Puterman)** | strongly poly for fixed γ                        | Optimal but higher constants            |
| **Any deterministic planner**   | Ω$(S^{2}A)$                                      | Information bound                       |

Thus, **within the classical table model**, VI is near‑optimal; PI can remove the $\ln(1/\varepsilon)$ factor but at the cost of $Θ(S^{3})$ solves per sweep.

#### 12.7 Key take‑aways

* **Representation matters.** Table MDPs ⇒ problems in P; succinct MDPs ⇒ PSPACE/EXPTIME‑complete.
* **Information lower bounds** (needle‑in‑haystack construction) show Ω$(S^{2}A)$ reads are inevitable even *before* any computation.
* **Value Iteration** is time‑optimal up to logs; **Policy Iteration** achieves strong‑polynomial time when γ is a constant.
* Removing the $1/(1-\gamma)$ horizon term or designing average‑reward algorithms with matching bounds remains open.

#### Additional reading

* Chen, Y. & Wang, M. (2017). *Lower bound on the computational complexity of discounted MDPs.* arXiv 1705.07312.
* Ye, Y. (2011). *The simplex and policy‑iteration methods are strongly polynomial for the MDP with a fixed discount rate.*&#x20;
* Szepesvári, C. **RL‑Theory Lecture 3** — sections “Representations matter” & “Computational complexity lower bound”.

---

### 13 Error Control: δ – ε Dependencies in Value‑Iteration–Based Planning

Accurate planning hinges on rigorously relating **algorithmic tolerances** (per‑iteration residuals, backup noise) to **solution quality** (value and policy sub‑optimality).
Below we make each link explicit, quantify the δ/ε trade‑offs, and collect the tightest bounds known for tabular MDPs.

#### 13.1 Error metrics

| Symbol                                      | Definition                              | Typical scale                               |
| ------------------------------------------- | --------------------------------------- | ------------------------------------------- |
| $e_v := \lVert v-v^\*\rVert_\infty$         | **Value error**                         | ≤ $(1-\gamma)^{-1}$                         |
| $e_\pi := \lVert v^\*-v^{\pi}\rVert_\infty$ | **Policy error**                        | ≤ $(1-\gamma)^{-1}$                         |
| $r_k := \lVert v_{k+1}-v_k\rVert_\infty$    | **Residual** between sweeps             | shrinks geometrically                       |
| $b_k := \lVert Tv_k-v_k\rVert_\infty$       | **Bellman error**                       | interchangeable with $r_k$ up to $1-\gamma$ |
| $\varepsilon$                               | target *absolute* policy error          | user‑chosen                                 |
| $\delta$                                    | per‑backup noise / evaluation tolerance | algorithm‑chosen                            |

All norms are max‑norms; rewards are scaled to \[0,1].

---

#### 13.2 Value error ⇄ residual

$$
(1-\gamma)\,e_v \;\le\; b_k \;\le\; r_k \;\le\; \frac{1}{1-\gamma}\,b_k.\tag{13.1}
$$

*Proof.* Triangle inequality plus $Tv_{k+1}=v_{k+1}$.  (Szepesvári, Lec 3, p. 2)

---

#### 13.3 Policy error ⇄ value error (greedy bound)

$$
e_\pi \;\le\; \frac{2\gamma}{1-\gamma}\,e_v.\tag{13.2}
$$

Equality can occur (Singh‑Yee 1994 two‑state example).

---

#### 13.4 Putting it together – stop‑and‑greedy rule

Desired policy error $\varepsilon$ ⇒ pick residual tolerance

$$
r_\text{stop}\;=\;\frac{\varepsilon(1-\gamma)}{2\gamma}.\tag{13.3}
$$

Then
$e_v\le\frac{\varepsilon}{2\gamma}$ by (13.1) and (13.2) ⇒ $e_\pi\le\varepsilon$.

---

#### 13.5 Iteration count vs. ε

Starting from $v_0=0$:

$$
e_v(k)=\frac{\gamma^k}{1-\gamma},\quad
k(\varepsilon)=\Bigl\lceil\frac{\ln\!\bigl(\frac{2\gamma}{\varepsilon(1-\gamma)}\bigr)}{1-\gamma}\Bigr\rceil.
\tag{13.4}
$$

Hence **linear** $1/(1-\gamma)$ dependence and **logarithmic** $ \ln(1/\varepsilon)$ dependence—both known tight.

---

#### 13.6 Relative‑error variant

If user specifies relative tolerance $0<\delta_{\text{rel}}<1$:

$$
e_v\le \frac{\delta_{\text{rel}}(1-\gamma)}{2\gamma}\,\lVert v^\*\rVert_\infty
\;\Longrightarrow\;
k=Θ\!\Bigl(\frac{\ln(1/\delta_{\text{rel}})}{1-\gamma}\Bigr).\tag{13.5}
$$

Because $\lVert v^\*\rVert_\infty \le (1-\gamma)^{-1}$, absolute and relative targets coincide up to a constant factor when $v^\*$ is maximal.

---

#### 13.7 Finite‑horizon truncation error

$$
e_v^{H}:=\lVert v^\pi-v_H^\pi\rVert_\infty\le\frac{\gamma^{H}}{1-\gamma}.\tag{13.6}
$$

Setting $H=H_{\gamma,\varepsilon}$ from Eq. (2.6) meets any ε.

---

#### 13.8 Backup noise accumulation (approximate VI)

Suppose each sweep incurs additive error ≤ $\delta$:

$$
\tilde v_{k+1}=T\tilde v_k + \xi_k,\qquad \lVert\xi_k\rVert_\infty\le\delta.\tag{13.7}
$$

Then (Bertsekas & Tsitsiklis, 1996)

$$
\lVert\tilde v_k - v^\*\rVert_\infty
      \;\le\; \gamma^{k}\lVert \tilde v_0 - v^\*\rVert_\infty
             +\frac{\delta}{1-\gamma}.\tag{13.8}
$$

Hence *bias floor* $≈\delta/(1-\gamma)$; choose δ ≈ ε(1‑γ) to keep total error ≤ 2ε.

---

#### 13.9 Summary table

| Quantity controlled          | Dependence on target ε                                | Tight?                    | Citation   |
| ---------------------------- | ----------------------------------------------------- | ------------------------- | ---------- |
| Sweeps $k(\varepsilon)$      | $Θ\!\bigl(\tfrac{\ln(1/\varepsilon)}{1-\gamma}\bigr)$ | ✓ (two‑state lower bound) | Eq. (13.4) |
| Policy error vs. value error | factor $2\gamma/(1-\gamma)$                           | ✓ (Singh‑Yee)             | Eq. (13.2) |
| Residual vs. value error     | factor $1/(1-\gamma)$                                 | ✓                         | Eq. (13.1) |
| Backup noise δ → value bias  | $\delta/(1-\gamma)$                                   | ✓                         | Eq. (13.8) |
| Finite‑horizon H(ε)          | $Θ\!\bigl(\tfrac{\ln(1/\varepsilon)}{1-\gamma}\bigr)$ | ✓                         | Eq. (13.6) |

---

#### 13.10 Practical tuning checklist

1. **Pick ε for policy quality.**
2. Compute residual threshold via (13.3).
3. If approximate backups (sampling) give variance σ², set step size or sample count so that δ ≈ ε(1‑γ).
4. Monitor both $r_k$ and $b_k$; stop when either crosses threshold (robust).

Adhering to these δ/ε links guarantees the runtime bounds of § 11 while avoiding unnecessary computation.

---

### 14 Upper and Lower Bounds on Value‑Iteration Convergence

We close the theory roadmap by **pinning down how fast value iteration *can* and *must* converge**, independent of implementation details.

#### 14.1 Universal upper bound — γ‑geometric rate

From Banach (§ 4) every Bellman sweep contracts the error by γ:

$$
\lVert v_k-v^\*\rVert_\infty \;\le\; \gamma^{k}\,\lVert v_0-v^\*\rVert_\infty.\tag{14.1}
$$

With rewards in \[0,1] and $v_0=0$:

$$
\lVert v_k-v^\*\rVert_\infty \;\le\; \frac{\gamma^{k}}{1-\gamma},\tag{14.2}
$$

yielding the effective‑horizon expression $k=Θ\!\bigl(\tfrac{\ln(1/\varepsilon)}{1-\gamma}\bigr)$ (§ 2).
**No algorithm that performs *pure* Bellman sweeps can beat the γ‑factor** – it is intrinsic to the operator.

---

#### 14.2 Tight lower bound — Singh–Yee two‑state construction

> **Proposition 14.1 (Singh & Yee 1994).**
> There exists a deterministic MDP with $|S|=2, |A|=2$ s.t. equality holds in (14.2) for **every** $k\ge0$.

*Sketch of the MDP* (see Lec 3, p. 3):

* State 0 transitions to itself with reward 1 under action a₁;
* State 1 transitions to state 0 under a₂ with reward 0;
* Other actions give strictly lower rewards.

Starting from $v_0=0$, value iteration improves only the *currently sub‑optimal* state each sweep, achieving exactly γ‑geometric progress.  Thus **bound (14.2) cannot be improved in general**.

---

#### 14.3 Worst‑case iteration count

Combining the tight lower example with (14.2) gives a *matching* worst‑case sweep bound:

$$
\boxed{k_{\min}(\varepsilon)\;=\;\left\lceil\frac{\ln\!\bigl(\tfrac{1}{\varepsilon(1-\gamma)}\bigr)}{1-\gamma}\right\rceil.}\tag{14.3}
$$

Any algorithm restricted to synchronous Bellman updates and monotone initialisation needs at least $k_{\min}(\varepsilon)$ sweeps to reach ε‑value accuracy.

---

#### 14.4 Monotone vs. non‑monotone starts

* If $v_0\le v^\*$ (standard initialisation) the lower bound applies.
* Starting **above** $v^\*$ (optimistic Q‑learning style) yields identical γ‑rate by symmetry.
* Mixed or oscillating starts cannot surpass the γ‑contraction barrier because operator norm is *exactly* γ.

---

#### 14.5 Beyond the γ‑barrier — policy iteration & acceleration

* **Howard Policy Iteration** enjoys *quadratic local rate* once the greedy policy stabilises, explaining its practical speed‑ups (§ 11); but its *global* bound remains polynomial in S and A (Ye 2011).
* **Successive over‑relaxation (SOR)** reduces constants in (14.2) but not the exponent.
* **Multi‑grid / extrapolation** heuristics can super‑linearly accelerate smooth control problems yet have no worst‑case guarantees.

Thus γ‑geometric decay is the *best attainable uniform guarantee* for general finite MDPs.

---

#### 14.6 Open quantitative gaps

| Scenario                          | Known bound                               | Tight?    | Open question                                    |
| --------------------------------- | ----------------------------------------- | --------- | ------------------------------------------------ |
| Asynchronous single‑state backups | γ‑geometric (Bertsekas & Tsitsiklis)      | Not tight | State‑ordering‑dependent speed‑ups un‑quantified |
| Stochastic backups (TD(0))        | RMSE $≤\frac{\sigma}{\sqrt{(1-\gamma)N}}$ | Loose     | Sharper constants, non‑asymptotic lower bounds   |
| Average‑reward VI                 | Rate $γ\!\to\!1$ ⇒ sub‑linear             | Gap       | Tight deterministic example missing              |

---

#### 14.7 Summary

* **Upper bound:** γ‑geometric convergence (14.2) derived from contraction.
* **Lower bound:** 2‑state Singh–Yee MDP matches the rate, proving optimality.
* Consequently, the **effective horizon** $Θ(1/(1-\gamma))$ and sweep count in (14.3) are unimprovable for worst‑case tabular MDPs under Bellman‑update schemes.
* Faster global rates require *algorithmic regime changes* (policy iteration, LP solves, problem structure).

---

#### References

Singh, S. P. & Yee, R. (1994). *An Upper Bound on the Loss from Approximate Optimal Policies.* AI Journal. (Example re‑stated in Szepesvári Lec 3, p. 3)
Szepesvári, C. **RL Theory – Lecture 3** — Bound derivations and tightness discussion.

---

### 15 Conclusion & Outlook

Value iteration remains the **canonical baseline** for exact planning in finite, discounted Markov decision processes:

* **Fixed‑point bedrock.** Banach’s contraction theorem yields a uniquely optimal value $v^{\*}$ and a **γ‑geometric** error decay.
* **Effective horizon.** All analytic and computational quantities scale with $H_{\gamma,\varepsilon}=Θ\!\bigl(\tfrac{\ln(1/\varepsilon)}{1-\gamma}\bigr)$; this single expression unifies convergence proofs (§2), finite‑horizon truncations (§5) and runtime bounds (§11).
* **Greedy bridge.** The tight policy‑error bound ($ \frac{2\gamma}{1-\gamma} e_v$) turns value accuracy into policy optimality, enabling the “stop‑and‑greedy’’ algorithm (§10).
* **Nearly‑optimal complexity.** For explicit (table) MDPs value iteration’s $Θ(S^{2}A)$ per‑sweep cost is **information‑theoretically minimal** (Chen–Wang lower bound), and the total work is optimal up to logarithmic factors.
* **Geometry matters.** Dadashi et al.’s polytope view (§7) shows that Bellman updates traverse monotone line segments inside a piece‑wise‑linear value landscape—illuminating why deterministic policies suffice and how asynchronous or partial updates behave.
* **Linear programming duality** (§9) completes the picture: value iteration, policy iteration and LP solvers are different routes to the same primal‑dual optimum.

#### Open research avenues

1. **Near‑undiscounted regimes (γ → 1).** Can structure (e.g. deterministic dynamics, sparsity, low tree‑width) break the 1∕(1−γ) barrier while retaining polynomial time?
2. **Average‑reward complexity.** Strongly‑polynomial bounds for policy iteration with average cost remain unknown.
3. **Contraction beyond ℓ∞.** Weighted or non‑linear norms might tighten constants and accelerate practice without violating lower bounds.
4. **Polytope‑guided updates.** Algorithms that stay on low‑dimensional faces of the value‑function polytope could combine the locality of VI with the super‑linear jumps of PI.
5. **Succinct MDPs.** PSPACE hardness leaves room for approximation schemes—identifying best‑possible guarantees is largely open.

The interplay between **operator theory**, **computational complexity** and **geometric structure** continues to drive progress in planning and reinforcement learning.

---

### 16 References

The numeric labels used throughout map to the following sources:

| \[#]      | Bibliographic entry                                                                                                                                                            |
| --------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **\[1]**  | Szepesvári, Csaba. *Reinforcement Learning Theory* — Lecture 2: “Bellman Operators & Contraction.” (PDF `lec2_pdf.pdf`, 2020).                                                 |
| **\[2]**  | Szepesvári, Csaba. *Reinforcement Learning Theory* — Lecture 3: “Policy Error Bounds & Computational Complexity.” (PDF `lec3_pdf.pdf`, 2020).                                  |
| **\[3]**  | Jiang, Nan. *MDP Preliminaries and Linear‑Programming Formulations.* (PDF `note1.pdf`, 2019/24 update).                                                                        |
| **\[4]**  | Dadashi, R., Taïga, A., Le Roux, N., Schuurmans, D., & Bellemare, M. G. “The Value Function Polytope in Reinforcement Learning.” In *Proc. ICML 2019*. (PDF `dadashi19a.pdf`). |
| **\[5]**  | Singh, S. P., & Yee, R. (1994). “An Upper Bound on the Loss from Approximate Optimal Policies.” *Artificial Intelligence*, 67(1).                                              |
| **\[6]**  | Mundhenk, M. et al. (1997/2000). “Complexity of Finite‑Horizon and Discounted MDP Decision Problems.” *Journal of the ACM*, 47(4).                                             |
| **\[7]**  | Chen, Y. & Wang, M. (2017). “Lower Bounds on the Computation of Discounted MDPs.” *arXiv* 1705.07312.                                                                          |
| **\[8]**  | Ye, Y. (2011). “The Simplex and Policy‑Iteration Methods are Strongly Polynomial for MDPs with a Fixed Discount Rate.” *Math. Oper. Res.*, 36(4).                              |
| **\[9]**  | Bertsekas, D. P., & Tsitsiklis, J. N. (1996). *Neuro‑Dynamic Programming*. Athena Scientific.                                                                                  |
| **\[10]** | de Farias, D. P. & Van Roy, B. (2003). “The Linear Programming Approach to Approximate Dynamic Programming.” *Operations Research*, 51(6).                                     |

*(Entries \[5]–\[10] were cited for context; they are standard literature not included in the uploaded PDFs.)*

---
