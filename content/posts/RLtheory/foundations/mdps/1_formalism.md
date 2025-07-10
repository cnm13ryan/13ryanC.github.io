---
date: "2025-07-09"
title: "(1.1) Formalism: MDPs, policies, Value Functions, and Trajectories"
summary: "Formalism: MDPs, policies, Value Functions, and Trajectories"
lastmod: "2025-07-09"
category: "Notes"
series: ["RL Theory"]
author: "Bryan Chan"
hero: /assets/images/hero3.png
image: /assets/images/card3.png
---

# 1. Mathematical Foundations: Measure-theoretic set-up

This section establishes the measure-theoretic framework required for a rigorous analysis of stochastic dynamical systems, including those with continuous or hybrid state-action spaces. All of which are essential in the study of reinforcement learning. 

## 1.1. Measurable Spaces and Key Structures

We begin with the fundamental structure for defining probability measures, assuming familarity with set theory and the basic constructs of probability spaces.

For readers who need a refresher on set theory, I recommend you to study chapter 3 of Analysis I by Terry Tao. For a starter on probability spaces, I recommend you to study chapter 1 of Probability and Random Processes by Grimmett and Stirzaker.


**Definition 1.1.1 ($\sigma$-algebra)**

Let $E$ be some non-empty set and $P(E)$ be its power set. 

We have a set $\Sigma_E$ and we define some set $\Sigma_E$ as a $\sigma$-algebra on $E$ where $\Sigma_E \subseteq P(E)$ and satisfies the three axioms:

**A1** **Non-emptiness**: $E \in \Sigma_E$

**A2**  **Closure under complements**: If $A \in \Sigma_E$ then $E \setminus A = A^c \in \Sigma_E$

**A3**  **Closure under countable unions**: 

  * If $\lbrace A_n \rbrace_{n \in \mathbb N} \in \Sigma_E$

  * Then $U \in \Sigma_E$, where $U := \bigcup_{n \in \mathbb N} A_n$

> **Remark: Why is a $\sigma$-algebra necessary?**
> The need to define a specific collection of "measurable" sets arises from paradoxes in mathematics. For spaces like the real numbers, it is impossible to define a notion of "size" or "probability" for *every* possible subset in a way that is consistent with intuitive properties (like countable additivity). Pathological constructions, such as **Vitali sets**, demonstrate that some sets are inherently "immeasurable." The $\sigma$-algebra is the formal mechanism for selecting a well-behaved family of subsets (the "events") for which a measure can be coherently defined, thus avoiding these contradictions.


---

### Derived Properties of a σ-algebra

The following tables summarize the key properties that can be derived from the three fundamental axioms of a $\sigma$-algebra ($\Sigma_E$).

#### **1. Elementary Consequences**

These properties follow from a finite number of applications of the axioms.

| # | Property | Proof Sketch (using Axioms A1, A2, A3) |
| :--- | :--- | :--- |
| **1.1** | **The empty set is measurable.** | From **A1**, we have $E \in \Sigma_E$. Applying the closure under complements axiom (**A2**), we get $E^c = \emptyset \in \Sigma_E$. |
| **1.2** | **Closure under finite unions.** | For any two sets $A, B \in \Sigma_E$, we can form a countable sequence $\lbrace A_n \rbrace_{n \in \mathbb{N}}$ where $A_1 = A$, $A_2 = B$, and $A_n = \emptyset$ for $n \ge 3$. By **A3**, the union $\bigcup_{n \in \mathbb{N}} A_n = A \cup B$ is in $\Sigma_E$. |
| **1.3** | **Closure under finite intersections.** | Using De Morgan's laws: $A \cap B = (A^c \cup B^c)^c$. Since $A, B \in \Sigma_E$, **A2** implies $A^c, B^c \in \Sigma_E$. By property (1.2), their union is in $\Sigma_E$, and by **A2** again, the final complement is in $\Sigma_E$. |
| **1.4** | **Closure under set difference.** | The set difference can be written as $A \setminus B = A \cap B^c$. This follows from applying **A2** to get $B^c$ and then property (1.3) for the intersection. |
| **1.5** | **Closure under symmetric difference.** | The symmetric difference is $A \triangle B = (A \setminus B) \cup (B \setminus A)$. The result follows from applying property (1.4) to each term and then property (1.2) to their union. |

---

#### **2. Countable-Operation Consequences**

These properties involve a countable number of operations.

| # | Property | Proof Sketch |
| :--- | :--- | :--- |
| **2.1** | **Closure under countable intersections.** | For a sequence of sets $\lbrace A_n \rbrace_{n \in \mathbb{N}} \subseteq \Sigma_E$, we use De Morgan's laws: $\bigcap_{n \in \mathbb{N}} A_n = \left(\bigcup_{n \in \mathbb{N}} A_n^c\right)^c$. The result follows from **A2** (for the complements) and **A3** (for the countable union). |
| **2.2** | **Limit sets are measurable.** | For any sequence $\lbrace A_n \rbrace_{n \in \mathbb{N}}$, the limit superior and limit inferior are defined as:<br>$\limsup_{n \to \infty} A_n = \bigcap_{k \in \mathbb{N}} \bigcup_{n \ge k} A_n$<br>$\liminf_{n \to \infty} A_n = \bigcup_{k \in \mathbb{N}} \bigcap_{n \ge k} A_n$<br>Since these are formed by countable unions and intersections of sets in $\Sigma_E$, they are also in $\Sigma_E$ by **A3** and property (2.1). |
| **2.3** | **Closure under countable disjoint unions.** | This is a special case of **A3**, which guarantees closure under countable unions whether the sets are disjoint or not. |
| **2.4** | **Countable Boolean algebra.** | The combination of closure under complements (**A2**), countable unions (**A3**), and countable intersections (2.1) ensures that $\Sigma_E$ is closed under any countable combination of Boolean operations. |

---

#### **3. Structural Consequences**

These properties relate to the structure of $\sigma$-algebras themselves.

| # | Property | Proof Sketch |
| :--- | :--- | :--- |
| **3.1** | **Intersection stability.** | The intersection of any collection $\lbrace \Sigma_i \rbrace_{i \in I}$ of $\sigma$-algebras on the same set $E$ is itself a $\sigma$-algebra. This is because the closure properties defined by the axioms are preserved under arbitrary intersection. |
| **3.2** | **Generated $\sigma$-algebra.** | For any collection of subsets $\mathcal{C} \subseteq P(E)$, there exists a unique smallest $\sigma$-algebra containing it, denoted $\sigma(\mathcal{C})$. It is constructed as $\sigma(\mathcal{C}) = \bigcap \lbrace \Sigma \mid \mathcal{C} \subseteq \Sigma \text{ and } \Sigma \text{ is a } \sigma\text{-algebra on } E \rbrace$. This relies on property (3.1) and the fact that the power set $P(E)$ is always a valid $\sigma$-algebra containing $\mathcal{C}$. |
| **3.3** | **Trace $\sigma$-algebra.** | For a subset $Y \subseteq E$, the family $\Sigma_Y = \lbrace A \cap Y \mid A \in \Sigma_E \rbrace$ forms a $\sigma$-algebra on the set $Y$. The closure properties can be verified by noting that intersections and unions within $Y$ correspond to intersections and unions of the parent sets in $E$. |
---


**Definition 1.1.2 (Measurable Set)**

Any set $A$ is called a **measurable set** if it is an element of the $\sigma$-algebra $\Sigma_E$.

In other words, a set $A \subseteq E$ is measurable if and only if $A \in \Sigma_E$. The sets within the $\sigma$-algebra are, by definition, the sets that we are choosing to "measure."

**Definition 1.1.3 (Measurable Space)**

A **measurable space** is the pair $(E, \Sigma_E)$, consisting of a non-empty set $E$ and a $\sigma$-algebra $\Sigma_E$ defined on it.

---

## 1.2 Standard Borel Spaces

Useful results in probability theory and dynamic programming requires additional regularity. This motivates the use of a specific class of spaces.

The first step is to equip our non-empty set $E$ with a **topology**. A topology defines a notion of "openness" or "nearness" without necessarily defining a distance.

**Definition 1.2.1 (Topology)**

A **topology** on a set $E$ is a collection of subsets of $E$, denoted by $\mathcal{T}$, that satisfies three axioms:

**T1** **Contains empty set and the whole space**: $\emptyset , E \in \mathcal T$

**T2** **Closure under finite intersections**: If $A, B \in \mathcal{T}$, then $A \cap B \in \mathcal T$

**T3** **Closure under arbitrary unions**: If $\lbrace A_i \rbrace_{i \in I}$ is any collection of sets where each $A_i \in \mathcal T$, then $\bigcup_{i \in I} A_i \in \mathcal T$

The pair $(E, \mathcal{T})$ is called a **topological space**. The elements of $\mathcal{T}$ are called the **open sets**. A set $C \subseteq E$ is called a **closed set** if its complement, $C^c = E \setminus C$, is an open set.

---

With a topology in place, we can now define a canonical $\sigma$-algebra that connects the topological structure to the measurable structure we discussed previously.

**Definition 1.2.2 (Borel $\sigma$-algebra)**

Let $(E, \mathcal{T})$ be a topological space. The **Borel $\sigma$-algebra** on $E$, denoted $\mathcal{B}(E)$, is the $\sigma$-algebra generated by the collection of open sets $\mathcal{T}$.

Formally, using the "generated $\sigma$-algebra" concept (property 3.2 from our previous derivation of properties of $\sigma$-algebra):

$$\mathcal{B}(E) := \sigma(\mathcal{T})$$

This is the smallest $\sigma$-algebra on $E$ that contains all the open sets. The elements of $\mathcal{B}(E)$ are called **Borel sets** or **Borel-measurable sets**. A space equipped with its Borel $\sigma$-algebra, $(E, \mathcal{B}(E))$, is a **Borel space**.

For example, the standard Borel $\sigma$-algebra on the real numbers, $\mathcal{B}(\mathbb{R})$, is generated by the collection of all open intervals. Similarly, for $\mathbb{R}^n$, the Borel sets are generated by the collection of all open rectangles.
 
---

### 1.2.6 Finite‑Dimensional Vector‑Space Primer (Hoffman & Kunze §2–§3)

Much of reinforcement‑learning analysis silently assumes the algebra of finite‑dimensional vector spaces.  
For completeness we state the axioms **once**, together with the headline structure theorems.  
Everything below lives over an arbitrary field $F$ (typically $\mathbb R$).

* **Vector space.** $(V,+,\cdot)$ is an *abelian group* under “$+$” equipped with scalar multiplication  
  $F\times V \to V$ obeying (VS1) $a(bv)=(ab)v$, (VS2) $1v=v$, (VS3) $(a+b)v=av+bv$, (VS4) $a(v+w)=av+aw$.
* **Span / linear independence / basis.** A *basis* is a minimal spanning set ≡ maximal independent set. If $V$ is spanned by *any* finite set, every basis has the same size, called $\dim V$.

* **Coordinate representation.** Fix an ordered basis $\beta=(v_1,\dots,v_n)$; every $v\in V$ has a unique coordinate column $[v]_\beta\in F^n$.

* **Linear map.** $T:V \to W$ is *$F$–linear* iff $T(av+bw)=aT(v)+bT(w)$.  
  Kernel $\ker T$ and image $\text{im}T$ are subspaces; **rank–nullity** gives  
  $\displaystyle\dim V=\operatorname{rank}T+\operatorname{nullity}T$.

* **Matrix of a map.** Given ordered bases $\beta,\gamma$, the matrix $[T]_{\gamma \leftarrow \beta}$ is defined by $T[v]_\beta=[T]_{\gamma\leftarrow\beta}[v]_\beta$. Basis changes act by similarity: $[T]_{\gamma\leftarrow\beta'}=P^{-1}[T]_{\gamma\leftarrow\beta}P$.

> **Why here?** Once these objects are declared, later sections can freely talk about *operators*, *rank*, *eigen‑values*, and *coordinate vectors* without re‑introducing linear‑algebra syntax. Readers comfortable with matrices may skim. Formally careful readers now see that every “vector” below is an element of a well‑defined finite‑dimensional $F$‑space.


---

For a Borel space to be "standard," it must have a particularly "nice" underlying topology. This "niceness" is captured by the properties of a Polish space, which relies on the concept of a metric.

**Definition 1.2.3 (Polish Space)**

A topological space $(E, \mathcal{T})$ is a **Polish space** if its topology $\mathcal{T}$ is induced by a metric $d$ that is both **complete** and makes the space **separable**.

Let's break this down:

* A **metric space** $(E, d)$ is a set $E$ with a function $d: E \times E \to [0, \infty)$ that defines a distance between elements.

* A metric space is **complete** if every Cauchy sequence of points in $E$ converges to a limit that is also within $E$. This property is essential for analysis, guaranteeing that iterative procedures have well-defined limits.

* A metric space is **separable** if it contains a countable dense subset. This property prevents the space from being "too large" and is crucial for analysis. For instance, it ensures that probability measures can be uniquely characterized by their values on a countable collection of simpler sets (a generating $\pi$-system), which is essential for both theory and computation.
  
 

The real numbers $\mathbb{R}$ with their usual distance metric are a canonical example of a Polish space.

---
Before defining the standard borel space, we need to first understand measurable spaces that are isomorphic to the Polish space.

**Measurable Isomorphism**

For two measurable spaces to be considered structurally identical, there must exist a special kind of mapping between them.

Let $(S, \Sigma_S)$ and $(X, \Sigma_X)$ be two measurable spaces.

1.  A function $f: S \to X$ is **measurable** if the preimage of every measurable set in $X$ is a measurable set in $S$. Formally, for every set $A \in \Sigma_X$, it must be that $f^{-1}(A) \in \Sigma_S$.

2.  A **measurable isomorphism** is a bijective function $f: S \to X$ such that both $f$ and its inverse $f^{-1}: X \to S$ are measurable functions.

---

## 1.2.4 Borel Isomorphism Theorem (See Appendix B.2 for details)
With these constructs, we can now define a standard Borel space. It is a measurable space that is structurally identical (isomorphic) to the Borel space of a Polish space.


**Definition 1.2.4 (Standard Borel Space)**

A measurable space $(S, \Sigma_S)$ is called a **standard Borel space** if it is measurably isomorphic to a Polish space equipped with its Borel $\sigma$-algebra.

Therefore, a standard Borel space is any measurable space $(S, \Sigma_S)$ for which there exists some Polish space $P$ and a measurable isomorphism $f: S \to P$. 

This abstraction is powerful: it allows us to apply the strong results from Polish spaces to other spaces that are not naturally Polish but are structurally equivalent from a measure-theoretic perspective (e.g., the space of all policies). From this viewpoint, $S$ is indistinguishable from a Polish space like the real numbers.

The definition relies on a deep result in descriptive set theory: any two uncountable standard Borel spaces are isomorphic to each other. This means that for many purposes, they can all be treated as if they were the real numbers with their Borel sets, $(\mathbb{R}, \mathcal{B}(\mathbb{R}))$.

Standard Borel spaces are crucial in probability theory and stochastic processes because they are "well-behaved" and exclude the pathological sets that can complicate measure theory.

### 1.2.5 The Role of Regularity: Why Standard Borel?

The requirement for state and action spaces to be Standard Borel is not merely a technical convenience; it is the bedrock upon which the theory of stochastic processes is built. Its primary importance lies in guaranteeing the existence of **regular conditional probability distributions**.

Let $(\Omega, \mathcal{F}, \mathbb{P})$ be a probability space, $X$ be a random variable taking values in a Standard Borel space $(S, \mathcal{B}(S))$, and $\mathcal{G} \subseteq \mathcal{F}$ be a sub-$\sigma$-algebra. The theory ensures that a function $\kappa: \Omega \times \mathcal{B}(S) \to [0,1]$ exists, such that:

1.  For a fixed $\omega \in \Omega$, $\kappa(\omega, \cdot)$ is a probability measure on $(S, \mathcal{B}(S))$.
2.  For a fixed set $B \in \mathcal{B}(S)$, $\kappa(\cdot, B)$ is a $\mathcal{G}$-measurable function.
3.  It satisfies the conditional probability definition: $\mathbb{P}(X \in B | \mathcal{G})(\omega) = \kappa(\omega, B)$ almost surely.

This result is precisely what allows us to define the transition kernel $p(ds'|s,a)$ and policy kernel $\pi(da|s)$ as functions that map points to measures. Without it, the structural integrity of the MDP definition would be compromised.

> **Remark: Why Polish Topology is the Key Ingredient**
> The existence of regular conditional probabilities fails for general measurable spaces. The proof for Standard Borel spaces hinges on the properties of the underlying Polish topology. Here's the intuition:
>
> 1.  **Separability**: A Polish space has a countable dense subset. This allows us to "approximate" any open set with a countable collection of simpler sets (e.g., open balls centered on the dense points). By extension, this provides a countable family of sets (a $\pi$-system) that generates the entire Borel $\sigma$-algebra.
>
> 2.  **Characterizing Measures**: A probability measure is uniquely determined by its values on a generating $\pi$-system. Therefore, we only need to define our conditional probability kernel, $\kappa(\omega, \cdot)$, on this countable collection of simple sets.
>
> 3.  **Construction**: For each set $A$ in our countable generator, we can find a version of the conditional probability $\mathbb{P}(X \in A \mid \mathcal{G})$. We can then piece these together for all sets in the generator and, using a limiting argument (leveraging the completeness of the space), extend this definition to the entire $\sigma$-algebra. This process guarantees that for almost every $\omega$, the resulting function is a true, countably additive probability measure.
>
> In essence, the topological "niceness" of Polish spaces ensures there's a countable "scaffolding" upon which we can construct the conditional measure, avoiding the paradoxes of uncountable sets.
 

> **Remark: What Breaks Without Regularity?**
> If the space is not Standard Borel, the existence of a regular conditional probability is not guaranteed. This means that while a conditional probability might exist as an abstract object, there may be no function-like kernel $\kappa(\omega, \cdot)$ to represent it. Specifically, one of two things could fail:
> 1.  For some conditioning events $\omega$, the object $\kappa(\omega, \cdot)$ might not be a valid, countably additive probability measure.
> 2.  For some outcome sets $B$, the function $\omega \mapsto \kappa(\omega, B)$ might not be measurable, making it impossible to integrate and compute expectations.
> Without this guarantee, the transition kernel $p(ds'|s,a)$ itself might not be well-defined for all $(s,a)$, causing the entire MDP formalism to collapse.

**Common Examples**:

* Any **Polish space** with its Borel $\sigma$-algebra is, by definition, a standard Borel space. This includes $(\mathbb{R}, \mathcal{B}(\mathbb{R}))$, $(\mathbb{R}^n, \mathcal{B}(\mathbb{R}^n))$, and the Baire space $(\mathbb{N}^{\mathbb{N}}, \mathcal{B}(\mathbb{N}^{\mathbb{N}}))$.

* Any **Borel subset** of a standard Borel space is also a standard Borel space.

* Any non-empty **finite** or **countable** set, equipped with the discrete $\sigma$-algebra (the power set), is a standard Borel space.

---

## 1.3 Dynkin's $\pi-\lambda$ Theorem

Dynkin's $\pi-\lambda$ Theorem is important to ensuring the existence of a unique and well-defined probability measure over the infinite sequences of states and actions that constitutes the process's trajectories, which leads to the Ionescu-Tulcea Theorem.

To state Dynkin's $\pi-\lambda$ Theorem, we first need to define two key structures: a **$\pi$-system** and a **$\lambda$-system**. These are families of subsets of a given set that satisfy specific closure properties, which are weaker than those of a $\sigma$-algebra.


**Definition 1.3.1 ($\pi$-System)**

A **$\pi$-system** is a collection of subsets of a non-empty set $E$ that is closed under finite intersections.

Let $E$ be a non-empty set and $\mathcal{P}$ be a collection of subsets of $E$, so $\mathcal{P} \subseteq P(E)$. We call $\mathcal{P}$ a **$\pi$-system** if it satisfies the following axiom:
* **Closure under finite intersections**: If $A \in \mathcal{P}$ and $B \in \mathcal{P}$, then their intersection $A \cap B$ is also in $\mathcal{P}$.

By induction, this extends to any finite number of sets. A key feature is that a $\pi$-system is not required to contain $E$ or the empty set, nor is it necessarily closed under unions or complements.

**Definition 1.3.2 ($\lambda$-System)**

A **$\lambda$-system**, also known as a **Dynkin system (d-system)**, is a collection of subsets of $E$ that satisfies a different set of closure properties.

Let $E$ be a non-empty set and $\mathcal{L}$ be a collection of subsets of $E$, so $\mathcal{L} \subseteq P(E)$. We call $\mathcal{L}$ a **$\lambda$-system** if it satisfies the following three axioms:

**L1 Non-emptiness**: The entire set is included.
$$E \in \mathcal{L}$$

**L2 Closure under set difference**: If $A, B \in \mathcal{L}$ and $A \subseteq B$, then their difference is also in $\mathcal{L}$.
$$B \setminus A \in \mathcal{L}$$

**L3 Closure under countable increasing unions**: If $\lbrace A_n \rbrace_{n \in \mathbb N}$ is a sequence of sets in $\mathcal L$ such that $A_1 \subseteq A_2 \subseteq \dots$ (an increasing sequence), then their union is also in $\mathcal{L}$.

$$\bigcup_{n=1}^{\infty} A_n \in \mathcal{L}$$

An alternative, and often more common, set of axioms for a $\lambda$-system is:
1.  $E \in \mathcal{L}$.
2.  If $A \in \mathcal{L}$, then its complement $A^c \in \mathcal{L}$.
3.  If $\lbrace A_n \rbrace_{n \in \mathbb N}$ is a sequence of **pairwise disjoint** sets in $\mathcal L$ (i.e., $A_i \cap A_j = \emptyset$ for $i \neq j$), then their countable union $\bigcup_{n=1}^{\infty} A_n$ is in $\mathcal{L}$.

A crucial insight is that a collection of sets that is both a $\pi$-system and a $\lambda$-system is also a $\sigma$-algebra.

The two sets of axioms presented for a $\lambda$-system are equivalent.

With these definitions, we can now state the theorem. The Dynkin $\pi-\lambda$ Theorem provides a powerful tool for showing that a collection of sets is a $\sigma$-algebra. It is particularly useful in probability theory for proving that two measures are identical if they agree on a simpler class of sets.

**Theorem 1.3.3 (Dynkin's $\pi-\lambda$ Theorem)**

Let $E$ be a non-empty set. Let $\mathcal{P}$ be a **$\pi$-system** on $E$ and $\mathcal{L}$ be a **$\lambda$-system** on $E$.

If the $\pi$-system is contained within the $\lambda$-system, i.e., $\mathcal{P} \subseteq \mathcal{L}$

Then the $\sigma$-algebra generated by the $\pi$-system is also contained within the $\lambda$-system $\sigma(\mathcal{P}) \subseteq \mathcal{L}$

Here, $\sigma(\mathcal{P})$ denotes the smallest $\sigma$-algebra containing all the sets in $\mathcal{P}$, which was introduced as the "Generated $\sigma$-algebra" (Property 3.2) in the previous passage.

> **Motivation**: The theorem's power lies in simplifying proofs. To show two measures $\mu_1$ and $\mu_2$ are identical on a complex $\sigma$-algebra, we don't need to check every set. If we find a simple collection of sets $\mathcal{P}$ (e.g., open intervals on $\mathbb{R}$) that is a $\pi$-system and generates the $\sigma$-algebra, we only need to show $\mu_1(A) = \mu_2(A)$ for all $A \in \mathcal{P}$. The collection of sets where they agree forms a $\lambda$-system, and the theorem guarantees this agreement extends to the entire $\sigma$-algebra.


---

## 1.4 Probability Space and Probability Measures

Before defining probability spaces and probability measures, recall our prior definition of measurable sets and spaces in definitions 1.1.2 and 1.1.3 respectively.


**Definition 1.1.2 (Measurable Set)**

Any set $A$ is called a **measurable set** if it is an element of the $\sigma$-algebra $\Sigma_E$.

In other words, a set $A \subseteq E$ is measurable if and only if $A \in \Sigma_E$. The sets within the $\sigma$-algebra are, by definition, the sets that we are choosing to "measure."

**Definition 1.1.3 (Measurable Space)**

A **measurable space** is the pair $(E, \Sigma_E)$, consisting of a non-empty set $E$ and a $\sigma$-algebra $\Sigma_E$ defined on it.


**Definition 1.4.1 (Measurable Function)**:

Let $(E, \Sigma_E)$ and $(F, \Sigma_F)$ be two measurable spaces.

A function $f: E \to F$ is **measurable** if the preimage of every measurable set in the target space $F$ is a measurable set in the source space $E$. 

Formally, for every set $B \in \Sigma_F$, it must hold that $f^{-1}(B) \in \Sigma_E$, where $f^{-1}(B) = \{x \in E \mid f(x) \in B\}$.


We now define a function that assigns a "size" to the measurable sets in a measurable space.

**Definition 1.4.2 (Measure)**

Given a **measurable space** $(E, \Sigma_E)$, a **measure** is a function $\mu: \Sigma_E \to [0, \infty]$ that satisfies the following two properties:

1.  **Null Empty Set**: The measure of the empty set is zero, i.e., $\mu(\emptyset) = 0$.

2.  **Countable Additivity**: For any countable collection $\lbrace A_n \rbrace_{n \in \mathbb{N}}$ of **pairwise disjoint** sets in $\Sigma_E$ (meaning $A_i \cap A_j = \emptyset$ for $i \neq j$), the measure of their union is the sum of their individual measures:

    $$\mu\left(\bigcup_{n=1}^{\infty} A_n\right) = \sum_{n=1}^{\infty} \mu(A_n)$$

The triplet $(E, \Sigma_E, \mu)$ is called a **measure space**.

---

A **probability measure** is simply a measure that assigns a total "size" of 1 to the entire space.

**Definition 1.4.2 (Probability Measure)**

A **probability measure** $\mathbb{P}$ on a measurable space $(E, \Sigma_E)$ is a measure that satisfies the additional property $\mathbb{P}(E) = 1$.

This leads directly to the definition of a probability space, the fundamental object of modern probability theory.

**Definition 1.4.3 (Probability Space)**

A **probability space** is a measure space $(E, \Sigma_E, \mathbb{P})$ where the measure $\mathbb{P}$ is a probability measure. The triplet consists of:

* **Sample Space ($E$)**: A non-empty set of all possible outcomes.
* **$\sigma$-algebra ($\Sigma_E$)**: The set of all "events" (measurable subsets) to which we can assign a probability.
* **Probability Measure ($\mathbb{P}$)**: The function that assigns a probability between 0 and 1 to each event in $\Sigma_E$.

**Definition 1.4.4 (Push-forward Measure)**

Let $(E, \Sigma_E, \mu)$ be a measure space and $(F, \Sigma_F)$ be a measurable space. Let $f: E \to F$ be a measurable function.

The **push-forward measure** of $\mu$ by $f$, denoted $f_\mu$ or $\mu \circ f^{-1}$, is a measure defined on the target space $(F, \Sigma_F)$. 

For any measurable set $B \in \Sigma_F$, its measure is defined as $(f_\mu)(B) := \mu(f^{-1}(B))$

> **Remark**: If $\mathbb{P}$ is a probability measure on $(E, \Sigma_E)$ and $X: E \to F$ is a measurable function (a random variable), then the push-forward measure $X_\mathbb{P}$ is a probability measure on $(F, \Sigma_F)$. This push-forward measure is called the **law** or **distribution** of the random variable $X$.
 

---

## 1.5 Random Variables and Expectation

With probability spaces defined, we can formalize the concept of a random variable and its expectation.

**Definition 1.5.1 (Random Variable)**

A **random variable** is simply a **measurable function** from the sample space of a probability space to another measurable space (the "state space").

Let $(\Omega, \mathcal{F}, \mathbb{P})$ be a probability space and $(E, \Sigma_E)$ be a measurable space. 

A function $X: \Omega \to E$ is called an $E$-valued **random variable** if it is measurable, i.e., for every set $A \in \Sigma_E$, its preimage $X^{-1}(A) = \lbrace \omega \in \Omega \mid X(\omega) \in A \rbrace$ is an event in $\mathcal{F}$.

**Definition 1.5.2 (Expectation)**

The **expectation** of a real-valued random variable is its average value, weighted by the probability measure. It is defined by the **Lebesgue integral** of the random variable with respect to the measure $\mathbb{P}$.

Let $X: \Omega \to \mathbb{R}$ be a random variable. Its expectation, denoted $\mathbb{E}[X]$, is defined as:

$$ 
\mathbb{E}[X] := \int_{\Omega} X(\omega) d\mathbb{P}(\omega) \quad \text{or simply} \quad \int X d\mathbb{P}
$$

The integral is constructed in three steps:

1.  **Simple Functions**: If $X = \sum_{i=1}^n a_i \mathbf 1_{A_i}$ for disjoint events $A_i \in \mathcal F$, then $\mathbb{E}[X] = \sum_{i=1}^n a_i \mathbb{P}(A_i)$.

2.  **Non-negative Functions**: If $X \ge 0$, $\mathbb{E}[X]$ is the supremum of expectations of all simple functions $Y$ such that $0 \le Y \le X$.

3.  **General Functions**: 
    * For any $X$, we write $X = X^+ - X^-$, where $X^+ = \max(X, 0)$ and $X^- = \max(-X, 0)$. 
    * The variable $X$ is said to be **integrable** if $\mathbb{E}[|X|] = \mathbb{E}[X^+] + \mathbb{E}[X^-] < \infty$. 
    * If it is integrable, its expectation is defined as $\mathbb{E}[X] = \mathbb{E}[X^+] - \mathbb{E}[X^-]$. 
    * This condition is crucial to ensure the result is not an undefined form like $\infty - \infty$.


> **Remark: The Power of the Lebesgue Integral**
> The true analytical power of the Lebesgue integral comes from its behavior with limits. Two cornerstone results, omitted here for brevity but essential for rigorous proofs, are the **Monotone Convergence Theorem (MCT)** and the **Dominated Convergence Theorem (DCT)**. These theorems provide precise conditions under which one can exchange the order of limits and integration ($\lim \int = \int \lim$). This operation is fundamental to proving the convergence of iterative algorithms and establishing the completeness of function spaces used in dynamic programming.


The integral in the definition of the Markov operator, $(Pg)(x) := \int_F g(y) \kappa(x, dy)$, is precisely this expectation, where the probability measure is the kernel $\kappa(x, \cdot)$.

## 1.6 Probability Kernels


**Definition 1.6.1 (Probability Kernel)**

A **probability kernel** (also known as a stochastic kernel or Markov kernel) is a function that maps every point in a starting measurable space to a probability measure on a target measurable space. This mapping must be done in a "measurable" way.

Let $(E, \Sigma_E)$ and $(F, \Sigma_F)$ be two measurable spaces. 

A **probability kernel** $\kappa$ from $(E, \Sigma_E)$ to $(F, \Sigma_F)$ is a function $\kappa: E \times \Sigma_F \to [0, 1]$ satisfying two conditions:

1.  **Probability Measure Condition**: For every fixed point $x \in E$, the function $\kappa(x, \cdot): \Sigma_F \to [0, 1]$ is a **probability measure** on $(F, \Sigma_F)$. This means:
    * $\kappa(x, F) = 1$.
    * For any countable collection of pairwise disjoint sets $\lbrace B_n \rbrace_{n \in \mathbb N} \subseteq \Sigma_F$, we have $\kappa(x, \bigcup_{n=1}^{\infty} B_n) = \sum_{n=1}^{\infty} \kappa(x, B_n)$.

2.  **Measurability Condition**: For every fixed measurable set $B \in \Sigma_F$, the function $x \mapsto \kappa(x, B)$ is a **measurable function** from $(E, \Sigma_E)$ to $([0, 1], \mathcal{B}([0, 1]))$.

Essentially, a probability kernel $\kappa(x, B)$ can be interpreted as the probability of transitioning from a point $x$ into the set $B$.

> **Remark**: The measurability condition for $\kappa$ is crucial: for any measurable set of outcomes $C \in \mathcal{B}(S) \otimes \mathcal{B}(\mathbb{R})$, the function $(s, a) \mapsto \kappa(s, a, C)$ must be a measurable function. This ensures the dynamics are well-behaved. This joint formulation is more general and rigorous than separating transitions and rewards, as it naturally models situations where the distribution of the reward $r'$ is statistically dependent on the resulting next state $s'$.

> Without this condition, fundamental quantities like the expected reward $r(s,a)$ or the state-value function $v_\pi(s)$ might not be well-defined, as their definitions rely on integrals that require measurable integrands. See Appendix A.4 for the full $\pi$-$\lambda$/monotone-class chain we rely on.

A profound result connecting kernels to conditioning is that, on standard Borel spaces, conditional probabilities can always be represented by a kernel.

**Theorem 1.6.2 (Rokhlin's Regular Conditional Probability Theorem)**

Let $(\Omega, \mathcal{F}, \mathbb{P})$ be a probability space, $X$ be a random variable taking values in a standard Borel space $(E, \Sigma_E)$, and $\mathcal{G} \subseteq \mathcal{F}$ be a sub-σ-field. 

Then there exists a probability kernel $K: \Omega \times \Sigma_E \to [0, 1]$ such that for any $B \in \Sigma_E$:
$$
K(\omega, B) = \mathbb{P}(X \in B \mid \mathcal{G})(\omega) \quad \text{for } \mathbb{P}\text{-almost every } \omega \in \Omega
$$

This kernel is called the **regular conditional distribution** of $X$ given $\mathcal{G}$. Its existence is fundamental for defining belief states in partially observable models and for disintegrating measures.

> **Remark (Rigorous Belief States).** This theorem provides the rigorous foundation for the concept of a **belief state** in POMDPs. It guarantees that an agent's belief—the conditional probability distribution over a hidden state, given a history of actions and observations—exists as a well-defined probability kernel. This allows the belief itself to be treated as the state in a new, fully-observable decision process.
 

---

**Lemma 1.6.2 (Composition of Kernels)**

The composition of two probability kernels defines a new kernel that represents a two-step probabilistic transition.

Let $(E, \Sigma_E)$, $(F, \Sigma_F)$, and $(G, \Sigma_G)$ be three measurable spaces.
* Let $\kappa_1$ be a probability kernel from $(E, \Sigma_E)$ to $(F, \Sigma_F)$.
* Let $\kappa_2$ be a probability kernel from $(F, \Sigma_F)$ to $(G, \Sigma_G)$.

The **composition** of these kernels, denoted $\kappa_1 \otimes \kappa_2$, is a probability kernel from $(E, \Sigma_E)$ to $(G, \Sigma_G)$ defined as follows:
For any $x \in E$ and any measurable set $C \in \Sigma_G$, the composition is given by the integral:

$$(\kappa_1 \otimes \kappa_2)(x, C) := \int_{F} \kappa_2(y, C)  \kappa_1(x, dy)$$

This integral represents the expected probability of reaching the set $C$ from an intermediate point $y \in F$, where the intermediate point $y$ is chosen according to the probability measure $\kappa_1(x, \cdot)$. 

The integral is well-defined because the integrand $y \mapsto \kappa_2(y, C)$ is a measurable function, a direct consequence of the definition of $\kappa_2$ as a probability kernel.

> **Remark**: Kernel composition is important for analysing multi-step dynamics. Apply a policy kernel to a transition kernel yields the system's evolution under that policy. Repeatedly composing this resulting kernel with itself allows for the evaluation of the system over horizons, useful for deriving theoretical results for value iteration and policy iteration.

---

**Definition 1.6.3 (Markov Operator)**

A probability kernel induces a linear operator, known as a **Markov operator** or **transition operator**, which describes how the kernel acts on measurable functions. This is fundamental to defining the evolution of expectations in a stochastic system.

Let $\kappa$ be a probability kernel from $(E, \Sigma_E)$ to $(F, \Sigma_F)$. The corresponding **Markov operator**, often denoted $P_\kappa$ or simply $P$, maps bounded measurable functions on the target space $F$ to bounded measurable functions on the source space $E$.

For any bounded, measurable function $g: (F, \Sigma_F) \to (\mathbb{R}, \mathcal{B}(\mathbb{R}))$, the operator $P$ produces a new function $Pg: E \to \mathbb{R}$ defined by:

$$(Pg)(x) := \int_{F} g(y)  \kappa(x, dy)$$


> **Linear‑algebra view.** $B(F)$ is a vector space and $P_\kappa:B(F) \to B(E)$ is its linear image.  
> Rank–nullity implies that when $B(F)$ is finite‑dimensional (e.g. tabular RL), injectivity $\Longleftrightarrow$ surjectivity.  
> In that case $P_\kappa$ can be represented by an $n\times n$ matrix and the spectrum $\sigma(P_\kappa)$ controls convergence of value‑iteration—see §2.2.3.
 

The operator $P$ is a **linear transformation** (or **linear operator if input and output space is identifical**) in the sense of abstract linear algebra. Its linearity can be verified directly:

* For any two functions $g_1, g_2$ in the domain, $P(g_1 + g_2)(x) = \int (g_1(y) + g_2(y))\kappa(x, dy) = (Pg_1)(x) + (Pg_2)(s)$.

* For any scalar $c$, $P(c \cdot g)(x) = \int c \cdot g(y)\kappa(x, dy) = c \cdot (Pg)(s)$.

This reframes the action of the kernel from a mere probabilistic rule to an algebraic object acting on a vector space of functions.

The value $(Pg)(x)$ is the **expected value** of the function $g$ after a one-step transition, given that the process starts at point $x$. The expectation is calculated with respect to the probability measure $\kappa(x, \cdot)$ on the target space $F$.

> **A Note on Notation**: The expression $\kappa(x, dy)$ is standard measure-theoretic notation. It denotes that the integral is taken with respect to the variable $y$, over the target space $F$, using the specific measure $\kappa(x, \cdot)$. For a fixed $x$, $\kappa(x, \cdot)$ is a probability measure on $F$.


> **Remark: Duality of Kernels and the Adjoint Operator**. The Markov operator $P$ represents the kernel's "pull-back" action on a vector space of functions. The "push-forward" action on measures is its formal **adjoint**.
>
> In the framework of abstract linear algebra, for a vector space $V$, its **dual space** $V^\ast$ is the space of all linear functionals on $V$. The **Riesz-Markov-Kakutani representation theorem** establishes that for a well-behaved topological space $S$, the dual of the space of continuous functions $C(S)$ is the space of regular Borel measures $M(S)$.
>
> This provides the formal foundation for duality. The push-forward operator, let's call it $P^\ast: M(E) \to M(F)$, is the **adjoint** of the Markov operator $P: B(F) \to B(E)$. They are linked by the defining property of an adjoint: for any function $g \in B(F)$ and any measure $\mu \in M(E)$, we have:
> $$ \langle Pg, \mu \rangle_E = \langle g, P^\ast \mu \rangle_F \quad \text{where} \quad \langle f, \nu \rangle := \int f d\nu $$
> The push-forward action $(\mu P)(B)$ in the original text is precisely this adjoint operator $P^\ast$ acting on the measure $\mu$. This dual view of an operator acting on a vector space and its adjoint acting on the dual space is a cornerstone of functional analysis and operator theory.
 

---

## 1.7 Stochastic Processes and Product Spaces

To analyze trajectories, which are infinite sequences of states and actions, we need to construct a measure space for these sequences.

**Definition 1.7.1 (Product Measurable Space)**

Given a sequence of measurable spaces $\lbrace E_n, \Sigma_n \rbrace_{n \in \mathbb N}$, 

their **product space** is the Cartesian product $E = \prod_{n=1}^\infty E_n$. 

The corresponding **product $\sigma$-algebra**, denoted $\bigotimes_{n=1}^{\infty} \Sigma_n$, is the smallest $\sigma$-algebra on $E$ that makes all projection maps $\pi_k: E \to E_k$ measurable.

Intuitively, this is the $\sigma$-algebra generated by all **cylinder sets** (or "finite-dimensional rectangles"), which are sets of trajectories constrained on a finite number of coordinates.
 

**Definition 1.7.2 (Stochastic Process)**

A **stochastic process** is a collection of random variables indexed by time. 

For a discrete-time process, this is a sequence $\lbrace X_n \rbrace_{n \in T}$ where $T \subseteq \mathbb{N}$, and each $X_n$ is a random variable mapping from a probability space $(\Omega, \mathcal{F}, \mathbb{P})$ to a state space $(S, \Sigma_S)$.

The **path space** of the process is the product space of all possible trajectories, e.g., $S^\mathbb{N}$. A key question is whether there exists a probability measure on this path space that is consistent with the process's one-step dynamics.

**Theorem 1.7.3 (Ionescu-Tulcea Extension Theorem)**

Let $\lbrace E_n, \Sigma_n \rbrace_{n \geq 1}$ be a sequence of measurable spaces. 

Let $\mu_1$ be a probability measure on $(E_1, \Sigma_1)$. For each $n \geq 1$, let $\kappa_{n+1}$ be a probability kernel from $(\prod_{i=1}^n E_i, \bigotimes_{i=1}^n \Sigma_i)$ to $(E_{n+1}, \Sigma_{n+1})$.

Then there exists a **unique** probability measure $\mu$ on the product space $(\prod_{n=1}^\infty E_n, \bigotimes_{n=1}^\infty \Sigma_n)$ that is consistent with the initial distribution and the sequence of kernels.

This theorem is the cornerstone for defining stochastic processes. It guarantees that if we specify an initial distribution and the transition dynamics (via kernels), a well-defined probability measure exists over the space of all possible futures. 

The assumption that the underlying spaces are **standard Borel** is critical because it guarantees the existence of **regular conditional probabilities**. This property ensures that the kernels $\kappa_{n+1}$ are "regular enough" for the theorem to apply robustly, providing the mathematical backbone for defining complex stochastic processes.
  
## Appendix A — Measure‑Theory Toolbox (self‑contained)

### A.0 Basic set‑system terminology

- **Algebra**: non‑empty $\mathcal{A} \subset P(E)$ closed under finite unions and complements.  

- **$\sigma$‑algebra**: algebra closed under countable unions (denoted $\Sigma$).  

- **$\pi$‑system**: non‑empty $\mathcal{C}$ closed under finite intersections.  

- **$\lambda$‑system**: $\mathcal{L}$ such that:
  (i) $E\in\mathcal{L}$, 
  (ii) $A\subset B$ with $B\in\mathcal{L} \Rightarrow B\setminus A\in\mathcal{L}$, 
  (iii) $A_i \uparrow \Rightarrow \bigcup A_i\in\mathcal{L}$.  

A **monotone class** satisfies (iii) and its decreasing analogue.

### A.1 Outer measures and Carathéodory construction

**Definition A.1.1 (outer measure).**  

An outer measure on $E$ is a map $\mu^\ast:P(E)\to[0,\infty]$ that is (i) null on $\emptyset$, (ii) monotone, (iii) countably sub‑additive.

**Definition A.1.2 (Carathéodory measurability).**  

$A\subset E$ is *Carathéodory‑measurable* w.r.t. $\mu^\ast$ if $\forall B\subset E$  

$\mu^{\ast}(B)=\mu^{\ast}(B\cap A)+\mu^{\ast}(B\setminus A).$

**Theorem A.1.3 (Carathéodory Extension).**  

Let $\mathcal{A}$ be an algebra on $E$ and $\mu_0:\mathcal{A}\to[0,\infty]$ a **pre‑measure** ($\sigma$‑additive on $\mathcal{A}$).  

Define $\mu^\ast$ by the infimum over $\mathcal{A}$‑covers. Then  

- $\mu^\ast$ is an outer measure;  
- $\Sigma^\ast := \{A\subset E : A \text{ Carathéodory‑measurable}\}$ is a $\sigma$‑algebra containing $\mathcal{A}$;  
- the restriction $\mu := \mu^\ast|_{\Sigma^\ast}$ is **complete**;  
- if $\mu_0$ is **$\sigma$‑finite or finite**, $\mu$ is the unique extension of $\mu_0$ to $\sigma(\mathcal{A})$.

### A.2 $\pi$‑$\lambda$ and monotone‑class machinery

**Theorem A.2.1 ($\pi$‑$\lambda$).**  

For a $\pi$‑system $\mathcal{C}$ and a $\lambda$‑system $\mathcal{L}$ with $\mathcal{C}\subset\mathcal{L}$ we have $\sigma(\mathcal{C})\subset\mathcal{L}$.

**Theorem A.2.2 (Monotone‑class).**  

If $\mathcal{M}$ is a monotone class containing an algebra $\mathcal{A}$, then $\sigma(\mathcal{A})\subset\mathcal{M}$.

These tools give uniqueness by comparing two measures on a generating $\pi$‑system.

### A.3 Kolmogorov Extension (projective‑limit form)

**Theorem A.3.1.**  
Let $\{(E_n,\Sigma_n)\}_{n\geq 1}$ be measurable spaces and $\{\mu_I\}$ finite‑dimensional probability measures indexed by finite $I\subset\mathbb{N}$, satisfying the *projection consistency* condition.  

Then there exists a unique probability $\mu$ on $\left(\prod_{n=1}^{\infty}E_{n},\bigotimes_{n=1}^{\infty}\Sigma_{n}\right)$ with $\mu\circ\pi_I^{-1}=\mu_I$.  

*Sketch*: define a pre‑measure on cylinder sets ($\pi$‑system) $\to$ Carathéodory; uniqueness via $\pi$‑$\lambda$.
### A.4 $\sigma$‑finite measures

**Definition A.4.1.**  

$\mu$ on $(E,\Sigma)$ is **$\sigma$‑finite** if $E=\bigcup_n E_n$ with $\mu(E_n)<\infty$.

| $\mu$ | Space $(E,\Sigma)$ | $\sigma$‑finite decomposition |
|---|-------------|------------------------|
| Lebesgue $\lambda$ | $(\mathbb{R},\mathfrak{B}(\mathbb{R}))$ | $\bigcup_n [-n,n]$ |
| Counting on $\mathbb{N}$ | $(\mathbb{N},P(\mathbb{N}))$ | $\bigcup_n \{n\}$ |
| *Non‑$\sigma$‑finite*: counting on an uncountable set | $(2^{\aleph_0},P(2^{\aleph_0}))$ | impossible |

### A.5 Convergence theorems (on a measure space $(E,\Sigma,\mu)$)

- **A.5.1 Monotone Convergence (Beppo‑Levi)**: $f_n\nearrow f$, $f_n\geq 0 \Rightarrow \int f  d\mu = \lim\int f_n  d\mu$.  
- **A.5.2 Dominated Convergence**: $f_n\to f$ $\mu$‑a.e., $|f_n|\leq g$ with $g\in L^1(\mu) \Rightarrow f\in L^1(\mu)$ and $\int f = \lim\int f_n$.  
- **A.5.3 Radon–Nikodym**: $\sigma$‑finite $\mu,\nu$ and $\nu\ll\mu \Rightarrow \exists! h=d\nu/d\mu$ in $L^1_+(\mu)$ with $\nu(A)=\int_A h  d\mu$.
### A.6 Product integration: Fubini–Tonelli

Let $(E_1,\Sigma_1,\mu_1)$ and $(E_2,\Sigma_2,\mu_2)$ be $\sigma$‑finite.

1. **Tonelli (non‑negative $f$).**  
   $f\geq 0 \Rightarrow$ iterated integrals exist (possibly $\infty$) and  
   
   $\int fd(\mu_1\otimes\mu_2)=\int_{E_1}\left(\int_{E_2}f\right)d\mu_1
                   =\int_{E_2}\left(\int_{E_1}f\right)d\mu_2.$

2. **Fubini (integrable $f$).**  
   $f\in L^1(\mu_1\otimes\mu_2) \Rightarrow$ the iterated integrals are finite, measurable and equal to the product integral.


---

# 2. The Markov Decision Process Framework

### 2.1 The Agent-Environment Interaction Loop

The formal MDP tuple provides the mathematical blueprint for the core interaction in reinforcement learning. This interaction is best understood as a continuous loop between an **agent** and its **environment**.

The loop unfolds in discrete time steps `t = 0, 1, 2, ...`:
1.  **Observation**: The agent observes the current state of the environment, $s_t \in S$.

2.  **Action**: Based on this state, the agent selects an action, $a_t \in A$, according to its policy.

3.  **Response**: The environment receives the state-action pair $(s_t, a_t)$ and responds by transitioning to a new state, $s_{t+1}$, and issuing a corresponding reward, $r_{t+1} \in \mathbb{R}$. Formally, the outcome pair $(s_{t+1}, r_{t+1})$ is a single sample drawn from the probability measure defined by the transition kernel $\kappa(\cdot | s_t, a_t)$.

This cycle then repeats from the new state $s_{t+1}$.

## 2.2 Formal Definition of an Markov Decision Process

We can now use the preceding concepts to provide a rigorous, unified definition of a Markov Decision Process (MDP).

A **Markov Decision Process** is a tuple $(S, A, \kappa, \gamma)$, where:

* **State Space ($S$)**: A standard Borel space $(S, \mathcal{B}(S))$.

* **Action Space ($A$)**: A standard Borel space $(A, \mathcal{B}(A))$.

* **Discount Factor ($\gamma$)**: A scalar $\gamma \in [0, 1)$.

* **Transition Kernel ($\kappa$)**: A **probability kernel** that maps a state-action pair to a distribution over possible **outcomes**. To provide a unified treatment of payoffs, the reward is considered an integral part of the outcome. The kernel therefore maps from the state-action space $(S \times A, \mathcal{B}(S) \otimes \mathcal{B}(A))$ to the joint outcome space of the next state and its associated reward, $(S \times \mathbb{R}, \mathcal{B}(S) \otimes \mathcal{B}(\mathbb{R}))$.

This single kernel, written $\kappa(ds' dr' | s, a)$, defines a joint probability measure over the next state $s'$ and the received reward $r'$. This approach provides a fully unified treatment of all payoff structures:

* **Deterministic Rewards**: The marginal measure on the reward space is a Dirac delta measure, e.g., $\delta_{R(s,a)}(dr')$.

* **Discrete Stochastic Rewards**: The marginal measure is a finite or countable sum of weighted Dirac delta measures.

* **Continuous Stochastic Rewards**: The marginal measure is described by a probability density function over $\mathbb{R}$.
 

> **Remark**: The measurability condition for $\kappa$ is crucial: for any measurable set of outcomes $C \in \mathcal{B}(S) \otimes \mathcal{B}(\mathbb{R})$, the function $(s, a) \mapsto \kappa(s, a, C)$ must be a measurable function. This ensures the dynamics are well-behaved.

From this unified kernel, we can derive the simpler, more common transition and reward functions:

1.  **State Transition Kernel ($p$)**: This is the marginal distribution over the next state $S$. It is a probability kernel from $(S \times A)$ to $(S)$ obtained by integrating out the reward:

$$
p(ds'|s, a) := \int_{\mathbb{R}} \kappa(ds', dr' | s, a)
$$

2.  **Expected Reward Function ($r$)**: This is the expected value of the reward, given state $s$ and action $a$. It is a bounded, measurable function $r: S \times A \to \mathbb{R}$ obtained by integrating the reward against its distribution:

$$
r(s, a) := \int_{S \times \mathbb{R}} r' \ \kappa(ds', dr' | s, a)
$$

For this integral to be well-defined and finite, the reward variable must be integrable. 

This requires the explicit assumption that $\int |r'| \kappa(ds'dr'|s,a) < \infty$ for all $(s,a)$. 

A common sufficient condition is that all rewards are uniformly bounded. This formulation, by treating the reward as a component of the transition's outcome, is the proper and most general application of the measure-theoretic framework.

#### Concrete Construction of the Unified Kernel

To bridge the gap between this abstract formulation and the more common $(S, A, P, R, \gamma)$ tuple, consider an MDP where the reward is a deterministic function of the state and action, $R(s,a)$. The environment's dynamics are given by the state transition kernel $p(ds'|s,a)$.

In this common scenario, the unified kernel $\kappa$ takes a specific form. The distribution over the next reward $r'$ is a **Dirac delta measure** concentrated at the point $R(s,a)$. The kernel is therefore the product of the state transition measure and this Dirac measure:

$$
\kappa(ds' dr' | s, a) = p(ds' | s, a) \otimes \delta_{R(s,a)}(dr')
$$

When we compute the expected reward function $r(s,a)$ using this kernel, we recover the original function:

$$
r(s, a) := \int_{S \times \mathbb{R}} r' \ [p(ds' | s, a) \otimes \delta_{R(s,a)}(dr')] = \int_{\mathbb{R}} r' \delta_{R(s,a)}(dr') = R(s,a)
$$



---

## 2.2 Policies: The Agent's Strategy

A **policy** defines the agent's behavior.

**Definition 2.2.0 (Policy)**

A **policy** $\pi$ is a **probability kernel** from the state space $(S, \mathcal{B}(S))$ to the action space $(A, \mathcal{B}(A))$. For each state $s \in S$, the policy defines a probability measure $\pi(da|s)$ over the action space $A$.

### 2.2.1 Formal Definition of Policies and Types of Policies

As previously defined, a policy $\pi$ is a probability kernel mapping states to distributions over actions. This definition is both general and powerful, encompassing several crucial special cases that form the basis of reinforcement learning algorithms.

* **General vs. Memoryless Policies**:
    * A **general (or history-dependent) policy** is the most expansive class of strategies, where the action at time $t$ can depend on the entire history up to that point, $h_t = (s_0, a_0, r_1, \dots, s_t)$. Formally, it is a sequence of kernels $\pi_t(da_t | h_t)$. Such policies are necessary for finite-horizon problems or when the state signal is not fully Markovian.

    * A **memoryless (or stationary) policy** is a crucial subclass where the action distribution depends only on the current state $s_t$, not on the time step $t$ or the prior history. It is represented by a single, time-independent kernel $\pi(da|s)$. For the discounted, infinite-horizon MDPs considered here, a foundational result is that an optimal policy can always be found within this simpler class, making the search for solutions tractable.

* **Stochastic vs. Deterministic Policies**:
    * A **stochastic policy** is the general form where for a state $s$, $\pi(\cdot|s)$ is a non-degenerate probability measure over the action space $A$.

    * A **deterministic policy** is a special case where the policy kernel is a Dirac delta measure concentrated on a single action for each state, representable as a measurable function $\mu: S \to A$.
 
For the remainder of this analysis, we will focus on **stationary, stochastic policies**, whose optimality is a direct consequence of the time-independent nature of the Bellman optimality principle.

### 2.2.2 Trajectories, Returns, and the Objective

A policy, when executed by an agent in an environment, induces a sequence of states, actions, and rewards known as a **trajectory** or **history**:

$$
H = (S_0, A_0, R_1, S_1, A_1, R_2, \dots)
$$

This trajectory is a random variable whose distribution is determined by the initial state distribution $\mu$, the policy $\pi$, and the environment's transition kernel $\kappa$.

To evaluate the desirability of a trajectory, we define the **return**, which aggregates the rewards. For an infinite-horizon problem, we use the **discounted return**:

**Definition 2.2.1 (Discounted Return)**

The **discounted return** at time $t$, denoted $G_t$, is the sum of all future rewards, discounted by the factor $\gamma \in [0, 1)$ at each step:

$$
G_t := \sum_{k=0}^{\infty} \gamma^k R_{t+k+1}
$$

The **discount factor** $\gamma$ serves two purposes:

1.  **Mathematical Convergence**: For bounded rewards ($|R_t| \le R_{max}$), it ensures that the infinite sum $G_t$ is a well-defined, finite random variable, since $|G_t| \le \sum_{k=0}^{\infty} \gamma^k R_{max} = \frac{R_{max}}{1-\gamma}$.

2.  **Behavioral Control**: It encodes a preference for immediate rewards over future rewards. A $\gamma$ close to 0 leads to myopic behavior, while a $\gamma$ close to 1 leads to far-sighted behavior.

> **Remark: On the Discount Factor and the Effective Horizon**
> The discount factor $\gamma$  makes future rewards matter less than present ones. If we truncate the return sum after $H$ terms, the error (the "tail" of the sum) is bounded. Assuming rewards are bounded by $R_{max}$, the magnitude of this truncated part is:
>
> $$
> \left\lvert \sum_{k=H}^{\infty} \gamma^k R_{t+k+1} \right\rvert \le \sum_{k=H}^{\infty} \gamma^k R_{max} = \gamma^H \sum_{j=0}^{\infty} \gamma^j R_{max} = \frac{\gamma^H R_{max}}{1 - \gamma}
> $$
>
> We can determine the number of steps $H$ required for this error to be smaller than some tolerance $\varepsilon$. For simplicity, if we assume $R_{max}=1$, we solve for the $H$ that satisfies $\dfrac{\gamma^H}{1 - \gamma} \le \epsilon$:
>
> $$H \geq H_{\gamma, \epsilon}^\ast = \dfrac{\ln\left(\dfrac{1}{\epsilon(1 - \gamma)}\right)}{\ln(1 / \gamma)}
> $$
>
> For any $H$ satisfying this, the optimal action sequence is unlikely to change by considering horizons longer than $H$. This critical value of $H$ is called the **effective horizon**.
>
> Oftentimes, for simplicity, $H_{\gamma , \epsilon}^\ast$ is replaced with the following upper bound (often also called the effective horizon):
>
> $$
> H_{\gamma ,\epsilon} := \dfrac{\ln\left(\dfrac{1}{\epsilon(1 - \gamma)}\right)}{1 - \gamma}
> $$
>
> The relative difference between these two quantities is small when $\gamma$ is close to 1, which is the regime of primary interest (i.e., far-sighted agents).
>
> The discounted setting can sometimes feel arbitrary. Where does $\gamma$ come from? One view is that we first choose an effective horizon $H$ that feels appropriate for the problem, and then work backward to find a $\gamma$ that produces this horizon. A more honest admission is that the discounted objective may not perfectly capture every decision problem. Other objectives exist, such as finite-horizon (discounted or not), total reward (undiscounted infinite-horizon), or average reward. Each has pros and cons. We stick to the discounted objective for now for pedagogical reasons: the underlying mathematics is particularly simple and elegant, and many of the results transfer to other settings with minor modifications.



The ultimate goal of the agent is to select a policy that maximizes the expected discounted return.

**The Reinforcement Learning Objective**

The objective is to find an **optimal policy**, denoted $\pi^*$, which achieves the highest possible expected return from any initial state. 

Formally, for any policy $\pi$, we define its value starting from a state $s$ as $v_\pi(s) = \mathbb{E}[G_t | S_t=s; \pi]$. An optimal policy $\pi^*$ must satisfy:

$$
v_{\pi^\ast}(s) \ge v_{\pi}(s) \quad \text{for all } s \in S \text{ and for all policies } \pi.
$$

The existence and uniqueness of such an optimal stationary policy is a cornerstone theorem of MDP theory.

### 2.2.2a The Vector Space of Value Functions

To apply the full power of linear algebra, we first frame the objects of our analysis within their proper algebraic context. The set of all bounded, real-valued, measurable functions on the state space, which we denote $B(S)$, forms a **vector space** over the field of real numbers $\mathbb{R}$.

* **Vector Addition**: $(v_1 + v_2)(s) := v_1(s) + v_2(s)$
* **Scalar Multiplication**: $(c \cdot v)(s) := c \cdot v(s)$

This vector space can be endowed with the supremum norm, $||v||_\infty = \sup_{s \in S} |v(s)|$, which makes it a **Banach space** (a complete normed vector space). This completeness is the key property that guarantees the convergence of iterative algorithms like Value Iteration.

Within this framework, any **state-value function** $v_\pi$ is not just a function but a **vector** in the space $B(S)$. The goal of policy evaluation is to find this specific vector.

---


### 2.2.3 Value Functions and Bellman's Expectation Equation

To find an optimal policy, we first need a way to precisely evaluate a given policy. This is the role of **value functions**. There are two fundamental types of value functions.

**Definition 2.2.2 (State-Value Function)**

The **state-value function** $v_\pi: S \to \mathbb{R}$ of a policy $\pi$ gives the expected return when starting in state $s$ and following $\pi$ thereafter:

$$
v_\pi(s) := \mathbb E_{\pi} \left[ G_t \mid S_t = s \right] = \mathbb E_{\pi} \left[ \sum_{k=0}^{\infty} \gamma^k R_{t+k+1} \mid S_t = s \right]
$$

This function measures the long-term desirability of a state $s$ under policy $\pi$.

**Definition 2.2.3 (Action-Value Function)**

The **action-value function** $q_\pi: S \times A \to \mathbb{R}$ of a policy $\pi$ gives the expected return after taking action $a$ in state $s$ and subsequently following policy $\pi$:

$$
q_\pi(s, a) := \mathbb E_{\pi} \left[ G_t \mid S_t = s, A_t = a \right] = \mathbb E_{\pi} \left[ \sum_{k=0}^{\infty} \gamma^k R_{t+k+1} \mid S_t = s, A_t = a \right]
$$

This function measures the desirability of taking a specific action $a$ in state $s$.

These two value functions are linked. The value of a state is the expected value of the action-values, averaged over the policy's action choices:
$$v_\pi(s) = \int_A \pi(da'|s) q_\pi(s, a')$$

Crucially, value functions satisfy a fundamental recursive relationship known as the **Bellman Expectation Equation**. By decomposing the return $G_t = R_{t+1} + \gamma G_{t+1}$, we can express the value of the current state in terms of the expected value of the next state.

**Theorem 2.2.1 (Bellman Expectation Equation)**

For a given policy $\pi$, its state-value function $v_\pi$ is the unique fixed point of the **Bellman Expectation Operator**, $T_\pi: B(S) \to B(S)$. This operator is an affine transformation defined as:

$$ (T_\pi v)(s) := r_\pi(s) + \gamma (P_\pi v)(s) $$

where $r_\pi(s) = \int_A \pi(da|s) r(s,a)$ is the expected reward vector and $P_\pi$ is the policy-conditioned Markov operator. The Bellman equation is therefore the operator equation:

$$ v_\pi = T_\pi v_\pi $$

1.  **For the state-value function $v_\pi$**:

$$ v_\pi(s) = \mathbb E_{\pi} [R_{t+1} + \gamma v_\pi(S_{t+1}) | S_t = s] $$

Using the derived expected reward function $r(s,a)$ and state transition kernel $p(ds'|s,a)$, this simplifies to the more common form:

$$
v_\pi(s) = \int_A \pi(da|s) \left( r(s,a) + \gamma \int_S p(ds'|s,a) v_\pi(s') \right)
$$

2.  **For the action-value function $q_\pi$**:

$$
q_\pi(s, a) = \mathbb{E} [R_{t+1} + \gamma v_\pi(S_{t+1}) | S_t = s, A_t = a]
$$

Since the first action $a$ is fixed, the expectation is only over the environment's response. Substituting the definition of $v_\pi$ gives the recursion in terms of $q_\pi$:

$$
q_\pi(s, a) = r(s,a) + \gamma \int_S p(ds'|s,a) \int_A \pi(da'|s') q_\pi(s', a')
$$

These equations establish a self-consistency condition: 

* The value of the current state (or state-action pair) under $\pi$ must equal the expected immediate reward plus the discounted expected value of the next state, assuming the agent continues to follow $\pi$. 

* This relationship is the foundation for nearly all policy evaluation and control algorithms in reinforcement learning.

#### Finite‑state case (matrix form)

When $|S| < \infty$ we may fix the canonical basis of $\mathbb R^{|S|}$ and write  
$$
(I-\gamma P_\pi)v_\pi = r_\pi ,
$$
where $P_\pi$ is the $|S|\times|S|$ transition matrix and $r_\pi$ the reward vector.  

Invertibility of $I-\gamma P_\pi$ follows from $\det(I-\gamma P_\pi)=\chi_{P_\pi}(1/\gamma)\neq0$ for $\gamma\in[0,1)$ (Cayley–Hamilton).  
Thus *policy evaluation* reduces to solving a linear system; Gaussian elimination (H&K Ch. 1) or sparse linear‑solver packages can be used directly.
 

> **Remark: Eigen-analysis of the Bellman Operator**
> The fixed-point equation $v_\pi = r_\pi + \gamma P_\pi v_\pi$ directly connects to the theory of eigenvalues and eigenvectors. If we consider a reward-free process, the equation becomes $v_\pi = \gamma P_\pi v_\pi$, which can be rewritten as:
> $$ P_\pi v_\pi = \left(\frac{1}{\gamma}\right) v_\pi $$
> This shows that the value function $v_\pi$ is an **eigenvector** of the policy-conditioned transition operator $P_\pi$ with corresponding **eigenvalue** $1/\gamma$.
>
> More generally, the convergence of policy evaluation is guaranteed because the operator $T_\pi$ is a **contraction mapping**. This property is equivalent to the **spectral radius** of its linear part, $\gamma P_\pi$, being strictly less than 1. Since the spectral radius of the Markov operator $P_\pi$ is 1, the spectral radius of $\gamma P_\pi$ is $\gamma$, which is in $[0, 1)$. This perspective, grounded in spectral theory, provides a deep understanding of why value iteration converges to a unique solution.
 

### 2.2.4 The State Occupancy Measure: A Dual Perspective

While value functions provide a recursive, state-centric view of a policy's performance, the **state occupancy measure** offers an alternative, holistic perspective. 

It quantifies the expected total discounted time a process spends in each state or state-action pair, effectively transforming the RL problem from a dynamic program into a linear one.

**Definition 2.2.4 (State and State-Action Occupancy Measures)**

Let $\pi$ be a stationary policy and $\mu$ be an initial state distribution.

1.  The **discounted state occupancy measure** $\rho_\mu^\pi$ is a measure on the state space $(S, \mathcal{B}(S))$ defined for any measurable set $B \in \mathcal{B}(S)$ as:

$$
\rho_\mu^\pi(B) := \mathbb E_\mu^\pi \left[ \sum_{t=0}^{\infty} \gamma^t \mathbf 1_{\{S_t \in B\}} \right] = \sum_{t=0}^{\infty} \gamma^t \mathbb P_\mu^\pi(S_t \in B)
$$
This measures the total expected discounted number of visits to the set of states $B$.

2.  The **discounted state-action occupancy measure** $\rho_\mu^\pi$ is a measure on the product space $(S \times A, \mathcal{B}(S) \otimes \mathcal{B}(A))$ defined for any measurable set $C \in \mathcal{B}(S) \otimes \mathcal{B}(A)$ as:

$$
\rho_\mu^\pi(C) := \mathbb E_\mu^\pi \left[ \sum_{t=0}^{\infty} \gamma^t \mathbf 1_{\lbrace (S_t, A_t) \in C \rbrace} \right]
$$

This measure is fundamental because it directly relates to the policy's total expected return.

**Theorem 2.2.2 (Objective as an Inner Product)**

The total expected discounted return of a policy $\pi$ for an initial distribution $\mu$ can be expressed as the integral of the immediate expected reward function $r(s,a)$ with respect to the state-action occupancy measure:

$$
J(\pi) = \mathbb E_\mu^\pi \left[ \sum_{t=0}^{\infty} \gamma^t R_{t+1} \right] = \int_{S \times A} r(s,a) d \rho_\mu^\pi(s,a)
$$

**Proof Sketch:**
The proof relies on a direct application of the Tonelli-Fubini theorem, which allows the exchange of expectation and summation.
$$J(\pi) = \mathbb E_\mu^\pi \left[ \sum_{t=0}^{\infty} \gamma^t r(S_t, A_t) \right] = \sum_{t=0}^{\infty} \gamma^t \mathbb E_\mu^\pi [r(S_t, A_t)]$$
By definition of expectation and the state-action occupancy measure, this is precisely $\int_{S \times A} r(s,a) d\rho_\mu^\pi(s,a)$.

> **Remark: The Discrete Case**
> In the common scenario of finite state and action spaces, the state-action occupancy measure $\rho_\mu^\pi$ can be viewed as a simple table or vector, where for each pair $(s,a)$, the value is:
> $$
> \rho_{\mu}^{\pi}(s,a) = \sum_{t = 0}^{\infty} \gamma^{t} \mathbb P_{\mu}^{\pi}(S_{t} = s,A_{t} = a)
> $$
> In this setting, the integral for the objective function $J(\pi)$ simplifies to a sum, which can be interpreted as an inner product between the occupancy measure and the reward vector:
> $$
> J(\pi) = \sum_{s \in S, a \in A} \rho_{\mu}^{\pi}(s,a) r(s,a) := \langle \rho_{\mu}^{\pi},r\rangle
> $$
> This shows that maximizing the expected return is equivalent to choosing a policy that "stirs" the occupancy measure to maximally align with the reward function. A better alignment results in a higher value for the policy.


A key step in proving the sufficiency of memoryless policies for optimal control is the following result:

**Theorem 2.2.3 (Equivalent Memoryless Policy)**

For any (potentially history-dependent) policy $\pi$ and a start state distribution $\mu \in \mathcal{M}_1(S)$, there exists a memoryless policy $\pi'$ such that their state-action occupancy measures are identical:

$$
\rho_{\mu}^{\pi'} = \rho_{\mu}^{\pi}
$$

This implies they achieve the exact same total return, $J(\pi') = J(\pi)$.

**Proof Hint:**
First, define the state occupancy measure $\tilde \rho_{\mu}^{\pi}(s) := \sum_{a \in A} \rho_{\mu}^{\pi}(s,a)$. 

Then, show that the theorem holds for the stationary policy $\pi'$ defined as follows:

$$
\pi^{\prime}(a \mid s) = \left\lbrace \begin{array}{ll} \dfrac{\rho_{\mu}^{\pi}(s,a)}{\tilde \rho_{\mu}^{\pi}(s)} & \mathrm{if~} \tilde \rho_{\mu}^{\pi}(s) > 0 \newline \text{arbitrary distribution} & \mathrm{otherwise}. \end{array} \right\rbrace
$$

Note that it is crucial that the memoryless policy obtained depends on the start state distribution. 
There exist non-memoryless policies whose value function cannot be reproduced by a *single* memoryless policy across *all* possible start states simultaneously.

This formulation is powerful because the set of all valid state-action occupancy measures forms a convex set defined by a linear constraint. Any such measure $\rho$ induced by a policy must satisfy a "Bellman flow" equation: the total flow into a set of states $B$ must equal the initial flow plus the discounted flow from all other states.

**Bellman Flow Constraint:**
For any state-action measure $\rho$ to be valid (i.e., induced by some policy $\pi$), it must satisfy:

$$
\int_A \rho(B, da) = \mu(B) + \gamma \int_{S \times A} p(B \mid s,a) d\rho(s,a) \quad \text{for all } B \in \mathcal{B}(S)
$$

This constraint is a system of linear equations, where the unknown variable is the measure $\rho$. The reinforcement learning problem can thus be reframed as a linear program, a classic problem in applied linear algebra:

$$
\max_{\rho} \int_{S \times A} r(s,a) d\rho(s,a)
$$

subject to the Bellman flow constraint.

> **Dimension check.** For finite sets the constraint matrix $A$ has rank $|S|$; hence the feasible polytope lives in an affine subspace of dimension $|S||A|-|S|$. Rank–nullity makes explicit how many independent directions a policy can steer occupancy, a fact exploited by occupancy‑measure RL algorithms.
 

> **Remark: Duality in Reinforcement Learning**
> This formulation is the linear programming **dual** to the problem solved by the Bellman optimality equation. The Bellman equation seeks a value function in a function space (a dynamic programming approach), while this method seeks an occupancy measure in a space of measures. This dual view is not only computationally useful but also provides deep theoretical insights, particularly for analyzing policy gradient methods and imitation learning algorithms, where matching occupancy measures is the central goal.

---

### 2.2.5 The Process Measure and Markov Properties

In the study of reinforcement learning and decision-making, we need a solid mathematical framework to describe the entire ongoing interaction between an agent and its environment. This section explains how we can create a single, unique probability rule that governs the whole sequence of states and actions over time.

**The Goal: A Unified Probability Rule for an Entire Timeline**

Imagine an agent (like a robot or a game character) operating in an environment. At every moment, it's in a certain **state** ($S_t$), it takes an **action** ($A_t$), and the environment responds by moving it to a **new state** ($S_{t+1}$). This creates an endless timeline of events: $S_0, A_0, S_1, A_1, S_2, \dots$.
 
We need a way to answer questions like, "What is the probability of *this specific sequence* of events happening?" To do this, we need a single, overarching probability rule, which we call the **Process Measure**. This measure, denoted $\mathbb{P}_\mu^\pi$, is the master rulebook that assigns a probability to any possible trajectory the agent could experience.
 

1.  **Initial Distribution ($\mu$)**: This is the rule that determines the starting point. It tells us the probability of the agent beginning in any particular state. For example, it might state there's a 100% chance of starting at 'Position A' or a 50/50 chance of starting at 'Position A' or 'Position B'.
2.  **Policy ($\pi$)**: This is the agent's strategy or "brain." Given its current state, the policy tells us the probability of choosing any possible action. For example, if in state 'Crossroads', the policy might be "go left 70% of the time, go right 30% of the time."
3.  **Environment Dynamics ($p$)**: These are the "laws of physics" for the environment. This rule tells us the probability of transitioning to a new state, given the agent's current state and the action it just took. For instance, if the agent is in state 'Icy Patch' and takes the action 'Step Forward', the dynamics might be "90% chance of moving to 'Safe Ground', 10% chance of transitioning to 'Fallen Down'."
 
The challenge is to combine these three separate rules into one single measure, $\mathbb{P}_\mu^\pi$, that consistently describes the whole process. The mathematical tool that allows us to do this is the **Ionescu-Tulcea Extension Theorem**.
 
---

### Constructing the Process Measure (Theorem 2.2.3)

The Ionescu-Tulcea theorem guarantees that if we can define the probability of each step in the sequence based only on the history so far, then there is one and only one probability measure for the entire infinite sequence. Our construction relies on three specific conditions that define the structure of this agent-environment interaction.
 
1.  **Initialization**: The probability that the process starts in a particular region of states, $B$, is given directly by our initial distribution, $\mu$.
    $$
    \mathbb{P}_\mu^\pi(S_0 \in B) = \mu(B)
    $$
    In simple terms: The chance of starting somewhere is whatever the starting rule says it is.
 
2.  **Action Selection**: The probability of choosing an action from a set of possible actions, $C$, depends *only* on the current state, $S_t$. It doesn't matter how we got to $S_t$. The decision is made solely based on the agent's policy, $\pi$.
    $$
    \mathbb P_\mu^\pi(A_t \in C \mid S_0, A_0, \dots, S_t) = \pi(C \mid S_t)
    $$
    The probability of action $A_t$ being in the set $C$, given the entire history up to state $S_t$, is simply the probability dictated by the policy $\pi$ at state $S_t$.

3.  **State Transition**: The probability of the next state, $S_{t+1}$, falling into a region $B$ depends *only* on the current state $S_t$ and the current action $A_t$. The past history before this point is irrelevant. This is determined by the environment's dynamics, $p$.
    $$
    \mathbb P_\mu^\pi(S_{t+1} \in B \mid S_0, A_0, \dots, S_t, A_t) = p(B \mid S_t, A_t)
    $$
    The probability of the next state $S_{t+1}$ being in $B$, given the whole history up to the action $A_t$, is simply the probability given by the environment's rules $p$ for the outcome of action $A_t$ in state $S_t$.
 

Note: The term ($\mathbb P_\mu^\pi$-a.s.) stands for "almost surely." It's a technical requirement in probability theory that means the statement holds true except for a set of outcomes that have a total probability of zero. For all practical purposes, you can read it as "this is always true."
 
The Ionescu-Tulcea theorem takes these step-by-step rules and "extends" them to uniquely define the single process measure, $\mathbb{P}_\mu^\pi$, over the space of all possible infinite trajectories.
 
---

### The Resulting Markov Property

A direct and crucial consequence of building our process measure this way is that the resulting system has the **Markov Property**.
 

The Markov Property states that **the future is independent of the past, given the present**.

In our context, the "history" up to time $t$ is the sequence of all states and actions seen so far, $(S_0, A_0, \dots, S_t)$. This history is formally called the **natural filtration** and is denoted $\mathcal{F}_t$. The "present" is just the current state, $S_t$.

The Markov Property means that to predict what happens next (the probability of $S_{t+1}$), you only need to know the current state $S_t$. You gain no extra predictive power from knowing the full history of how the agent arrived at $S_t$.
 
Mathematically, this is expressed as:

$$
\mathbb P_\mu^\pi(S_{t+1} \in B \mid \mathcal F_t) = \mathbb P_\mu^\pi(S_{t+1} \in B \mid S_t)
$$

This equation says: "The probability of the next state landing in region $B$, given the *entire history* $\mathcal{F}_t$, is exactly the same as the probability given *only the current state* $S_t$."
 
---

### A Simplified View: The Policy-Conditioned Kernel

To make things neater, we can combine the agent's decision-making and the environment's response into a single new function, $p^\pi$. This is called the **policy-conditioned transition kernel**.

$$p^\pi(ds'|s) := \int_A p(ds'|s, a) \pi(da|s)$$

Let's break this down:
* This formula calculates the overall probability of moving from a state $s$ to a next state $s'$.
* It does this by considering every possible action $a$ the agent could take.
* For each action $a$, it multiplies the probability of choosing that action ($\pi(da|s)$) by the probability of that action leading to state $s'$ ($p(ds'|s, a)$).
* The integral sign $\int_A$ simply sums up these possibilities over all actions.

With this tool, the Markov property becomes even cleaner to write:
$$\mathbb P_\mu^\pi(S_{t+1} \in B \mid \mathcal F_t) = p^\pi(B|S_t)$$
This says the probability of transitioning into a set of states $B$ from our current state $S_t$ is given directly by our combined agent-environment rule, $p^\pi$.

> **Spectral viewpoint (finite $S$).** Write $P^\pi$ for the matrix of $p^\pi$.  
> Via the Schur/Jordan form (H&K Ch. 7) we have $(P^\pi)^t=U\operatorname{diag}(\lambda_i^t)U^{-1}$.  
> Mixing speed is governed by the second‑largest $|\lambda_i|$, giving geometric‑rate bounds that match classical coupling results.
 

Finally, using this framework, the probability density for any specific finite sequence of events (like $s_0, a_0, s_1, a_1, \dots, s_t$) can be calculated by simply multiplying the probabilities of each step in the chain:

$$
\mathbb{P}_\mu^\pi(\text{history}) = 
  \underset{\text{Prob. of start}}{\mu(ds_0)}
  \cdot
  \underset{\text{Prob. of 1st action}}{\pi(da_0 \mid s_0)}
  \cdot
  \underset{\text{Prob. of 1st transition}}{p(ds_1 \mid s_0, a_0)}
  \cdot
  \underset{\text{Prob. of 2nd action}}{\pi(da_1 \mid s_1)}
  \cdot
  \dots
$$

This shows how the entire probability measure is built from the ground up from our three core ingredients.

---

Given an initial distribution $\mu$ on the state space $S$ and a policy $\pi$, we can now formally establish the existence of a unique probability measure governing the entire evolution of the agent-environment interaction. This is a direct and critical application of the Ionescu-Tulcea Extension Theorem presented earlier.

**Theorem 2.2.3 (Existence of the Process Measure)**

1.  **Initialization**: The probability of the initial state $S_0$ being in $B$ is determined by $\mu$.
$$
\mathbb{P}_\mu^\pi(S_0 \in B) = \mu(B)
$$

2.  **Action Selection**: The conditional probability of selecting action $A_t$ from $C$, given the history up to time $t$, is determined solely by the policy $\pi$ applied to the current state $S_t$.
$$
\mathbb P_\mu^\pi(A_t \in C \mid S_0, A_0, \dots, S_t) = \pi(C \mid S_t) \quad (\mathbb P_\mu^\pi\text{-a.s.})
$$

3.  **State Transition**: The conditional probability of the next state $S_{t+1}$ being in $B$, given the history up to time $t$ and the action $A_t$, is determined by the MDP's state transition kernel $p$.
$$
\mathbb P_\mu^\pi(S_{t+1} \in B \mid S_0, A_0, \dots, S_t, A_t) = p(B \mid S_t, A_t) \quad (\mathbb P_\mu^\pi\text{-a.s.})
$$

**Justification via Ionescu-Tulcea (Theorem 1.7.3):**

This measure is constructed by defining a sequence of kernels on the growing product spaces. We start with the initial measure \$\mu\$ on \$S\$. Then we define a sequence of kernels:

$\kappa_1(da_0 \mid s_0) = \pi(da_0 \mid s_0)$ followed by $\kappa_2(ds_1 \mid s_0, a_0) = p(ds_1 \mid s_0, a_0)$ followed by $\kappa_3(da_1 \mid s_0, a_0, s_1) = \pi(da_1 \mid s_1)$ and so on. 

The Ionescu–Tulcea theorem guarantees that this sequence of consistent one-step-ahead conditional distributions uniquely defines the measure $\mathbb{P}_{\mu}^{\pi}$ over the infinite-horizon path space. 

The uniqueness of this measure ensures that any two constructions satisfying these conditions will produce identical joint distributions for the sequence of states and actions.


The probability density for a specific finite history (a cylinder set) $H_t = (s_0, a_0, \dots, s_{t-1}, a_{t-1}, s_t)$ is given by the product of the individual kernel densities:

$$
\mathbb P_\mu^\pi (S_0 \in ds_0, A_0 \in da_0, \dots, S_t \in ds_t) = \mu(ds_0) \prod_{k=0}^{t-1} \left( \pi(da_k|s_k) p(ds_{k+1}|s_k, a_k) \right)
$$
 
The resulting stochastic process $(S_0, A_0, S_1, A_1, \ldots)$ on the probability space $(H, F, \mathbb P_{\mu}^{\pi})$ has the **Markov Property** as a direct consequence of its construction.

The history of the process up to time $t$ is captured by the **natural filtration**, $F_t = \sigma(S_0, A_0, \ldots, S_t)$. The Markov property then formally states that the future is independent of the past, given the present:

$$
\mathbb P_\mu^\pi(S_{t+1} \in B \mid \mathcal F_t) = \mathbb P_\mu^\pi(S_{t+1} \in B \mid S_t) = \int_A \pi(da|S_t) p(B|S_t, a) \quad (\mathbb P_\mu^\pi\text{-a.s.})
$$

---

A random variable $\tau: \Omega \to \mathbb{N} \cup \{\infty\}$ is a **stopping time** with respect to the filtration $(\mathcal{F}_t)$ if the event $\{\tau = t\} \in \mathcal{F}_t$ for all $t \in \mathbb{N}$. 

Intuitively, the decision to "stop" at time $t$ depends only on the history up to time $t$.

The process has the **Strong Markov Property**, a crucial strengthening, if the Markov property holds not just at fixed times $t$, but also at random **stopping times** $\tau$. 

For any almost surely finite stopping time $\tau$ and any bounded, measurable function $g: S \to \mathbb{R}$:

$$
\mathbb E_\mu^\pi [g(S_{\tau+1}) \mid \mathcal F_\tau] = \mathbb E_\mu^\pi [g(S_{\tau+1}) \mid S_\tau] \quad (\mathbb P_\mu^\pi\text{-a.s.})
$$

where $\mathcal F_\tau$ is the stopping time $\sigma$-algebra. 

This property states that the process probabilistically "restarts" from the state $S_\tau$, regardless of how the random stopping time $\tau$ was reached. This is essential for analyzing event-triggered strategies (e.g., the value of the state upon first hitting a goal region).


> **Remark: The Role of the Feller Property**
> The Strong Markov Property is not automatic. It is a powerful result that follows from the underlying structure of the MDP, specifically the regularity of the transition kernel. Kernels on Polish spaces often have the **Feller property**, which is the key ingredient needed.
>
> A kernel has the **Feller property** if its corresponding Markov operator (Definition 1.6.3) maps the space of bounded, *continuous* functions to itself. That is, if $g$ is a bounded and continuous function, then the function $(Pg)(x) = \int g(y) \kappa(x, dy)$ is also bounded and continuous in $x$.
>
> **Proof Sketch: Why Feller implies Strong Markov**
> The simple Markov property, $\mathbb E [g(S_{t+1}) \mid \mathcal F_t] = (Pg)(S_t)$, holds for any fixed (deterministic) time $t$. The challenge is to show this relationship holds for a random stopping time $\tau$.
>
> 1.  **Approximation**: The proof hinges on approximating the continuous function $g$ and the stopping time $\tau$. First, for any bounded, measurable function $g$, the Strong Markov Property holds. However, the proof is more intuitive for continuous $g$. The core idea is to approximate the stopping time $\tau$ by a sequence of discrete-valued stopping times (e.g., $\tau_n = \lceil 2^n \tau \rceil / 2^n$). For each discrete $\tau_n$, the Strong Markov property can be shown to hold by piecing together the simple Markov property at each possible discrete time step.
>
> 2.  **The Role of Continuity (Feller Property)**: The crucial step is taking the limit as $n \to \infty$, so $\tau_n \to \tau$. We need to show that $\mathbb E [g(S_{\tau_n+1}) \mid \mathcal F_{\tau_n}] \to \mathbb E [g(S_{\tau+1}) \mid \mathcal F_\tau]$ and that the right-hand side, $(Pg)(S_{\tau_n})$, also converges to the correct limit, $(Pg)(S_\tau)$.
>
> 3.  **Convergence**: As $n \to \infty$, we have $S_{\tau_n} \to S_\tau$ almost surely. The **Feller property** guarantees that the operator $P$ maps continuous functions to continuous functions, meaning $Pg$ is continuous. This continuity is precisely what's needed to ensure that $S_{\tau_n} \to S_\tau$ implies $(Pg)(S_{\tau_n}) \to (Pg)(S_\tau)$. Without the Feller property, the mapping $s \mapsto (Pg)(s)$ could be discontinuous. In that case, knowing the specific path leading to $S_\tau$ (i.e., the history in $\mathcal F_\tau$) could provide information about which side of a discontinuity the process is on, thus breaking the property that the process "restarts" based only on the current state $S_\tau$.

> For the policy-conditioned process to be Feller, certain regularity assumptions must be placed on the MDP's primitive components. For instance, the property holds if the environment kernel $p(ds'|s,a)$ and the policy kernel $\pi(da|s)$ are both weakly continuous in their respective arguments.
> This continuity is what bridges the gap between fixed and random times. For a fixed time `t`, the Markov property is a direct consequence of the process's construction. For a random stopping time `τ`, we must be sure that the history leading up to the stop does not provide extra information. If the kernel were not Feller, the transition probabilities could be discontinuous in the state. An agent could then exploit this discontinuity; the specific path taken to arrive at state $S_\tau$ would inform it about a likely "jump" in transition dynamics, breaking the "restart" property. Feller continuity ensures that the future evolution depends smoothly on the present state $S_\tau$, regardless of the history of how that state was reached.


> From the perspective of operator theory, the Feller property means the Markov operator $P$ leaves the subspace of continuous functions, $C(S)$, **invariant** (i.e., $P(C(S)) \subseteq C(S)$). The Strong Markov property relies on the continuity of the operator when restricted to this subspace. This algebraic viewpoint clarifies why the topological property of the kernel (weak continuity) leads to the powerful probabilistic result of the process restarting at random times.
 
> **Remark: Beyond Borel Measurability**
> This entire framework relies on Borel measurability. However, in more advanced analyses, key sets (e.g., the set of states where one policy is better than another) are not guaranteed to be Borel but rather **analytic**. An analytic set is a continuous image of a Borel set.
>
> A cornerstone of advanced theory is that analytic subsets of standard Borel spaces are **universally measurable**, meaning they are measurable with respect to the completion of any probability measure. This wider class of sets and functions is often necessary to establish the existence of optimal value functions and policies under the most general conditions, providing a deeper layer of rigor than a strict adherence to Borel sets allows.
> The primary motivation for this extension arises when proving the existence of optimal policies under the most general conditions. For instance, a key step in policy iteration involves analyzing the set of states where one policy is superior to another, i.e., the set $U = \{s \in S \mid v_{\pi_1}(s) > v_{\pi_2}(s)\}$. While the value functions $v_\pi$ are guaranteed to be measurable, the optimal value function $v^\ast = \sup_{\pi} v_\pi$ is not guaranteed to be Borel-measurable when the supremum is taken over an uncountable number of policies. It is, however, **analytic**. This means the set $\{s \in S \mid v^\ast(s) > c\}$ may not be a Borel set, which would prevent us from verifying the measurability of policies derived from it. Universal measurability guarantees that these crucial analytic sets are still part of a valid, extended $\sigma$-algebra, ensuring that all integrals in the analysis remain well-defined and the theory holds.
> See Appendix B.1 for the precise theorem.

---

# Appendix B: Descriptive Set Theory Capsule

Throughout, **Polish space** means a separable complete metric space equipped with its Borel σ-algebra $\mathcal{B}(E)$.

A **standard Borel space** is a measurable space measurably isomorphic to some $(E,\mathcal{B}(E))$. All maps are assumed measurable unless explicitly strengthened (e.g. "continuous"). References: [Kechris 1995], [Srivastava 1998].

## B.1 Analytic Sets and Universal Measurability

**Definition B.1.1 (Analytic / Souslin set).**  
Let $E$ be a Polish space. A subset $A \subseteq E$ is called **analytic** (or **Souslin**) if any of the following equivalent conditions holds:

1. There exist a Polish space $S$, a Borel set $B \subseteq S$, and a *Borel-measurable* map $f:S \to E$ such that $A = f(B)$.

2. There exists a Borel set $C \subseteq S \times E$ with $A = \pi_E[C]$, the *projection* of $C$ onto $E$.

3. $A$ is the image of Baire space $\mathbb{N}^{\mathbb{N}}$ under a continuous map.

The class of analytic sets strictly contains the Borel σ-algebra.

Analytic sets are a broader class than Borel sets, containing all Borel sets and also sets that are not Borel. They arise naturally in many constructions in measure theory and stochastic processes, particularly when dealing with projections or optimizing over uncountable spaces.

**Example.** The classical **Souslin set**

$S = \lbrace x \in \mathbb{R} : \exists y \in \mathbb{R} \text{ such that } (x,y) \in C\rbrace,$
where $C \subseteq \mathbb{R}^2$ is a Borel yet carefully constructed by Souslin (1917), is analytic but *not* Borel. Such projections arise routinely when one eliminates "hidden" variables in stochastic models or optimizations over uncountable spaces.

**Theorem B.1.2 (Universal measurability of analytic sets).**  
For every Polish space $E$ and every analytic $A \subseteq E$, $A$ is *universally measurable*; i.e. for every probability measure $\mu$ on $E$ there exist Borel sets $B_1 \subseteq A \subseteq B_2$ with $\mu(B_2 \setminus B_1) = 0$.

**Proof sketch.**  
The **Lusin Separation Theorem** provides Borel sets $B_1, B_2$ with $B_1 \subseteq A \subseteq B_2$ and $B_1, B_2$ disjoint from any Borel set disjoint from $A$. Applying the **Choquet capacity theorem** (or equivalently the inner/outer regularity of capacities) yields, for every $\varepsilon > 0$, closed $K \supseteq B_1$ and open $U \supseteq B_2$ such that $\mu(U \setminus K) < \varepsilon$. Letting $\varepsilon \downarrow 0$ gives the desired Borel sandwich.

## B.2 Structure Theorems for Standard Borel Spaces

**Theorem B.2.1 (Borel-Isomorphism).**  
Let $S, S'$ be uncountable standard Borel spaces. There exists a bijection $f: S \to S'$ such that both $f$ and $f^{-1}$ are measurable. Consequently every uncountable standard Borel space is measurably isomorphic to $(\mathbb{R}, \mathcal{B}(\mathbb{R}))$. 

**Caution:** this is an isomorphism of *measurable structures*, not of topologies; no continuity is implied.

**Theorem B.2.2 (Kuratowski–Ryll-Nardzewski measurable-selection).**  
Let $(E, \Sigma_E)$ be a measurable space, $F$ a Polish space, and $\Phi: E \to 2^F$ such that each $\Phi(x)$ is *non-empty and closed*. If the graph
$\text{Graph}(\Phi) = \lbrace(x,y) \in E \times F : y \in \Phi(x)\rbrace$
is $\Sigma_E \otimes \mathcal{B}(F)$-measurable then a (single-valued) measurable *selection* $f: E \to F$ exists with $f(x) \in \Phi(x)$ for all $x$.

Variants cover analytic-valued multifunctions and yield measurable optimal policies in stochastic control.

## B.3 Radon and Tight Measures on Polish Spaces

**Definition B.3.1 (Radon measure).**  
A Borel measure $\mu$ on a Hausdorff topological space $E$ is **Radon** if:

(i) $\mu$ is finite on compact sets (*locally finite*),

(ii) *inner regular*: $\mu(A) = \sup\lbrace\mu(K) : K \subseteq A, K \text{ compact}\rbrace$,

(iii) *outer regular*: $\mu(A) = \inf\lbrace\mu(U) : A \subseteq U, U \text{ open}\rbrace$

for all Borel $A$.

**Proposition B.3.2.**  
Every finite Borel measure on a Polish space is Radon *and* **tight**:
$$\forall \varepsilon > 0 \quad \exists K \text{ compact such that } \mu(E \setminus K) < \varepsilon.$$

**Outline of proof.**  
Polish spaces are **Lindelöf** and **second-countable**; applying Urysohn's metrization plus regularity of the Borel σ-algebra gives outer regularity. Inner regularity follows from tightness of finite measures on metric spaces (Portmanteau theorem). Conversely, a Radon measure is automatically tight. These facts underpin **Prokhorov's theorem**, ensuring sequential compactness of probability measures under weak topology—vital for existence proofs of optimal controls or invariant measures.
