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

**A1** **Non-emptineess**: $E \in \Sigma_E$

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

4his is the smallest $\sigma$-algebra on $E$ that contains all the open sets. The elements of $\mathcal{B}(E)$ are called **Borel sets** or **Borel-measurable sets**. A space equipped with its Borel $\sigma$-algebra, $(E, \mathcal{B}(E))$, is a **Borel space**.

For example, the standard Borel $\sigma$-algebra on the real numbers, $\mathcal{B}(\mathbb{R})$, is generated by the collection of all open intervals. Similarly, for $\mathbb{R}^n$, the Borel sets are generated by the collection of all open rectangles.
 

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

Dynkin's $\pi-\lambda$ Theorem is important to ensuring the existence of a unique and well-defined probability measure over the infinite seqeunces of states and actions that constitutes the process's trajectories, which leads to the Ionescu-Tulcea Theorem.

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

> Without this condition, fundamental quantities like the expected reward $r(s,a)$ or the state-value function $v_\pi(s)$ might not be well-defined, as their definitions rely on integrals that require measurable integrands.

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

The value $(Pg)(x)$ is the **expected value** of the function $g$ after a one-step transition, given that the process starts at point $x$. The expectation is calculated with respect to the probability measure $\kappa(x, \cdot)$ on the target space $F$.

> **A Note on Notation**: The expression $\kappa(x, dy)$ is standard measure-theoretic notation. It denotes that the integral is taken with respect to the variable $y$, over the target space $F$, using the specific measure $\kappa(x, \cdot)$. For a fixed $x$, $\kappa(x, \cdot)$ is a probability measure on $F$.


> **Remark: Duality of Kernels**. The Markov operator $P$ represents the kernel's "pull-back" action on functions. There is a dual "push-forward" action on measures. Given a measure $\mu$ on $(E, \Sigma_E)$, the kernel produces a new measure on $(F, \Sigma_F)$, denoted $\mu P$, defined for any $B \in \Sigma_F$ as $(\mu P)(B) := \int_E \kappa(x, B) \mu(dx)$. This describes how an initial distribution $\mu$ evolves. This dual view of acting on functions (expectations) and measures (distributions) is a cornerstone of the theory.

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

The ultimate goal of the agent is to select a policy that maximizes the expected discounted return.

**The Reinforcement Learning Objective**

The objective is to find an **optimal policy**, denoted $\pi^*$, which achieves the highest possible expected return from any initial state. 

Formally, for any policy $\pi$, we define its value starting from a state $s$ as $v_\pi(s) = \mathbb{E}[G_t | S_t=s; \pi]$. An optimal policy $\pi^*$ must satisfy:

$$
v_{\pi^\ast}(s) \ge v_{\pi}(s) \quad \text{for all } s \in S \text{ and for all policies } \pi.
$$

The existence and uniqueness of such an optimal stationary policy is a cornerstone theorem of MDP theory.

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

For a given policy $\pi$, its value functions $v_\pi$ and $q_\pi$ are the unique bounded solutions to the following system of integral equations:

1.  **For the state-value function $v_\pi$**:

$$
v_\pi(s) = \mathbb E_{\pi} [R_{t+1} + \gamma v_\pi(S_{t+1}) | S_t = s]
$$

Expanding this using the unified kernel $\kappa$ and the policy $\pi$ yields:

$$
v_\pi(s) = \int_A \pi(da|s) \int_{S \times \mathbb R} (r' + \gamma v_\pi(s')) \kappa(ds' dr' | s, a)
$$

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
The proof relies on a direct application of the Tonelli-Fubini theorem.

$$
J(\pi) = \mathbb E_\mu^\pi \left[ \sum_{t=0}^{\infty} \gamma^t r(S_t, A_t) \right] = \sum_{t=0}^{\infty} \gamma^t \mathbb E_\mu^\pi [r(S_t, A_t)]
$$

By definition of expectation and the state-action occupancy measure, this is precisely $\int_{S \times A} r(s,a) d\rho_\mu^\pi(s,a)$.

This formulation is powerful because the set of all valid state-action occupancy measures forms a convex set defined by a linear constraint. 

Any such measure $\rho$ induced by a policy must satisfy a "Bellman flow" equation: the total flow into a set of states $B$ must equal the initial flow plus the discounted flow from all other states.

**Bellman Flow Constraint:** 

For any state-action measure $\rho$ to be valid (i.e., induced by some policy $\pi$), it must satisfy:
$$
\int_A \rho(B, da) = \mu(B) + \gamma \int_{S \times A} p(B|s,a) d\rho(s,a) \quad \text{for all } B \in \mathcal{B}(S)
$$

The reinforcement learning problem can thus be reframed as a linear program:
$$\max_{\rho} \int_{S \times A} r(s,a) d\rho(s,a)$$
subject to the Bellman flow constraint.

> **Remark: Duality in Reinforcement Learning**
> This formulation is the linear programming **dual** to the problem solved by the Bellman optimality equation. The Bellman equation seeks a value function in a function space (a dynamic programming approach), while this method seeks an occupancy measure in a space of measures. This dual view is not only computationally useful but also provides deep theoretical insights, particularly for analyzing policy gradient methods and imitation learning algorithms, where matching occupancy measures is the central goal.

---

### 2.2.5 The Process Measure and Markov Properties

Given an initial distribution $\mu$ on $S$ and a policy $\pi$, we can define the state-to-state dynamics kernel $p^\pi(ds'|s) := \int_A p(ds'|s, a) \pi(da|s)$.

Given an initial state distribution $\mu$ and a policy $\pi$, the **Ionescu-Tulcea Extension Theorem (1.7.3)** provides the mechanism to construct a single, unique probability measure for the entire stochastic process. 

This measure, denoted $\mathbb{P}_\mu^\pi$, is defined over the canonical path space of infinite trajectories $H = (S \times A)^{\mathbb{N}}$.

The construction is governed by three fundamental conditions that define the process structure:

1.  **Initialization**: The process starts according to the initial distribution $\mu$. The probability that the initial state $S_0$ falls within a measurable set $B \subseteq S$ is precisely $\mathbb{P}_\mu^\pi(S_0 \in B) = \mu(B)$.

2.  **Recursive Evolution**: The evolution from state to state is determined by a policy-conditioned transition kernel, $p^\pi$. This kernel is formed by integrating the environment's dynamics against the policy's action choices:
    $$p^\pi(ds'|s) := \int_A p(ds'|s, a) \pi(da|s)$$
    The probability of the next state, $S_{t+1}$, depends only on the current state, $S_t$, according to this kernel.
 
3.  **Unique Extension**: The theorem guarantees that $\mathbb{P}_\mu^\pi$ is the **unique measure** on the infinite path space that is consistent with the initial distribution $\mu$ and the recursive kernel $p^\pi$ for all time steps.

The probability of a specific finite history (a cylinder set) $H_t = (s_0, a_0, \dots, s_{t-1}, a_{t-1}, s_t)$ is given by integrating over the sequence of kernels:

$$
\mathbb P_\mu^\pi (S_0 \in ds_0, \dots, S_t \in ds_t) = \mu(ds_0) \left( \prod_{k=0}^{t-1} \pi(da_k|s_k) p(ds_{k+1}|s_k, a_k) \right)
$$

This explicit construction for cylinder sets forms the basis that the Ionescu-Tulcea theorem extends to the entire infinite-horizon $\sigma$-algebra $\mathcal{F}$.



The resulting stochastic process $(S_0, A_0, S_1, A_1, \dots)$ on the probability space $(H, \mathcal F, \mathbb P_\mu^\pi)$ has the **Markov Property** as a direct consequence of its construction. 

The history of the process up to time $t$ is captured by the **natural filtration**, $\mathcal F_t = \sigma(S_0, A_0, \dots, S_t)$. 

The Markov property then formally states that the future is independent of the past given the present:
$$
\mathbb P_\mu^\pi(S_{t+1} \in B \mid \mathcal F_t) = \mathbb P_\mu^\pi(S_{t+1} \in B \mid S_t) = p^\pi(B|S_t) \quad (\mathbb P_\mu^\pi\text{-a.s.})
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
> 1.  **Approximation**: The core idea is to approximate a stopping time $\tau$ from above by a sequence of discrete-valued stopping times, $\tau_n = \lceil 2^n \tau \rceil / 2^n$. The event $\{\tau = t\}$ is $\mathcal F_t$-measurable, but for a general stopping time, the conditional expectation $\mathbb E [g(S_{\tau+1}) \mid \mathcal F_\tau]$ is tricky.
> 2.  **Simple Case**: For a fixed time $t$, the simple Markov property holds: $\mathbb E [g(S_{t+1}) \mid \mathcal F_t] = (Pg)(S_t)$.
> 3.  **Martingale Connection**: One can construct a martingale related to the process. For a Feller process, the function $v(s) = (Pg)(s)$ is continuous. The process $M_t = v(S_t) - \sum_{k=0}^{t-1} (Pv)(S_k)$ is related to a martingale. Doob's Optional Stopping Theorem states that under certain conditions, the expectation of a martingale is conserved at stopping times.
> 4.  **Continuity is Key**: When we take the limit as $\tau_n \to \tau$, the continuity of $g$ and, crucially, the continuity of $Pg$ (guaranteed by the Feller property) ensure that the expectation converges correctly: $\mathbb E [g(S_{\tau_n}) \mid \mathcal F_{\tau_n}] \to \mathbb{E}[g(S_\tau) | \mathcal F_\tau]$. Without Feller continuity, $S_{\tau_n} \to S_\tau$ would not imply $(Pg)(S_{\tau_n}) \to (Pg)(S_\tau)$, and the proof would fail. The specific path taken to arrive at $S_\tau$ would provide extra information, breaking the "restart" property. Feller continuity ensures that the future evolution depends smoothly on the present state $S_\tau$, regardless of the history of how that state was reached.

> For the policy-conditioned process to be Feller, certain regularity assumptions must be placed on the MDP's primitive components. For instance, the property holds if the environment kernel $p(ds'|s,a)$ and the policy kernel $\pi(da|s)$ are both weakly continuous in their respective arguments.
> This continuity is what bridges the gap between fixed and random times. For a fixed time `t`, the Markov property is a direct consequence of the process's construction. For a random stopping time `τ`, we must be sure that the history leading up to the stop does not provide extra information. If the kernel were not Feller, the transition probabilities could be discontinuous in the state. An agent could then exploit this discontinuity; the specific path taken to arrive at state $S_\tau$ would inform it about a likely "jump" in transition dynamics, breaking the "restart" property. Feller continuity ensures that the future evolution depends smoothly on the present state $S_\tau$, regardless of the history of how that state was reached.
 


> **Remark: Beyond Borel Measurability**
> This entire framework relies on Borel measurability. However, in more advanced analyses, key sets (e.g., the set of states where one policy is better than another) are not guaranteed to be Borel but rather **analytic**. An analytic set is a continuous image of a Borel set.
>
> A cornerstone of advanced theory is that analytic subsets of standard Borel spaces are **universally measurable**, meaning they are measurable with respect to the completion of any probability measure. This wider class of sets and functions is often necessary to establish the existence of optimal value functions and policies under the most general conditions, providing a deeper layer of rigor than a strict adherence to Borel sets allows.
> The primary motivation for this extension arises when proving the existence of optimal policies under the most general conditions. For instance, a key step in policy iteration involves analyzing the set of states where one policy is superior to another, i.e., the set $U = \{s \in S \mid v_{\pi_1}(s) > v_{\pi_2}(s)\}$. While the value functions $v_\pi$ are guaranteed to be measurable, the optimal value function $v^* = \sup_{\pi} v_\pi$ is not guaranteed to be Borel-measurable when the supremum is taken over an uncountable number of policies. It is, however, **analytic**. This means the set $\{s \in S \mid v^*(s) > c\}$ may not be a Borel set, which would prevent us from verifying the measurability of policies derived from it. Universal measurability guarantees that these crucial analytic sets are still part of a valid, extended $\sigma$-algebra, ensuring that all integrals in the analysis remain well-defined and the theory holds.
