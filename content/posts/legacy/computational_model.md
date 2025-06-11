---
date: 2025-06-10
title: "What is a computational model?"
summary: "errata"
category: Tutorial 
author: "Author: Bryan Chan with ChatGPT" 
hero: /assets/images/hero3.png
image: /assets/images/card3.png
---

A **computational model** is an abstract, formally specified representation of a real or hypothetical process that maps well‑defined **inputs** through a sequence of **state transitions** (governed by explicit **rules or equations**) to produce **outputs**, all within stated **resource constraints** (time, memory, precision).

Mathematically, a basic model can be written as a quintuple

$$
M = \langle S, I, O, \delta, \tau\rangle
$$

| Symbol   | Meaning                                                                   | Typical instantiation                                                                      |
| -------- | ------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------ |
| $S$      | Set of admissible states                                                  | tape configurations of a Turing machine, agent attribute vectors, discretized field values |
| $I$      | Input specification                                                       | binary string, initial conditions, dataset features                                        |
| $O$      | Output space                                                              | accept/reject, predicted quantities, visualization frames                                  |
| $\delta$ | Transition or update function $S \times I \to S$                          | next‑state function, differential equation integrator, neural‑network layer                |
| $\tau$   | Termination/observation criterion $S \to \{\text{halt},\text{continue}\}$ | halting state, fixed iteration budget, convergence threshold                               |

Essential characteristics

* **Formal semantics** – every step is unambiguously defined, enabling reproducibility and analysis.
* **Abstraction level** – omits irrelevant real‑world details yet retains causal structure crucial for the question at hand.
* **Resource accounting** – explicit or implicit bounds let us reason about feasibility (e.g., *O(n²)* time).
* **Deterministic vs. stochastic dynamics** – the transition rule may include randomness (Monte Carlo, agent heuristics).

This definition is broad enough to cover classical automata in theoretical computer science **and** high‑resolution climate simulations executed on supercomputers—both abide by the same input‑state‑rule‑output schema.

---

### 1 Do neural networks (NNs) and cellular automata (CAs) fit the definition of a computational model?

**Yes.**
Both systems can be expressed as formally specified, step‑wise state‑transition mechanisms that map inputs to outputs under explicit resource constraints.  They therefore instantiate the generic quintuple

$$
M=\langle S,I,O,\delta,\tau\rangle,
$$

where

* $S$ – admissible states
* $I$ – allowed inputs (subset of $S$ or encoded separately)
* $O$ – observable outputs (subset of $S$ or a derived space)
* $\delta$ – transition/update function
* $\tau$ – termination/observation rule

---

### 2 Mathematical embeddings

#### 2.1 Feed‑forward (static) neural network

Let a depth‑$L$ feed‑forward NN have layer widths $(d_0,\dots,d_L)$, weight matrices $W_\ell\in\mathbb R^{d_{\ell}\times d_{\ell-1}}$, biases $b_\ell\in\mathbb R^{d_\ell}$ and activation $\phi_\ell{:}\mathbb R^{d_\ell} \to \mathbb R^{d_\ell}$.

| Component               | Realisation                                                                             |
| ----------------------- | --------------------------------------------------------------------------------------- |
| **States $S$**          | $\bigsqcup_{\ell=0}^{L}\mathbb R^{d_\ell}$ (disjoint union of layer‑activation vectors) |
| **Inputs $I$**          | $\mathbb R^{d_0}$ (feature vector fed to layer 0)                                       |
| **Outputs $O$**         | $\mathbb R^{d_L}$ (activation of final layer)                                           |
| **Transition $\delta$** | $ \delta(s,\ell)=\phi_{\ell} \bigl(W_\ell s+b_\ell\bigr)$ for layer index $\ell$       |
| **Termination $\tau$**  | stop when $\ell=L$ (fixed depth ⇒ finite time)                                          |

*Proof sketch.*
- Given input $x\in I$, set $s_0=x$.  
- For $\ell=0,\dots,L-1$ apply $\delta$ once; finiteness of $L$ ensures $\tau$ halts, outputting $s_L\in O$.  
- Every operation (matrix‑vector product, activation) is deterministic and Turing‑computable, so the NN conforms to the computational‑model schema.

#### 2.2 Recurrent or unrolled NN

- For time‑indexed recurrent nets (RNNs, transformers, etc.) let the hidden state dimension be $h$.
$$
\boxed{S = \mathbb R^{h}},\qquad
x_t\in I = \mathbb R^{d},\qquad
\delta(s_{t-1},x_t)=\phi \bigl(W_s s_{t-1}+W_x x_t + b\bigr).
$$

- Choose $\tau$ as “halt after $T$ time steps” or “halt when $\lVert s\_t-s\_{t-1}\rVert<\varepsilon$)”.
- This turns the (potentially infinite) stream processor into a finite‑horizon computational model whenever $\tau$ is guaranteed to fire.

#### 2.3 Cellular automaton on a finite lattice

Let $G=(V,E)$ be a finite grid graph (e.g., $V=\{1,\dots,n\}^d$) and $\Sigma$ a finite alphabet of cell states.

| Component               | Realisation                                                                                                                                        |
| ----------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------- |
| **States $S$**          | $\Sigma^{V}$ (complete lattice configurations)                                                                                                     |
| **Inputs $I$**          | subset of $S$ chosen as initial conditions                                                                                                         |
| **Outputs $O$**         | often $S$ again, or a Boolean/event extracted from a configuration                                                                                 |
| **Transition $\delta$** | global map $F{:}\Sigma^{V} \to \Sigma^{V}$ induced by local rule $f{:}\Sigma^{\mathcal N} \to \Sigma$ with neighbourhood $\mathcal N\subset V$ |
| **Termination $\tau$**  | fixed number $T$ of steps; or *halt when configuration repeats*, *when a target pattern appears*, etc.                                             |

*Proof sketch.*
- Because $V$ is finite, $S$ is finite and $F$ is a total function; iterating $F$ $T$ times is a bounded computation.  
- For infinite lattices one restricts to finite windows or finite‑time observation, yielding the same quintuple with guaranteed halting.

---

### 3 Why these embeddings satisfy the essential characteristics

| Essential characteristic                              | Neural networks                                                                                                                                                     | Cellular automata                                                                                                                                            |
| ----------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Formal semantics** (unambiguous step rule)          | A composition of linear maps and non-linear activations \$\phi\_\ell\$ completely specifies the state-update rule for every layer.                                  | A local rule \$f : \Sigma^{\mathcal N} \to \Sigma\$ deterministically yields the next state of each cell.                                                    |
| **Abstraction** (keeps only causally relevant detail) | Weight parameters encapsulate functional dependencies; peripheral hardware details do not appear in the abstract model.                                             | Only the lattice, neighbourhood, and local rule matter; the physical substrate is abstracted away.                                                           |
| **Resource accounting** (time / memory)               | With \$L\$ layers: time \$O(L)\$; memory \$O \bigl(\sum\_{\ell} d\_{\ell-1}d\_\ell\bigr)\$ for weights plus \$O \bigl(\sum\_{\ell} d\_\ell\bigr)\$ for activations. | For a finite cell set \$V\$ and \$T\$ steps: time \$O(T\lvert V\rvert)\$; memory \$O(\lvert V\rvert)\$ (one state per cell).                                 |
| **Deterministic vs. stochastic dynamics**             | Standard feed-forward nets are deterministic; randomness can be injected via dropout, Bayesian weights, etc., but still under explicit probabilistic rules.         | Classical CAs are deterministic; stochastic CAs extend the local rule to a probability distribution over successor states.                                   |
| **Reproducibility & analysability**                   | Fixed weights and input guarantee identical output; differentiability enables gradient analysis and formal verification of properties.                              | Fixed initial configuration and rule yield an identical trajectory; rich theory on fixed points, attractors, Garden-of-Eden states, etc., supports analysis. |

Hence both NNs and CAs satisfy every criterion of the original definition. No amendment is required.

---

## Summary Theorem

> **Theorem.**
> Any feed‑forward neural network of finite depth and any cellular automaton on a finite lattice (or observed for finitely many steps) instantiate a computational model $M=\langle S,I,O,\delta,\tau\rangle$ as previously defined.

*Proof.*  
The constructions in §2 give explicit $S,I,O,\delta,\tau$.  Deterministic computability of $\delta$ and finiteness of the iteration horizon implied by $\tau$ guarantee halting and thus totality of the induced input–output map. ∎

---

### Concluding remark

Neural networks, cellular automata, agent‑based simulations, finite‑element solvers, and classical automata differ only in representational convenience or domain focus; all fall under the same abstract notion of a **computational model** once their state spaces, transition functions, and termination criteria are made explicit.
