---
date: 2025-06-10
title: "What is a computational model?"
summary: "errata"
category: Tutorial 
author: "Author: Bryan Chan with ChatGPT" 
hero: /assets/images/hero3.png
image: /assets/images/card3.png
---

A **computational model** is an abstract, formally specified representation of a real or hypothetical process, whose behaviour can be specified in an executable form (algorithm, program, or simulation script), such that a computer can generate or model the system's evolution over time or across discrete states.

In particular, a good characterisation of computational models is that they map well-defined **inputs** through a sequence of **state transitions** (governed by explicit **rules or equations**) to produce **outputs**, all within stated **resource constraints** (time, memory, precision).

A computational model $M$ can be defined as a quintuple

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
