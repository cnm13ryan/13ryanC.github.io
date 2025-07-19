---
date: "2025-07-19"
title: "Plan for MARL"
summary: "Plan for MARL"
category: Plan
series: ["RL Topics", "MARL"]
author: "Author: Bryan Chan"
hero: /assets/images/hero3.png
image: /assets/images/card3.png
---

# Multi-Agent Reinforcement Learning: Structured Question Framework

## 1. Fundamental Concepts and Problem Definition

**Sequential Decision Processes**
- What is a sequential decision process?
- What is a solution to the process?
- What is learning via repeated interaction?
- How does the state of the environment evolve?
- When (if ever) does the interaction end?

**Agent Knowledge and Observability**
- What do the agents actually know about the world and about each other?
- What can each agent observe directly, and what must it infer?
- Who knows that others know what? (common versus asymmetric knowledge)
- What level of observability should agents have?

**Actions and Rewards**
- What actions exist, and which of them affect the world versus only convey information?
- How are the agent's rewards coupled?
- Who actually deserves credit (or blame) for a joint reward? (the multi-agent credit assignment puzzle)

## 2. Game Theory and Solution Concepts

**Equilibrium Existence and Uniqueness**
- What does it mean for agents to interact optimally in a multi-agent system? In other words, is a solution guaranteed to exist?
- For a given game model and solution concept, is a solution guaranteed to exist?
- Is the solution unique, or could there be many (even infinitely many) solutions?
- Which equilibrium concept should we shoot for?

**Equilibrium Selection and Coordination**
- When many equilibria exist, which one should agents adopt, and how can they agree on it?
- If several equilibria exist, which one will we end up at? (the equilibrium-selection dilemma)
- What if equilibrium requires randomising?

**Computational Complexity**
- How hard is it, computationally, to compute an equilibrium?
- Do algorithms exist that can compute equilibria efficiently (i.e., polynomial time)?
- Can we learn game solutions instead of computing them from a model?

## 3. Learning Theory and Convergence

**Convergence Properties**
- Will a given learning and decision rule, when used by all agents, converge to a solution?
- Has learning really converged? and in what sense?
- Is there an algorithm that converges to equilibrium in every general‑sum stochastic game?
- Why won't plain gradient ascent just settle?

**Non-Stationarity and Multi-Agent Learning**
- Can agents learn stably while everyone else is also learning? (the non-stationarity problem)
- If each agent just treats the others as part of the environment, what goes wrong?
- Could a single learning rule guarantee 'no regrets' regardless of opponents?

**Learning Framework Definition**
- What game are we actually trying to solve? (which formal game model captures the environment?)
- What experience counts as data? (What goes into our dataset of histories?)
- How are policies updated? (What is the learning algorithm?)
- What counts as success? (What exact learning goal / solution concept are we aiming for?)

## 4. Algorithmic Approaches and Opponent Modeling

**Strategic Considerations**
- Is modelling opponents better than worst‑case assumptions?
- How much is it worth to probe another agent?
- Can one generic template cover all these algorithms?

**Scaling and Tractability**
- How do we keep learning tractable when the number of agents grows? (the scaling question)
- How well does the algorithm scale with the number of agents?

## 5. Deep Learning and Function Approximation

**Function Approximation Fundamentals**
- What does deep learning offer over other techniques used to learn value functions, policies, and models in RL?
- Can I still afford a tabular solution?
- How do we make a value or policy function generalise to states the agent has never seen?
- Why isn't a simple linear model good enough?
- What really happens inside a 'universal function approximator'?

**Neural Network Architecture and Training**
- If gradient descent is so old, why does it still work at billion‑parameter scale?
- How large should my batch size be?
- Why bother with specialised architectures like CNNs and RNNs instead of sticking to MLPs?
- How can an agent remember what it saw earlier?
- Is back‑propagation just the chain rule?

## 6. Multi-Agent Algorithm Design

**Policy and Value Function Learning**
- What information can agents use to make their action selection, that is, to condition their policy on?
- How might we provide agents with more explicit information about the policies of other agents?
- How can agents leverage the fact that other agents are learning by shaping their behaviour to their own advantage?
- Given these considerations, why would we want to train action‑value critics for multi-agent actor‑critic algorithms, instead of learning simpler critics?

**Centralised Training and Decentralised Execution**
- In this section, we will discuss how agents can efficiently learn and use individual utility functions to jointly approximate the centralised action‑value function…?
- Which reward would agent i have received if it instead had selected its default action?

## 7. Implementation and Practical Considerations

**System Design**
- What is the minimal agent‑environment interface we can all agree on?
- Should every agent get its own neural network, or can they share one?
- When is it worth giving the critic extra information during training?
- How can we break a single team reward into per‑agent signals so each agent can still act greedily?

**Memory and State Representation**
- When should we use each approach—stacking frames, adding recurrence, or ignoring history altogether?
- How important is the information contained in previous observations to the agents' decisions?

**Training Dynamics**
- Should we standardise rewards/returns?
- Is a single optimiser for all agents really faster (and is it safe)?
- What makes a "fair" learning curve in multi‑agent settings, especially zero‑sum games?
- How do we run a hyper‑parameter search that is both exhaustive and comparable across algorithms?

## 8. Evaluation and Benchmarking

**Algorithm Assessment**
- Which properties and learning abilities do we want to test in a MARL algorithm?
- Can the algorithm reliably converge to the desired solution concept on this task?
- Will this benchmark expose generalisation rather than over‑fitting?

**Environment Design**
- Is the state/action space sufficiently rich (or purposely minimal) for the question we care about?
- How dense or sparse is the reward signal—and is that desirable?
- Which high‑level skills (co‑operation, communication, role allocation, etc.) does the environment demand?
- Does the environment offer a scalable ladder of task variants? (agent count, map size, observability radius, etc.)?

**Validation and Resources**
- Are ground‑truth solutions available or at least testable?
- Do we have the practical resources (software, compute, licences) to run this environment at scale?
