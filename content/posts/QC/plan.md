---
date: "2025-07-07"
title: "(Lower-level) Tentative Plan for QC Theory Series"
summary: "(Lower-level) Tentative Plan for QC Theory Series"
category: Plan
series: ["QC Theory"]
author: "Author: Bryan Chan"
hero: /assets/images/hero_qc.png
image: /assets/images/card_qc.png
---

### **Comprehensive Quantum Computing: From Foundations to Frontiers**

---

### **Part I: Foundations**

**Chapter 1: The Quantum Computing Problem: Qubits, Gates & Circuits**
* **1.1. Formalism: Hilbert Spaces, Qubits, and Quantum States**
    * 1.1.1. Mathematical Foundations: Complex Vector Spaces, Inner Products
    * 1.1.2. The Qubit: Superposition and the Bloch Sphere
    * 1.1.3. Multi-Qubit Systems: Tensor Products and Entanglement
    * 1.1.4. Mixed States and the Density Matrix Formalism
    * 1.1.5. The Postulates of Quantum Mechanics
* **1.2. Evolution and Measurement: Gates, Circuits, and Observables**
    * 1.2.1. Quantum Gates as Unitary Transformations (Pauli, Hadamard, Phase, CNOT)
    * 1.2.2. The Universal Gate Set Theorem
    * 1.2.3. Quantum Circuits: Composition and Representation
    * 1.2.4. Measurement, Probabilities, and State Collapse
* **1.3. Computational Complexity: BQP, QMA, and Classical Limits**
    * 1.3.1. Classical Complexity Classes (P, NP, BPP)
    * 1.3.2. The Quantum Complexity Class BQP (Bounded-error Quantum Polynomial time)
    * 1.3.3. QMA and the Quantum Analogue of NP
    * 1.3.4. Relationship between Quantum and Classical Classes

**Chapter 2: Core Tools: Quantum Algorithms & Key Primitives**
* **2.1. The Quantum Fourier Transform (QFT)**
    * 2.1.1. Definition and Circuit Implementation
    * 2.1.2. Analysis of Computational Cost
* **2.2. Phase Estimation Algorithm (PEA)**
    * 2.2.1. The PEA Circuit and its Analysis
    * 2.2.2. Application to Order-Finding
* **2.3. Amplitude Amplification and Grover's Algorithm**
    * 2.3.1. The Unstructured Search Problem
    * 2.3.2. Geometric Interpretation: Rotation in a 2D Subspace
    * 2.3.3. Grover's Algorithm: Optimality and Generalisation
    * 2.3.4. Amplitude Estimation and Quantum Counting
* **2.4. Hamiltonian Simulation**
    * 2.4.1. The Problem of Simulating Physical Systems
    * 2.4.2. Product Formulas (Lie-Trotter-Suzuki)
    * 2.4.3. Advanced Methods: Quantum Signal Processing and Qubitization

---

### **Part II: Core Algorithmic Paradigms**

**Chapter 3: Algorithms with Super-polynomial Speedup**
* **3.1. Shor's Algorithm for Integer Factorisation**
    * 3.1.1. Reduction of Factoring to Order-Finding
    * 3.1.2. Integrating Phase Estimation with Classical Post-processing
    * 3.1.3. Implications for Cryptography (RSA)
* **3.2. The Discrete Logarithm Problem**
    * 3.2.1. Adapting Shor's Algorithm for Discrete Log
    * 3.2.2. Implications for Diffie-Hellman and Elliptic Curve Cryptography
* **3.3. The Abelian Hidden Subgroup Problem (HSP)**
    * 3.3.1. Formal Definition of the HSP
    * 3.3.2. A Unified Framework for Shor's, Simon's, and Deutsch-Jozsa Algorithms
* **3.4. The HHL Algorithm for Linear Systems**
    * 3.4.1. The Quantum Linear Systems Problem
    * 3.4.2. Algorithm Breakdown: Phase Estimation, Controlled Rotation, and Uncomputation
    * 3.4.3. Caveats: The Data Loading Problem and BQP-Completeness

**Chapter 4: Search, Quantum Walks, and Heuristic Algorithms**
* **4.1. Applications and Extensions of Grover's Algorithm**
    * 4.1.1. Solving NP-Complete Problems (e.g., 3-SAT) with Quadratic Speedup
    * 4.1.2. Collision Finding and its Cryptographic Relevance
* **4.2. Quantum Walks**
    * 4.2.1. Discrete vs. Continuous-Time Quantum Walks
    * 4.2.2. Application: Element Distinctness
    * 4.2.3. Search by Quantum Walk on Graphs
* **4.3. Adiabatic Quantum Computing (AQC)**
    * 4.3.1. The Adiabatic Theorem and Computational Equivalence
    * 4.3.2. Application to Optimisation Problems
    * 4.3.3. Quantum Annealing as a Heuristic Analogue
* **4.4. Variational Quantum Algorithms (VQE, QAOA)**
    * 4.4.1. The Hybrid Quantum-Classical Loop
    * 4.4.2. The Variational Quantum Eigensolver (VQE) for Chemistry and Materials
    * 4.4.3. The Quantum Approximate Optimisation Algorithm (QAOA) for Combinatorial Problems
    * 4.4.4. Barren Plateaus and Trainability Issues

**Chapter 5: Quantum Communication and Cryptography**
* **5.1. Fundamental Quantum Information Primitives**
    * 5.1.1. The No-Cloning Theorem
    * 5.1.2. Quantum Teleportation
    * 5.1.3. Superdense Coding
* **5.2. Quantum Key Distribution (QKD)**
    * 5.2.1. The BB84 Protocol
    * 5.2.2. Entanglement-Based QKD (E91 Protocol)
    * 5.2.3. Security Proofs and Practical Implementations
* **5.3. Quantum Shannon Theory**
    * 5.3.1. Von Neumann Entropy and Quantum Data Compression
    * 5.3.2. Holevo's Bound and Channel Capacity

---

### **Part III: Advanced Topics**

**Chapter 6: Quantum Error Correction (QEC)**
* **6.1. The Challenge of Decoherence and Noise**
    * 6.1.1. Quantum Operations and Noise Channels (Bit-flip, Phase-flip, Depolarising)
    * 6.1.2. The Discretisation of Errors
* **6.2. The Stabilizer Formalism**
    * 6.2.1. Pauli Group and Stabilizer Groups
    * 6.2.2. Defining Codespaces and Error Detection
    * 6.2.3. The Knill-Laflamme Conditions for QEC
* **6.3. Canonical Error-Correcting Codes**
    * 6.3.1. The 3-Qubit Repetition Code (Bit-flip Correction)
    * 6.3.2. The 9-Qubit Shor Code (Arbitrary Single-Qubit Error Correction)
    * 6.3.3. The 7-Qubit Steane Code
* **6.4. Fault-Tolerant Quantum Computation**
    * 6.4.1. Performing Logic on Encoded Qubits (Transversal Gates)
    * 6.4.2. The Threshold Theorem and the Promise of Scalability
    * 6.4.3. Topological Codes: The Surface Code

**Chapter 7: Quantum Machine Learning (QML)**
* **7.1. Data Encoding Strategies**
    * 7.1.1. Basis, Amplitude, and Angle Encoding
    * 7.1.2. The Data Loading Problem and QRAM
* **7.2. Quantum Kernels and Support Vector Machines**
    * 7.2.1. Feature Maps and the Quantum Kernel Trick
    * 7.2.2. Potential for Advantage and Practical Limitations
* **7.3. Quantum Neural Networks (QNNs)**
    * 7.3.1. Parameterised Quantum Circuits as Models
    * 7.3.2. Gradient-Based Training and Parameter Shift Rules
* **7.4. Generative Quantum Models**
    * 7.4.1. Quantum Circuit Born Machines
    * 7.4.2. Quantum Generative Adversarial Networks (qGANs)

**Chapter 8: Measurement-Based and Topological Quantum Computing**
* **8.1. Measurement-Based Quantum Computing (MBQC)**
    * 8.1.1. Cluster States as a Universal Resource
    * 8.1.2. Computation via Adaptive Single-Qubit Measurements
    * 8.1.3. Equivalence to the Circuit Model
* **8.2. Topological Quantum Computing (TQC)**
    * 8.2.1. Anyons and Non-Abelian Statistics
    * 8.2.2. Braiding as Quantum Gates
    * 8.2.3. Intrinsic Fault Tolerance through Topological Protection

---

### **Part IV: Cross-Cutting Challenges & Future Directions**

**Chapter 9: Physical Realizations and Hardware**
* **9.1. A Taxonomy of Qubit Platforms**
    * 9.1.1. Superconducting Circuits (Transmons)
    * 9.1.2. Trapped Ions
    * 9.1.3. Photonic Quantum Computing
    * 9.1.4. Neutral Atoms and Silicon Quantum Dots
* **9.2. Control, Readout, and Connectivity**
    * 9.2.1. Microwave Control and Cryogenics
    * 9.2.2. Qubit Readout Techniques and Fidelities
    * 9.2.3. Architectural Topologies (e.g., Linear, Grid)
* **9.3. Benchmarking and Performance Metrics**
    * 9.3.1. Coherence Times ($T_1, T_2$) and Gate Fidelities
    * 9.3.2. Quantum Volume and Cross-Entropy Benchmarking (XEB)

**Chapter 10: Compilation, Simulation, and Error Mitigation**
* **10.1. Quantum Compilation**
    * 10.1.1. Decomposing Gates into Native Hardware Operations
    * 10.1.2. Qubit Allocation and SWAP Insertion
    * 10.1.3. Noise-Aware Compilation
* **10.2. Classical Simulation of Quantum Circuits**
    * 10.2.1. State-Vector Simulation
    * 10.2.2. Tensor Network Methods (MPS, PEPS)
* **10.3. Quantum Error Mitigation**
    * 10.3.1. The NISQ (Noisy Intermediate-Scale Quantum) Era
    * 10.3.2. Techniques: Zero-Noise Extrapolation (ZNE), Probabilistic Error Cancellation (PEC)

**Chapter 11: Synthesis and Open Problems**
* **11.1. The Path to Quantum Advantage**
    * 11.1.1. Defining and Demonstrating Practical Advantage
    * 11.1.2. Resource Estimation for Fault-Tolerant Algorithms (e.g., Factoring)
* **11.2. Grand Challenges in Quantum Algorithms**
    * 11.2.1. The Non-Abelian Hidden Subgroup Problem
    * 11.2.2. Developing Novel Algorithms Beyond QFT and Search
* **11.3. Post-Quantum Cryptography**
    * 11.3.1. The Threat to Classical Cryptosystems
    * 11.3.2. Main Families of PQC: Lattice-based, Code-based, Hash-based, Multivariate
* **11.4. The Future of Quantum Computing**
    * 11.4.1. Integration with High-Performance Computing (HPC)
    * 11.4.2. Quantum Sensing and Metrology
    * 11.4.3. The Development of Quantum Networks
