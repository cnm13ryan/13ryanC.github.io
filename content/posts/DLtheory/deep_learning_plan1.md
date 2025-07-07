---
date: "2025-07-07"
title: "(Lower-level) Tentative Plan for Deep Learning Theory Series"
summary: "(Lower-level) Tentative Plan for Deep Learning Theory Series"
category: Plan
series: ["DL Theory"]
author: "Author: Bryan Chan"
hero: /assets/images/hero3.png
image: /assets/images/card3.png
---

# A Comprehensive Plan for Deep Learning

### **Part I: Foundations of Deep Learning**

* **Chapter 1 – The Building Blocks: From Linear Models to Neural Networks**

* **1.1 The Artificial Neuron: Definition, Analysis, and Biological Context**
    * 1.1.1 Affine Core of a Neuron
    * 1.1.2 Why Non‑linearity? Universal Approximation & Failure of Pure Affine Stacking
    * 1.1.3 Activation Functions: Taxonomy and Real‑Analysis Properties
      * 1.1.3.1 Classical S‑Shaped: Sigmoid and Tanh
      * 1.1.3.2 Piece‑wise Linear Family: ReLU, Leaky/Parametric ReLU
      * 1.1.3.3 Smooth Modern Variants: GELU, SiLU, Swish, Mish, Hard Approximations
    * 1.1.4 Geometric View: Hyperplanes, Half‑Spaces, and Decision Regions
    * 1.1.5 Classical Single‑Neuron Models
      * 1.1.5.1 Perceptron (hard threshold)
      * 1.1.5.2 Linear Regression (identity activation)
      * 1.1.5.3 Logistic Regression (sigmoid activation & probabilistic view)
    * 1.1.6 Multiclass Extensions: Softmax Units and One‑vs‑Rest Schemes
    * 1.1.7 Biological Analogy: Dendrites → Sum, Soma → Activation, Axon → Output

* **1.2 Network Architectures — Classical Anatomy and Geometric Interpretation**
    * 1.2.1 Prelude
      * **C‑1.2.0** Feed‑forward computation as *function composition*
      * **G‑1.2.0** Signals on spaces, group actions, and *equivariance*
    * 1.2.2 Layer Taxonomy
      * **C‑1.2.1** Basic layer types: fully connected, convolutional, recurrent, attention
      * **G‑1.2.1** Equivariant operators over different domains
        * Trivial group → dense layer
        * Translation group → convolution
        * Time‑shift semigroup → RNN
        * Permutation group → self‑attention
    * 1.2.3 Depth‑versus‑width & Expressive Power
      * **C‑1.2.2** Universal approximation, width bottlenecks, depth benefits
      * **G‑1.2.2** Stacking equivariant layers enlarges receptive‑field radius; Stone–Weierstrass under symmetry
    * 1.2.4 Computational Graphs
      * **C‑1.2.3** Static vs dynamic unrolling, automatic differentiation
      * **G‑1.2.3** Computation DAG as a morphism in Rep (G); reverse‑mode on morphisms
    * 1.2.5 Architectural Patterns
      * **C‑1.2.4** Pattern catalogue
        * **C‑1.2.4.1** Residual and highway connections
        * **C‑1.2.4.2** Dense and Inception‑style multi‑branch modules
      * **G‑1.2.4** Pattern as geometric stabiliser
        * **G‑1.2.4.1** Discrete ODE step; near‑identity operator‑norm intuition
        * **G‑1.2.4.2** Multi‑resolution bases and parallel orbits
    * 1.2.6 Parameter Sharing & Local Connectivity
      * **C‑1.2.5** Weight tying (CNN/RNN) and local receptive fields
      * **G‑1.2.5** Group orbits ⇒ convolution‑kernel sharing; locality from the space's metric
    * 1.2.7 Capacity, Parameter Count & Memory Footprint
      * **C‑1.2.6** Counting weights/activations, FLOPs, memory
      * **G‑1.2.6** Effective capacity under symmetry; reduced sample complexity

  * **1.3 Loss Functions: Formulating the Learning Objective**
    * 1.3.1 Regression Losses: MSE, MAE, Huber, Quantile
    * 1.3.2 Classification Losses: Cross‑Entropy, Hinge, Focal, Label‑Smoothing
    * 1.3.3 Probabilistic Foundations and Maximum‑Likelihood View
    * 1.3.4 Surrogate Losses for Non‑differentiable Metrics (e.g. AUC, F‑score)
    * 1.3.5 Class Imbalance, Cost‑Sensitive and Ordinal Losses

  * **1.3 Loss Functions: Formulating the Learning Objective**
    * 1.3.1 Foundational Lenses (Why a Loss Exists at All)
        * 1.3.1.1 Measure‑Theoretic Scaffold — σ‑algebras, measurability, Lebesgue‑integral risk
        * 1.3.1.2 Decision‑Theoretic / Subjective‑Bayesian Scaffold — acts, Bayes risk, Savage’s axioms
        * 1.3.1.3 Bridge — Proper Scoring Rules & KL Divergence: strict propriety, Radon–Nikodym link
        * 1.3.1.4 Minimax & Distributionally‑Robust Risk — Sion’s theorem, adversarial view
    * 1.3.2 Loss Families and Their Dual Interpretations
        * 1.3.2.1 Regression Losses: MSE, MAE, Huber, Quantile
          * (a) M‑T View — $L^p$ integrability, influence functions
          * (b) D‑T View — elicited statistics (mean, median, quantile)
        * 1.3.2.2 Classification & Probabilistic Losses: Cross‑Entropy, Hinge, Focal, Label‑Smoothing
          * (a) M‑T View — log‑density, margin surrogates
          * (b) D‑T View — 0‑1 Bayes rule, margin theory, smoothing as prior
        * 1.3.2.3 Surrogate Losses for Non‑Differentiable Metrics (e.g. AUC, F‑score)
          * (a) M‑T View — pair‑wise measure convergence
          * (b) D‑T View — Fisher consistency for target metric
        * 1.3.2.4 Class Imbalance, Cost‑Sensitive & Ordinal Losses
          * (a) M‑T Note — re‑weighting as change of measure
          * (b) D‑T Note — asymmetric utilities, ordinal Bayes risk
    * 1.3.3 Implementation & Optimisation Considerations
        * 1.3.3.1 Gradient Properties & Smoothness (M‑T)
        * 1.3.3.2 Generalisation Bounds (D‑T)
        * 1.3.3.3 Heteroscedastic & Distributional Regression
    * 1.3.4 Worked Examples & Case Studies
        * 1.3.4.1 Quantile‑Regression Toy Data (M‑T vs. D‑T)
        * 1.3.4.2 Focal Loss on Imbalanced CIFAR‑LT

  * **1.4A The Learning Problem: Measure‑Theoretic Route**
    * 1.4A.1 Probability & Measurability Foundations
    * 1.4A.2 Empirical Risk Minimisation and the Law of Large Numbers
    * 1.4A.3 Uniform Convergence & Capacity: VC Dimension, Rademacher Complexity, PAC‑Bayes
    * 1.4A.4 Over‑/Under‑fitting & Structural Risk Minimisation
    * 1.4A.5 Validation Protocols: Hold‑out, k‑Fold, Nested CV (σ‑algebra Separation)
    * 1.4A.6 Data Augmentation & Invariance via Group Orbits
    * 1.4A.7 Transfer Learning & Domain Adaptation as Change of Measure

  * **1.4B The Learning Problem: Decision‑Theoretic Route**
    * 1.4B.1 Statistical Decision Problem: Actions, Loss, Risk
    * 1.4B.2 Bayes Rules & Subjective Priors (Regularisation as Proper Priors)
    * 1.4B.3 Minimax & Capacity Control: Worst‑Case Risk, PAC‑Bayes Posteriors
    * 1.4B.4 Model Selection & Admissibility (Over‑/Under‑fitting as Domination)
    * 1.4B.5 Posterior‑Predictive Validation: Cross‑Validation as Empirical Bayes
    * 1.4B.6 Data Augmentation through Mixture Likelihoods
    * 1.4B.7 Transfer Learning via Hierarchical Bayes (Pre‑training & Fine‑tuning)

* **Chapter 2 – The Engine: Optimization and Regularization**
  * **2.1 Gradient‑Based Foundations: The Gradient, Backpropagation, and Automatic Differentiation**
    * 2.1.1 Gradients and Jacobians: Definitions and Notation
    * 2.1.2 Backpropagation Algorithm: Chain Rule in Practice
    * 2.1.3 Computational Cost, Memory Footprint, and Check‑pointing
    * 2.1.4 Automatic Differentiation: Forward‑ vs Reverse‑Mode, Operator Overloading

  * **2.2 Gradient‑Free Optimisation: Direct, Stochastic, and Surrogate Methods**
    * 2.2.1 Direct Search: Grid/Random Search, Nelder‑Mead, Pattern & Coordinate Methods
    * 2.2.2 Population‑Based Heuristics: Evolutionary Strategies, Genetic Algorithms, Particle/Ant Swarms, CMA‑ES
    * 2.2.3 Stochastic Sampling & Annealing: Simulated Annealing, Cross‑Entropy, Adaptive Monte‑Carlo
    * 2.2.4 Surrogate‑Model Approaches: Bayesian Optimisation, Gaussian‑Process & Trust‑Region Design

  * **2.3 First‑Order Gradient‑Based Optimisers**
    * 2.3.1 Gradient Descent Variants: Batch, Stochastic, Mini‑batch
    * 2.3.2 Momentum‑Driven Methods: Classical Momentum, Nesterov Accelerated Gradient
    * 2.3.3 Adaptive Learning‑Rate Methods: Adagrad, RMSProp, AdaDelta
    * 2.3.4 Adam Family: Adam, AdamW, AMSGrad, AdaBelief

  * **2.4 Second‑Order Gradient‑Based Optimisers**
    * 2.4.1 Newton's Method and Damping
    * 2.4.2 Quasi‑Newton Techniques: BFGS, L‑BFGS
    * 2.4.3 Structured Approximations: K‑FAC and Related Kronecker/Factored Schemes

  * **2.5 Learning‑Rate & Hyper‑parameter Schedules**
    * 2.5.1 Deterministic Decays: Step, Exponential, Polynomial
    * 2.5.2 Cyclical & Annealing Schedules: Cosine Annealing, Cosine Restarts
    * 2.5.3 Warm‑up and One‑Cycle Strategies

  * **2.6 The Problem of Generalization: Underfitting, Overfitting, and the Bias‑Variance Trade‑off**
    * 2.6.1 Bias‑Variance Decomposition and Model Complexity
    * 2.6.2 Double Descent and Modern Over‑parameterization Phenomena
    * 2.6.3 Robustness to Noise, Adversarial Examples, and Distribution Shifts
    * 2.6.4 Calibration, Uncertainty Estimation, and Reliability Diagrams

  * **2.7 Regularization Techniques: \$L\_1/L\_2\$ Penalties, Dropout, Batch Normalization, and Early Stopping**
    * 2.7.1 Norm‑Based Penalties: \$L\_1\$, \$L\_2\$, Elastic‑Net, Max‑Norm
    * 2.7.2 Sparsity‑Inducing Methods and Pruning
    * 2.7.3 Dropout Family: Dropout, DropConnect, Stochastic Depth
    * 2.7.4 Normalization Strategies: Batch, Layer, Instance, Group, Weight Norm
    * 2.7.5 Data‑Driven Regularizers: Mixup, CutMix, Cutout, Manifold Mixup
    * 2.7.6 Early Stopping, Snapshot Ensembles, and Checkpoint Averaging
    * 2.7.7 Weight Decay vs \$L\_2\$ Regularization: Equivalences and Differences

  * **2.8 Optimization Theory: The Loss Landscape, Saddle Points, and Convergence Properties**
    * 2.8.1 Non‑convex Landscapes: Critical Points and Topology
    * 2.8.2 Saddle‑Point Escaping Mechanisms and Noise‑Induced Transitions
    * 2.8.3 Ill‑Conditioning, Pathological Curvature, and Preconditioning
    * 2.8.4 Convergence Guarantees: Rates for GD, SGD, and Adaptive Methods
    * 2.8.5 Sharp vs Flat Minima and Generalization Correlations
    * 2.8.6 Implicit Bias of Gradient Descent and the Lottery‑Ticket Hypothesis
    * 2.8.7 Scaling Laws, Learning Curves, and Predictable Regimes

### **Part II: Core Architectures and Paradigms**

* **Chapter 3: Convolutional Neural Networks for Spatial Data**
    * 3.1. The Convolution Operation: Locality, Parameter Sharing, and Equivariance
    * 3.2. Core Building Blocks: Pooling Layers, Strides, Padding, and Channels
    * 3.3. Canonical Architectures: LeNet, AlexNet, VGG, GoogLeNet, and the Residual Network (ResNet)
    * 3.4. Modern CNNs: Inception, DenseNet, MobileNet, and EfficientNet
    * 3.5. Applications in Computer Vision: Image Classification, Object Detection, and Semantic Segmentation
* **Chapter 4: Processing Sequences with Recurrent Architectures**
    * 4.1. The Recurrent Neural Network (RNN): Unfolding, Backpropagation Through Time (BPTT)
    * 4.2. The Challenge of Long-Term Dependencies: Vanishing and Exploding Gradients
    * 4.3. Gated Architectures: Long Short-Term Memory (LSTM) and Gated Recurrent Units (GRU)
    * 4.4. Architectural Variants: Bidirectional RNNs, Deep RNNs, and Encoder-Decoder Models
    * 4.5. Sequence-to-Sequence Tasks: Machine Translation, Text Summarization, and Speech Recognition
* **Chapter 5: The Attention Revolution: Transformer Architectures**
    * 5.1. Motivation: Overcoming the Sequential Bottleneck of RNNs
    * 5.2. The Self-Attention Mechanism: Queries, Keys, and Values
    * 5.3. The Transformer Architecture: Multi-Head Attention, Positional Encodings, and Encoder-Decoder Stacks
    * 5.4. Foundational Models: BERT (Encoder-Only), GPT (Decoder-Only), and the original Transformer
    * 5.5. Beyond NLP: Vision Transformers (ViT) and Applications in other Modalities
* **Chapter 6: Generative Deep Learning**
    * 6.1. The Generative Goal: Density Estimation and Sampling
    * 6.2. Autoregressive Models (PixelRNN, WaveNet)
    * 6.3. Variational Autoencoders (VAEs): The Reparameterization Trick and the ELBO
    * 6.4. Generative Adversarial Networks (GANs): The Discriminator-Generator Game and Loss Formulations
    * 6.5. Advanced GANs and Training Stability (WGAN, StyleGAN)
    * 6.6. Denoising Diffusion Models: The Forward/Reverse Processes and Score Matching

### **Part III: Advanced Topics and Specialized Domains**

* **Chapter 7: Deep Learning on Graphs and Structured Data**
    * 7.1. Motivation: Beyond Grids and Sequences
    * 7.2. The Neural Message Passing Framework
    * 7.3. Graph Neural Network (GNN) Architectures: GCN, GAT, GraphSAGE
    * 7.4. Theoretical Aspects: Expressive Power and the WL Test
    * 7.5. Applications: Molecular Chemistry, Social Networks, and Recommendation Systems
* **Chapter 8: Learning with Alternative Supervision**
    * 8.1. The Data Bottleneck: Beyond Supervised Learning
    * 8.2. Self-Supervised Learning (SSL): Pretext Tasks and Foundational Principles
    * 8.3. SSL Paradigms: Contrastive, Regularized, and Generative Methods
    * 8.4. The Rise of Joint-Embedding Predictive Architectures (JEPAs).
    * 8.5. Semi-Supervised and Weakly-Supervised Learning
    * 8.6. Transfer Learning, Domain Adaptation, and Fine-Tuning Strategies
* **Chapter 9: Quantifying Uncertainty: Probabilistic Deep Learning**
    * 9.1. The Need for Uncertainty in Decision-Making
    * 9.2. Bayesian Neural Networks: Priors, Posteriors, and Approximate Inference (VI, MCMC)
    * 9.3. Practical Uncertainty Estimation: Deep Ensembles and Monte Carlo Dropout
    * 9.4. Normalizing Flows for Explicit Density Modeling
    * 9.5. Conformal Prediction for Distribution-Free Uncertainty Guarantees
* **Chapter 10: Mechanistic Interpretability and Explainability**
    * 10.1. The Black Box Problem: From Correlation to Causation
    * 10.2. Explainable AI (XAI): Saliency Maps, Feature Attribution, and Concept-Based Explanations
    * 10.3. Mechanistic Interpretability: Finding and Analyzing "Circuits" in Transformers and CNNs
    * 10.4. Probing and Diagnostic Classifiers
    * 10.5. The Link Between Interpretability and Safety
    * 10.6. Representational Pathologies: FER vs. UFR
* **Chapter 11: Biologically-Inspired Paradigms**
    * 11.1. Neuroscience as a Source of Architectural Priors
    * 11.2. Spiking Neural Networks (SNNs): Event-Driven and Efficient Computation
    * 11.3. Hebbian Learning and Self-Organization
    * 11.4. Memory-Augmented Neural Networks and Attention Mechanisms
    * 11.5. Future Directions: Brain-Computer Interfaces and Systems Neuroscience Links
* **Chapter 12: Physics-Inspired Deep Learning**
    * 12.1. The Synergy of Data and Physical Law
    * 12.2. Physics-Informed Neural Networks (PINNs): Encoding PDEs in the Loss Function
    * 12.3. Hamiltonian and Lagrangian Neural Networks: Architectural Priors for Conserved Systems
    * 12.4. Applications in Scientific Computing, Engineering, and Climate Modeling
    * 12.5. Future Directions: Differentiable Physics and Causal Discovery
* **Chapter 13: Abstract Structures: Geometric & Categorical Deep Learning**
    * 13.1. The Quest for a Principled Theory of Deep Learning
    * 13.2. The Geometric Perspective: Symmetry and Invariance
    * 13.3. The Categorical Perspective: Compositionality and Algebra
    * 13.4. Synthesis and Advanced Applications
    * 13.5. Future Directions: Towards a General Theory of Intelligence

### **Part IV: Cross-Cutting Challenges and Future Directions**

* **Chapter 13: Generalization and Adaptation**
    * 13.1. The Challenge of Out-of-Distribution (OOD) Generalization
    * 13.2. Meta-Learning: Learning to Learn and Fast Adaptation (MAML, Reptile, ProtoNets)
    * 13.3. Continual and Lifelong Learning: Overcoming Catastrophic Forgetting (EWC, Replay Methods)
    * 13.4. The Role of Inductive Biases in Generalization
    * 13.5. Open-World Learning and Novelty Detection
* **Chapter 14: Trustworthy AI: Safety, Fairness, and Alignment**
    * 14.1. Adversarial Robustness: Attacks (FGSM, PGD) and Defenses
    * 14.2. Fairness and Bias: Defining, Measuring, and Mitigating Algorithmic Bias
    * 14.3. Privacy-Preserving Deep Learning: Federated Learning and Differential Privacy
    * 14.4. Causal Inference and Deep Learning: Moving from Correlation to Causation
    * 14.5. AI Alignment: Ensuring Models Adhere to Human Values and Intent
* **Chapter 15: Synthesis and Future Frontiers**
    * 15.1. Scaling Laws: The Predictable Unpredictability of Large Models
    * 15.2. The Foundation Model Ecosystem: Emergent Abilities and Societal Impact
    * 15.3. Neuro-Symbolic AI: Combining Deep Learning with Symbolic Reasoning
    * 15.4. The Future of Hardware: Co-designing Chips and Algorithms
    * 15.5. Grand Challenges: Compositionality, Systematic Generalization, and a Unified Theory of Intelligence
