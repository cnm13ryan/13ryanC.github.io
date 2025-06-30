---
date: "2025-06-27"
title: "Start building the foundations of Reinforcement Learning that I Can Trust"
summary: "After years of frustration with hand-waving and leaky abstractions, this is the start of my informal notes to unify the theory and practice of RL."
category: Tutorial
series: ["RL Theory"]
author: "Author: Bryan Chan"
hero: /assets/images/hero3.png
image: /assets/images/card3.png
---

There’s plenty of introductory material on Reinforcement Learning, yet I still don’t fully grasp it. Even when authors show working demos, it often feels like hand‑waving—code that suddenly works.

I can replicate the experiments and vaguely follow the flow, but can I clearly explain *why* things happen, instead of simply pointing to some authoritative paper? Rarely the case. 

That disconnect—between **how** to make code run and **why** it must work at all—is exactly what finally pushed me to split the subject into what I’ll call a *low road* (process / implementation) and a *high road* (principle / justification).

Obviously this stems from my own insufficient understanding, so I've had enough of hand‑waving and pretending everything is fine. I'm going to write my own informal notes, analyzed from my own perspective.

For context, I’m self-taught in real analysis. Hand-waving doesn’t stick; my brain just filters it out. I need a solid baseline I can trust. My years in engineering felt wasted because everything relied on that same leap of faith. After studying rigorous math, it’s hard to accept flimsy reasoning in any domain.

Engineering showed me the **low‑road tricks**—debuggers, benchmarks, GPU hacks. Real analysis revealed the **high‑road mindset**—axioms, convergence proofs, sample‑complexity bounds. Neither side alone has helped me *truly* learn RL; taken together they might.
 

I really hope people start unifying theory and practice more closely. 

I think we should be much more serious about mathematical correctness and the conditions in which our methods apply. This reminds me of Terry Tao's justification for formal study in his [*Analysis I* textbook](#ref-tao "Tao, T. (2022). Analysis I (4th ed.). Springer Nature."), where he essentially asks, "why bother?" One might argue that you only need to know how things work to solve real-life problems. However, as Tao points out, you can get into serious trouble if you apply rules without knowing where they came from or what their limits are. 

The low road warns us when code breaks; the high road tells us whether it *had to* break given its assumptions.

If you’re still not convinced, perhaps Joel Spolsky’s concept of ["leaky abstractions"](#ref-spolsky "Spolsky, J. (2002, November 11). The law of leaky abstractions. Joel on Software.") will help explain: all non-trivial abstractions are imperfect, and sooner or later, they will fail.

Some time ago I read Sutton and Silver’s position paper on the coming ["era of experience,"](#ref-silver-sutton "Silver, D., & Sutton, R. S. (2023). The Era of Experience. DeepMind.") and I agree the next learning paradigm will likely emerge there.
. 

It’s important to understand that while deep learning has brought surprises, RL has always been actively present—from the era of simulation (AlphaGo) to the era of human data (RLHF, safety training). I think we can all agree that the next paradigm will be the era of experience, especially as we see diminishing returns from large-scale simulation and using human data as a supervised learning target.

In each era the **low road shipped the systems**, while the **high road explained their limits and sample costs.**
 

The success of this era of experience rests unequivocally on reinforcement learning and its adjoint fields like universal artificial intelligence, information theory, and Bayesian learning. 

We may reach this future sooner than we think. Regardless of the risks of superintelligence introduced in Nick Bostrom’s (2014) book [*Superintelligence*](#ref-bostrom "Bostrom, N. (2014). *Superintelligence: Paths, Dangers, Strategies*. Oxford University Press.") as well as [other risks](#ref-bengio "Bengio, Y. et al. (2025). *Superintelligent Agents Pose Catastrophic Risks: Can Scientist AI Offer a Safer Path?* arXiv:2502.15657.") it is critical for us to revisit the fundamentals of RL.&#x20;

Practitioners still struggle with experimentation; recent work highlights how difficult reproducing published results can be ([Gazeau et al., 2023](#ref-gazeau "Gazeau, M., Darvish, K., Romero, D., Sigaud, O., & Geist, M. (2023). Empirical Design in Reinforcement Learning. arXiv preprint arXiv:2304.01315.")).

For more rapid progress in this field, there must be a more unified and holistic view. Right now, information is scattered, and the bridge between experimentation and theory is weak.

I hope that through this blog, I can help the community lean more in that direction. Frankly, I’m just sick of retrieving materials and sticking them together in ways that don't lead to real learning or absorption.


So here is my plan: peel RL apart into its two interlocking stories. *Low road*: the craft of getting gradients to flow. *High road*: the logic that says gradients are even relevant. **Each post in the series will make the handshake explicit—_this algorithmic step corresponds to that Bellman axiom_, and vice‑versa.**
 
So, I will start by approaching RL from two strands: the "low road" which focuses on process and implementation and the "high road" of theory and justification.
* **Low Road — Process & Implementation**  
  I will use Sutton & Barto’s introductory textbook on RL and will be guided by Michael Frank’s [experimentology](#ref-frank "Frank, M. C. (2023) Experimentology book"), which emphasises rigorous, model‑driven experimental design: reproducible pipelines, careful hyper‑parameter sweeps, and transparent statistical analysis. This strand emphasises building and testing concrete algorithms—TD‑learning, deep actor–critic, etc.—and learning from the data they generate.  
* **High Road — Theory & Guarantees**  
  I will study the normative backbone of RL using resources from [Csaba Szepesvári's textbook](#ref-szepesvari "Szepesvári, C. (2010). Algorithms for Reinforcement Learning. Morgan & Claypool Publishers.") and the ongoing discussions in the [RL Theory seminars](#ref-rltheory "RL Theory Seminars."). These develop optimal‑control principles, Bellman equations, regret bounds, sample‑complexity, and runtime‑complexity results that explain *why* and *how fast* RL algorithms can work.
 

This dual approach is very much inspired by how Karl Friston and his colleagues frame active inference in their comprehensive book, [*Active Inference*](#ref-parr "Parr, T., Pezzulo, G., & Friston, K. J. (2022). Active Inference: The Free Energy Principle in Mind, Brain, and Behavior. MIT Press."). 

My hope is that by tackling RL from both angles, we can see how theory and experimentation can be connected more tightly. 

With the emerging maturity of formalizing mathematical proofs in languages like Lean, maybe one day the process can be seamless. A researcher could poke at a crazy idea, get help from an AI to flesh out the mathematics, and immediately implement it. Conversely, a discovery in implementation could be quickly formalized to verify its correctness, helping the theory side.

This sounds very idealistic, but I hope it gets there sooner than I think. It would help out beginners, practitioners, and researchers alike.


Because when the **low road’s empirical bumps** meet the **high road’s guard‑rails**, we finally get a path that is both *walkable* **and** *provably headed uphill*.

---

{{< bibliography "data/pubs.json" "cited" >}}
