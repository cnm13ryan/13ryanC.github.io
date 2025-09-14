---
date: "2025-09-14"
title: "Roofline Analysis: A Comprehensive Guide"
summary: "What are algorithms bounded by? How to measure and compare them?"
lastmod: "2025-09-14"
category: "Notes"
series: ["Compute"]
author: "Bryan Chan"
hero: /assets/images/hero3.png
image: /assets/images/card3.png
---

## Executive Summary

**The Simple Roofline in One Line:**
For a kernel that does $F$ floating-point operations and moves $B$ bytes on a machine with peak compute $\pi$ (FLOP/s) and peak bandwidth $\beta$ (B/s), the best sustained performance you can hope for is:

$$P_{\max}=\min\lbrace \pi,\ \beta\cdot I\rbrace\quad\text{with}\quad I=\frac{F}{B}$$

i.e., the smaller of the compute roof $\pi$ and the bandwidth roof $\beta I$. The "ridge point" $\pi/\beta$ on the $x$-axis is the minimum arithmetic intensity $I$ needed to be compute-bound; to the left you're bandwidth-bound, to the right compute-bound.

**Core Concept**: 
- Every kernel has two clocks: math time and data-movement time. 
- Performance is bounded by the maximum of these two when overlapped. The crossover point is determined by **arithmetic intensity** (FLOPs per byte moved). 
- Compare your kernel's intensity to the hardware's critical intensity to determine if you're compute-bound or bandwidth-bound. 
- Count **all** relevant bytes (reads, writes, collectives) for the active roofline (HBM, inter-chip, etc.).

**Key Takeaway**: 
- Rooflines turn performance questions into a simple visual bound, allowing you to identify the governing limit and select the right optimization lever. 
- When multiple rooflines exist (HBM, ICI/DCN/PCIe), the achievable throughput is the **minimum** over the roofs; in the time domain the lower bound is the **max** of compute time and each non-overlapped communication time.

---

## 1. Fundamental Concepts

### 1.1 What the Roofline is Really Doing

**Bound-and-bottleneck logic.**
Execution time has two irreducible pieces:

$$T_{\text{math}}=\frac{F}{\pi},\qquad T_{\text{comms}}=\frac{B}{\beta}$$

If math and movement overlap well, time is lower-bounded by $\max(T_{\text{math}},T_{\text{comms}})$ and upper-bounded by their sum. That max leads directly to the roofline min above (since $P=F/T$). The gap between these bounds is at most a factor of 2, which is often good enough to reason about where time goes. Compute-bound means $T_{\text{math}} > T_{\text{comms}}$; bandwidth-bound is the opposite.

### 1.2 The Two-Clock Model

Any algorithm's runtime on hardware is bounded by:
- **Computation speed**: How fast it can perform FLOPs
- **Data movement speed**: How fast it can move bytes
- **Memory capacity**: Whether data fits in memory

**Runtime Bounds:**
- **Lower bound (with perfect overlap)**: $T_{\min} = \max(T_{\text{math}}, T_{\text{comm}})$
- **Upper bound (no overlap)**: $T_{\max} = T_{\text{math}} + T_{\text{comm}} \leq 2T_{\min}$

> **Practical note on overlap.** Real programs rarely achieve perfect overlap across all paths. Treat $T_{\min}$ as a target and use profiling to locate non-overlapped regions; use $T_{\max}$ as a sanity upper bound.

### 1.3 Arithmetic Intensity

$$I_{\text{alg}} = \frac{\text{FLOPs}}{\text{bytes moved}}$$

This single metric predicts the operational regime:
- **Low intensity** → bandwidth-bound (waiting on data)
- **High intensity** → compute-bound (saturating compute units)

### 1.4 Hardware Critical Intensity (The Ridge Point)

$$I_{\text{hw}} = I^* = \frac{\text{peak FLOPs/s}}{\text{bandwidth}} = \frac{\pi}{\beta}$$

**Decision Rule:**
- If $I_{\text{alg}} < I^*$ → **bandwidth-bound** (left of ridge)
- If $I_{\text{alg}} > I^*$ → **compute-bound** (right of ridge)
- If $I_{\text{alg}} \approx I^*$ → **at the ridge** (balanced)

### 1.5 The Roofline Plot

**Axes and the "ridge"** on a log-log plot with $x=I$ (FLOPs/byte) and $y=P$ (FLOP/s):
- A **horizontal roof** at $y=\pi$ (peak compute)
- A **slanted roof** of slope 1: $y=\beta x$ (peak bandwidth)
- They meet at the **ridge** $I^\ast =\pi / \beta$

Points left of the ridge are bandwidth-bound; right are compute-bound.

### 1.6 Minimal Derivation (to see the gears turn)

$$
\begin{aligned}
T &\ge \max\Big(\tfrac{F}{\pi},\tfrac{B}{\beta}\Big) \\
P&=\tfrac{F}{T}\ \le\ \frac{F}{\max(F/\pi,\ B/\beta)}
=\min\Big(\pi,\ \beta\cdot \tfrac{F}{B}\Big)
=\min(\pi,\ \beta I)
\end{aligned}
$$

That's the whole model. From there, you only need to be honest about **what** bytes you're counting and **which** bandwidth they cross.

### 1.7 Terminology Note

The original paper uses **operational intensity** (ops per **DRAM** byte) to emphasize off-chip traffic; many ML texts say **arithmetic intensity** more broadly. The math above is the same; just be precise about **which** bytes you're counting (HBM vs L2 vs network).

### 1.8 Byte-Counting Conventions (what "bytes moved" includes)

When you compute "bytes moved," be explicit about *which* roofline you are evaluating and include the matching bytes:

- **HBM roofline (on-chip memory ↔ HBM):** Count **reads and writes** for all tensors touched by the kernel: inputs, parameters, outputs, temporaries written to HBM. Use datatype sizes (e.g., bf16=2 B, fp32=4 B, int8=1 B).  
- **Inter-chip roofline (e.g., ICI/DCN):** Count **collectives and sends** (e.g., all-reduce/all-gather/point-to-point). Example: 2-chip matmul partial-sum exchange moves **$2BF$** bytes (send + recv) in bf16.
- **Training-specific:** If you analyze a **training step**, also count gradient writes and optimizer state updates (often 2–3× parameter size, depending on optimizer).
- **Inference-specific:** For attention, count **KV-cache** reads/writes per token.
- **Small-$B$ caution:** At small $B$, the **output write** (e.g., $2BF$ bytes in bf16) can dominate the denominator in $I_{\text{alg}}$.

> **Tip.** Keep a small table next to each derivation: per-tensor bytes (read/write), dtype, and whether it hits HBM vs interconnect. This prevents under-counting and explains discrepancies between theory and profiles.

---

## 2. How to Use the Simple Framework (A Crisp Checklist)

1. **Pick the traffic you care about.** Single-GPU HBM? Inter-GPU network? (Each gives a different $\beta$ and thus a different roof.)
2. **Measure/estimate your kernel's work and traffic.** Count FLOPs $F$ and bytes $B$ actually moved over that link. (Use profiles or conservative static counts.) Compute $I=F/B$.
3. **Grab hardware peaks.** $\pi$ (FLOP/s at your precision / unit) and $\beta$ (sustained bandwidth).
4. **Classify.** Compare $I$ to $I^*=\pi/\beta$. If $I<I^*$, you're bandwidth-bound; if $I>I^*$, compute-bound.
5. **Bound time.** Lower bound $T\ge\max(F/\pi,\ B/\beta)$. If you need a rough upper bound, use $T\le F/\pi+B/\beta$.
6. **Decide the lever:**
   - Bandwidth-bound → increase $I$ (reuse, tiling, fusion, quantization, caching) or raise $\beta$ (faster memory/link)
   - Compute-bound → raise $\pi$ use (tensor cores, vectorization, better occupancy), or reduce $F$ (algorithmic changes)

---

## 3. Hardware Specifications

### 3.1 TPU v5e (MXU)
- **Peak FLOPs/s**: $1.97 \times 10^{14}$ (bf16)
- **HBM Bandwidth**: $8.2 \times 10^{11}$ B/s
- **Critical Intensity**: $I^* \approx 240$ FLOPs/byte
- **Practical Rule**: Need local batch $B \gtrsim 240$ for compute-bound matmul

### 3.2 TPU v6e
- **Peak FLOPs/s**: $9.1 \times 10^{14}$
- **HBM Bandwidth**: $1.6 \times 10^{12}$ B/s
- **Note on sustained vs peak**: In typical use, TPUs often achieve a high fraction of peak (≈95% reported in practice).

### 3.3 GPU H100
- **Peak FLOPs/s**: $9.89 \times 10^{14}$ (bf16)
- **HBM Bandwidth**: $3.35 \times 10^{12}$ B/s
- **Critical Intensity**: $I^* \approx 295$ FLOPs/byte
- **Practical Rule**: Need local batch $B \gtrsim 300$ for compute-bound matmul
- **Sustained vs peak**: Realized throughput is typically **80–85%** of the advertised peak; budget with sustained numbers when capacity-planning.

### 3.4 Network Interconnects
- **ICI (Inter-Chip Interconnect)**: Example $4.5 \times 10^{10}$ B/s
- Creates separate network roofline with different critical intensity

### 3.5 MXU vs VPU (why some ops have very low "knees")

- **MXU (matrix multiply unit)**: Very high peak OPs/s, governs matmuls. v5e critical intensity ≈240 FLOPs/byte (bf16).
- **VPU (vector/elementwise unit)**: Much lower peak OPs/s; many elementwise ops (e.g., dot on TPU) run here. A representative figure (TPU v5p VPU) is **≈7×10¹² FLOPs/s** with a **critical intensity ≈3 FLOPs/byte**, so even simple reductions remain comms-bound.

> Actionable implication: don't expect elementwise kernels to become compute-bound via batch size alone; raise intensity by **fusion** or reduce bytes moved.
 
---

## 4. Common Operations Analysis - Worked Mini-Examples (Units Checked)

### 4.1 Dot Product (bf16)

**Vectors**: $x, y \in \mathbb{R}^N$

Per JAX's derivation: loads $x,y$ (each $2N$ B), writes 2 B; FLOPs $N+(N-1)=2N-1$.
- **FLOPs**: $2N - 1$ (multiply + add)
- **Bytes**: $4N + 2$ (read two vectors, write one scalar)
- **Arithmetic intensity**: $I = \frac{2N-1}{4N+2} \xrightarrow[N\to\infty]{} \frac{1}{2}$ FLOPs/byte

On hardware with $\pi=100$ TFLOP/s and $\beta=2$ TB/s:
- Ridge $I^*=\pi/\beta=100/2=50$ FLOPs/byte (exact division: 100÷2=50)
- Performance bound $P_{\max}=\min(100,\ 2\times 0.5)=\min(100,\ 1)=1$ TFLOP/s

**Result**: Clearly bandwidth-bound; the machine's math units mostly wait on memory. On TPUs it typically executes on the **VPU**, whose critical intensity is only ~3 FLOPs/byte, and $0.5 \ll 3$ still leaves it comms-bound.

### 4.2 Matrix Multiplication (Single Chip, bf16)

**Operation**: $X[B,D] \times Y[D,F] \to Z[B,F]$

Bytes $\approx 2(BD+DF+BF)$; FLOPs $=2BDF$
- **FLOPs**: $2BDF$
- **HBM Bytes**: $2BD + 2DF + 2BF$
- **Intensity**: $I = \frac{2BDF}{2(BD + DF + BF)} = \frac{BDF}{BD + DF + BF} \approx B$ when $D,F \gg B$

Compute-bound when $B \gtrsim \pi/\beta$. On a TPU v5e example, $\pi/\beta \approx 240$, so once the per-replica token batch $B$ exceeds ~240, GEMM tends to saturate compute. (The exact crossover depends on where the work runs—MXU vs VPU—but the method is the same.)

### 4.3 Distributed Matrix Multiplication

**Setup**: 2-chip sharding along dimension $D$
- Each chip computes partial: $X[:, :D/2] @ Y[:D/2, :]$
- Exchange partial sums: $Z_{\text{part}} \in \mathbb{R}^{B \times F}$

**Per-chip calculations**:
- **FLOPs**: $BDF$ (half of total)
- **Network bytes**: $2BF$ (send/receive partials in bf16)

**Compute-bound condition** (with example numbers):
$$\frac{BDF}{1.97 \times 10^{14}} > \frac{2BF}{4.5 \times 10^{10}}$$

Simplifies to: $D > 8,755$

**Key Insight**: Threshold depends on $D$ (sharded dimension), not $B$.

### 4.4 Tiled Matmul (on-chip reuse changes the denominator)

Large GEMMs are implemented as tiles to fit on-chip memory (VMEM/SMEM/TMEM). For $(m,k)\cdot(k,n)$ with tile sizes $(b_m,b_k,b_n)$ and tile counts $(t_m,t_k,t_n)$:

- **FLOPs**: $2\cdot t_m t_n t_k \cdot b_m b_k b_n$
- **HBM bytes (dominant terms)**: $2\cdot t_m t_n \left[t_k (b_m b_k + b_k b_n)\right]$ (+ $2 b_m b_n$ writes)
- **Approx. intensity (ignoring writes)**: 
  $$I \approx \frac{b_m b_n}{b_m + b_n}$$

This shows why tile choices matter: increasing $b_m$ and $b_n$ boosts reuse and $I$, pushing you rightward on the roofline even at fixed global $(m,k,n)$.

---

## 5. What Actually Moves Your Dot (Intuition That Sticks)

- **Increase $I$** (same FLOPs, fewer bytes across the link): cache/blocking so tiles fit in fast memory; fuse kernels to reuse tensors; lower precision / quantize activations if the math still runs on higher-throughput units; restructure layouts for coalesced accesses. These shift you right on the $x$-axis.
- **Increase $\beta$**: faster HBM, better NVLink/ICI topology, overlapping copies; for distributed workloads, prefer collective patterns with less cross-replica traffic. These tilt the slanted roof up.
- **Increase effective $\pi$**: use tensor/matrix units, vectorize, unroll, raise occupancy; remove instruction mix bottlenecks. This raises the flat roof.

---

## 6. Quantization Effects

### 6.1 Full int8 matmul (activations **and** weights, int8 compute)

Here both the **algorithm bytes** and the **hardware peak OPs/s** change.

- **FLOPs**: still $2BDF$ OPs
- **HBM bytes**: $BD + DF + BF$ (all int8 → 1 B/elt)
- **Algorithm intensity** (assume $B \ll D,F$): $I_{\text{alg,int8}} \approx \dfrac{2BDF}{DF} = 2B$
- **Hardware critical intensity** (example numbers): $I_{\text{hw,int8}}=\dfrac{3.94\times10^{14}}{8.1\times10^{11}} \approx 486$
- **Compute-bound condition**: $2B > 486 \Rightarrow B > 243$

**Conclusion:** The **batch threshold is essentially unchanged** vs bf16 (≈240→≈243). Full int8 does **not** halve the threshold; both the numerator (intensity) **and** the hardware knee increase ~2×.

### 6.2 Mixed precision: bf16 compute × int8 **weights** (activations bf16)

- **HBM bytes**: $2BD + (1)DF + 2BF$
- **Intensity (for $B \ll D,F$)**: denominator halves vs bf16 (from ~$2DF$ to ~$DF$), so
  $$I_{\text{alg,mixed}} \approx \frac{2BDF}{DF} = 2B$$
- **Hardware knee** stays **bf16**: $I_{\text{hw}} \approx 240$
- **Compute-bound condition**: $2B > 240 \Rightarrow B > 120$ → **threshold halves**.

### 6.3 Other asymmetric cases (when to expect no change)

If you quantize **activations only** (weights bf16), the dominant term in the denominator may remain $2DF$, leaving $I$ and the threshold nearly unchanged. Always recompute the bytes expression to see which terms dominate.
 
---

## 7. Optimization Strategies

### 7.1 When Bandwidth-Bound ($I_{\text{alg}} < I^*$)

**Increase Arithmetic Intensity:**
- Increase local batch size $B$ (primary lever for matmul)
- Tile operations for better cache reuse
- Fuse adjacent operations to avoid intermediate writes

**Reduce Bytes Moved:**
- Quantize weights/activations (int8, int4)
- Compress communicated tensors
- Use on-chip memory (VMEM) effectively

**Increase Bandwidth:**
- Use faster memory tiers
- Optimize data layout for sequential access
- Choose better network topology for distributed ops

### 7.2 When Compute-Bound ($I_{\text{alg}} > I^*$)

**Reduce FLOPs:**
- Algorithmic improvements (sparsity, low-rank approximations)
- Smaller model dimensions
- Better numerical algorithms

**Increase Compute Capacity:**
- Use specialized units (MXU vs VPU)
- Add more devices (if network roofline permits)
- Improve kernel utilization
- Prefer sustained numbers (not vendor peaks) when estimating gains; calibrate with microbenchmarks.
 
---

## 8. Extensions You'll Likely Need (Still "Simple" to Reason With)

### 8.1 Multiple Roofs for a Memory Hierarchy

One slanted line per level (L1, L2, HBM, network), each with its own $\beta$. For a given kernel, pick the link that actually carries the bytes you counted.

### 8.2 Ceilings Inside the Roof

Practical limits (e.g., no FMA use, limited ILP, non-coalesced loads) create lower "ceilings" beneath the theoretical roof; they order your optimization to-dos.

### 8.3 Communication Rooflines

Treat inter-chip links exactly like memory roofs: bytes are messages across the fabric; bandwidth is the link's sustained GB/s. Same math, different $\beta$.

### 8.4 Multi-Level Rooflines

Create separate rooflines for:
- HBM bandwidth
- Inter-chip network (ICI)
- Data center network (DCN)
- PCIe bandwidth
- CPU RAM

Each has its own critical intensity and optimization strategies. For a given kernel, form **per-level intensities** using the **bytes that hit that level** and take the **minimum** performance across roofs. In time terms, $T_{\min}=\max(T_{\text{math}},T_{\text{HBM}},T_{\text{ICI}},\dots)$ when those paths do not overlap.

---

## 9. Common Gotchas (So You Don't Over-Trust the Picture)

### 9.1 Critical Pitfalls

- **Counting the wrong bytes.** Roofline cares about bytes **on the link you modeled** (e.g., DRAM↔SM, not L1↔register). Mismatch here breaks everything.
- **Precision mismatch.** $\pi$ and $\beta$ are precision- and unit-specific (FP16 tensor cores vs FP32 CUDA cores). Mix them and the ridge moves invisibly.
- **No overlap assumption.** The lower bound assumes good overlap; if your kernel cannot overlap compute and movement, actual time trends toward $T_{\text{math}}+T_{\text{comms}}$.
- **Pathological access patterns.** Irregular or strided accesses can cut sustained $\beta$ far below "spec," lowering the slanted roof. Measure sustained bandwidth, don't lift the spec sheet.

### 9.2 Other Pitfalls to Avoid

- **Wrong unit**: VPU vs MXU have different rooflines
- **Missing bytes**: Remember to count writes and network traffic
- **Peak vs sustained**: Vendor peaks assume ideal conditions
- **Overlap optimism**: Real overlap rarely perfect
- **Multiple bottlenecks**: May need separate rooflines for HBM, network, etc.

### 9.3 Special Considerations

- **Dot products**: Often run on VPU with different critical intensity
- **Mixed precision**: Changes bytes asymmetrically
- **Sharding effects**: Network roofline may depend on different dimensions
- **Memory hierarchy**: Each level (VMEM, HBM, network) has its own roofline

### 9.4 Edge Cases & Diagnostics

- **Write-dominant small-$B$:** $2BF$ output writes can dominate; intensity rises with $B$ until the $DF$ term dominates.
- **Small-$D$ shapes:** The $BD$ term becomes non-negligible; re-evaluate $I \approx \frac{BDF}{BD + DF + BF}$ without dropping terms.
- **Contention/variance:** In multi-tenant environments, effective bandwidth fluctuates; prefer percentiles (e.g., p95) for planning.

---

## 10. Quick Acceptance Checks You Can Run

### 10.1 Validation and Testing

**Sanity math:** For your kernel, compute $I=F/B$. Given $\pi,\beta$, verify $I$ vs $I^*=\pi/\beta$ and compute $P_{\max}=\min\lbrace \pi,\beta I\rbrace$.

**Time bound:** Check a profile's wall time $T$ obeys $T\ge \max(F/\pi,\ B/\beta)$. If it doesn't, your counts or peaks are off.

**Movement lever test:** Apply one change that should raise $I$ (e.g., fuse two adjacent ops). If you were bandwidth-bound, measured $P$ should increase roughly in proportion to the $I$ gain until you hit $\pi$.

### 10.2 Acceptance Tests

**Test 1 - Verify Classification:**
- Calculate $I_{\text{alg}}$ and $I^*$
- If predicting compute-bound but utilization is low, bytes are undercounted

**Test 2 - Time Bounds:**
- Measure actual runtime
- Verify $T_{\text{measured}} \in [T_{\min}, T_{\max}]$
- If outside bounds, missing communication or compute

**Test 3 - Lever Validation:**
- Double $B$ per device
- Compute-bound: time unchanged
- Bandwidth-bound: performance scales with $B$ until knee

**Test 4 - Dtype Changes (disambiguated):**
- **Mixed precision** (bf16 compute, **int8 weights**): threshold should drop ≈2× (e.g., v5e ≈240→≈120).
- **Full int8** (compute + activations + weights): threshold should be ~unchanged (both $I_{\text{alg}}$ and $I_{\text{hw}}$ scale).

**Test 5 - Distributed Flip (2-chip, shard on $D$):**
- Vary $D$ across $\lbrace 4\text{k}, 8\text{k}, 16\text{k}\rbrace$ with fixed $(B,F)$.
- Expect compute↔comms crossover near $D \approx 8755$ for the example numbers in §4.3.

### 10.3 How to Measure FLOPs/bytes (so the math meets the metal)

- Use your framework/compiler profiler to export **FLOP counts** and **per-tensor bytes** (reads, writes, and collectives).  
- Validate the **HBM bytes** against your hand-count (tensor sizes × dtypes × read/write multiplicity).  
- Attribute **collectives** (e.g., all-reduce) bytes to the **inter-chip** roofline, not HBM.  
- Re-run small **microbenchmarks** to estimate **sustained** compute/bandwidth and use these in $I^*$.

---

## 11. Practical Implementation

### 11.1 Quick Decision Framework

| Observation | Bottleneck | Primary Optimization |
|------------|------------|---------------------|
| $I \ll I^*$ (HBM) | HBM bandwidth | Increase batch, fuse ops, quantize |
| $I \ll I^*$ (network) | Network bandwidth | Change sharding, compress comms |
| $I \approx I^*$ | At the ridge | Small changes can flip the regime |
| $I \gg I^*$ | Compute | Reduce FLOPs or add compute |

### 11.2 Key Rules of Thumb

**TPU v5e (MXU):**
- Critical intensity: ~240 FLOPs/byte
- Matmul compute-bound when local $B \gtrsim 240$
- Mixed precision (int8 weights) reduces threshold to $B \gtrsim 120$

**GPU H100:**
- Critical intensity: ~295 FLOPs/byte
- Matmul compute-bound when local $B \gtrsim 300$

### 11.3 Example Python Implementation

```python
def roofline_matmul_multi(
    B, D, F,
    bytes_per_x, bytes_per_y, bytes_per_z,
    peak_flops_per_s,
    bandwidth_by_level,      # dict like {'hbm': 8.2e11, 'ici': 4.5e10, ...}
    bytes_by_level=None      # dict of per-level bytes, overrides HBM if given
):
    """
    Returns arithmetic intensity vs HBM, per-level comm times, and [T_min, T_max].
    If bytes_by_level is None, HBM bytes are computed from (X,Y,Z) and used for 'hbm'.
    Lower bound assumes compute can overlap with each comm level independently and
    takes max(T_math, *T_comm.values()). Upper bound sums non-overlapped comm.
    """
    flops = 2*B*D*F
    # Default HBM bytes from tensor dtypes if not supplied explicitly
    if bytes_by_level is None:
        bytes_hbm = bytes_per_x*B*D + bytes_per_y*D*F + bytes_per_z*B*F
        bytes_by_level = {'hbm': bytes_hbm}
    T_math = flops / peak_flops_per_s
    T_comm = {lvl: (bytes_by_level.get(lvl, 0.0) / bw) for lvl, bw in bandwidth_by_level.items()}
    # Arithmetic intensity relative to HBM
    I_alg_hbm = flops / max(bytes_by_level.get('hbm', 1.0), 1e-30)
    I_hw_hbm  = peak_flops_per_s / bandwidth_by_level.get('hbm', float('inf'))
    bound_min = max(T_math, *(T_comm.values()))
    bound_max = T_math + sum(T_comm.values())
    return dict(
        I_alg_hbm=I_alg_hbm,
        I_hw_hbm=I_hw_hbm,
        T_math=T_math,
        T_comm=T_comm,
        bound_min=bound_min,
        bound_max=bound_max,
        is_compute_bound_hbm=(I_alg_hbm > I_hw_hbm),
    )
```

---

## 12. Advanced Topics

### 12.1 Per-Example Weight Matrices

Operation: $X[B,D] \cdot Y[B,D,F] \to Z[B,F]$
- **Intensity**: $I = \frac{DF}{D+DF+F} \xrightarrow[\text{large } D,F]{} 1$ FLOP/byte
- **Result**: Extremely bandwidth-bound pattern

### 12.2 Overlap and Pipelining

- Design for concurrent compute and communication
- Use double-buffering for weight transfers
- Well-structured programs achieve near-optimal overlap
- If runtime equals sum (not max), improve overlap strategy

---

## References

- Williams, Waterman, Patterson, *Roofline: An Insightful Visual Performance Model for Floating-Point Programs and Multicore Architectures*, CACM (2009). Core definition, ridge point, and the $P_{\max}=\min(\pi,\beta I)$ formula.
- Austin et al., Google DeepMind, 2025. "All About Rooflines" (device knees, VPU/MXU specifics, int8 and mixed-precision worked examples)
- NERSC Roofline documentation. How to place kernels on the chart and compute $P = \text{FLOPs}/\text{Runtime}$; practical measurement guidance.

---

## Quick Reference Card

**Essential Formula:**
$$P_{\max}=\min\lbrace \pi,\ \beta\cdot I\rbrace\quad\text{where}\quad I=\frac{F}{B}$$

**Decision Rule:**
$$\text{Compute-bound if: } I = \frac{\text{FLOPs}}{\text{bytes}} > I^* = \frac{\text{peak FLOPs/s}}{\text{bandwidth}}$$

**TPU v5e Quick Facts:**
- Critical intensity (ridge point): $I^* = 240$ FLOPs/byte
- Matmul needs $B \gtrsim 240$ to saturate MXU
- **Mixed-precision (bf16 compute × int8 weights)** reduces threshold to **$B \gtrsim 120$**
- **Full int8 (compute+activations+weights)** leaves the threshold **~unchanged** (≈240→≈243), since both $I_{\text{alg}}$ and $I_{\text{hw}}$ scale similarly

**Optimization Decision Tree:**
1. Calculate arithmetic intensity $I = F/B$
2. Compare to hardware critical intensity $I^* = \pi/\beta$
3. If bandwidth-bound ($I < I^*$): increase batch/reuse, reduce bytes
4. If compute-bound ($I > I^*$): optimize kernels, reduce FLOPs
