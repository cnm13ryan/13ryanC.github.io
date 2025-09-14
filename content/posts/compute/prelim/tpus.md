---
date: "2025-09-14"
title: "How to think about TPUs"
summary: "How do TPUs work? How does that affect what models we can train and serve?"
lastmod: "2025-09-14"
category: "Notes"
series: ["Compute"]
author: "Bryan Chan"
hero: /assets/images/hero3.png
image: /assets/images/card3.png
---

> **As of:** 2025-09. Specs and topologies evolve; numbers below cite current product docs.  
>
> **Units & Metrics Legend**  
> • **GB** = 10^9 bytes; **GiB** = 2^30 bytes. We prefer **GB** unless the source uses **GiB**.  
> • **GB/s vs Gb/s:** 8 bits = 1 byte. Some docs report ICI/DCN in **Gb/s (aggregate)**; divide by 8 for **GB/s**.  
> • **Per-axis vs aggregate:** ICI may be listed **per axis (X/Y/Z)** or **aggregate** across axes/links—this patch labels which is used.  
> • **FLOPs vs TOPS:** A fused multiply-add (FMA) counts as **2 FLOPs**. **TOPS** generally refers to integer ops (e.g., int8) and is **not directly comparable** to FLOPs.

## The Simple Framework: "One Matmul Engine + Three Pipes + A Tiny Staging Buffer"

**Mental Model:** Reduce a TPU to **five fundamental boxes** to reason about performance on the back of an envelope:

1. **MXU (Matrix Multiply Unit)** – A systolic array that consumes tiles of matrices and produces results at enormous throughput. This is why TPUs are fast. The MXU operates best when both matmul dimensions are padded to array size (128 on v2–v5, 256 on v6e).
   - v5e chip: ~197 TFLOPs bf16 peak
   - v5p chip: ~459 TFLOPs bf16 peak  
   - v6e chip: ~918 TFLOPs bf16 peak

2. **VPU (Vector Processing Unit)** – Handles pointwise ops (ReLU, add, scale), reductions, and control-side math. These ops are often bandwidth-bound unless fused or overlapped with MXU work.

3. **VMEM (On-chip Scratchpad)** – Small but extremely fast staging SRAM near the MXU (~128 MiB on v5e; ~20× HBM bandwidth order-of-magnitude). If your working set fits here, you can be compute-bound at much lower arithmetic intensity. Use it to prefetch weights/tiles.

4. **HBM (High Bandwidth Memory)** – Tens of GB per chip, TB/s-class bandwidth. This is the main pipe feeding compute:
   - v5e: 16 GB @ ~819 GB/s
   - v5p: ~95 GB @ ~2,765 GB/s (2.8 TB/s)
   - v6e: 32 GB @ ~1,600 GB/s

5. **Interconnects** – Two "outer" pipes:
   - **ICI (Inter-Core Interconnect)**: A nearest-neighbor torus (2D for v5e/v6e; 3D for v5p) for sharded training. Fast, but still well below HBM.
     - v5e: ~200 GB/s aggregate (from 1,600 Gb/s)
     - v5p: ~90 GB/s per axis × 3 axes
     - v6e: ~400 GB/s aggregate (from 3,200 Gb/s)
   - **DCN/PCIe (pod↔pod & host↔TPU)**: Much slower; touch sparingly in the training hot path. Typical PCIe ~16 GB/s each way; v5p DCN egress ~6.25 GB/s per TPU.

> **Bottom line:** Think of a TPU as a **matmul appliance** bounded by **HBM** and **ICI** pipes, with **VMEM** as your precious staging cache. Balance compute with the right pipe.

## The Two Questions That Decide Performance

### 1) Are you compute-bound or bandwidth-bound?

Use the **roofline model**: *Attainable FLOPs* ≤ *min*(Peak FLOPs, **AI** × Bandwidth), where **AI (arithmetic intensity)** is FLOPs per byte moved across that level (HBM, VMEM, ICI).

If **AI ≥ Peak FLOPs/BW**, you're compute-bound; otherwise, you're bandwidth-bound.

**Critical thresholds at HBM level (FLOPs/byte needed to be compute-bound):**
- **v5e:** 197e12 / 819e9 ≈ **240 FLOPs/byte**
- **v5p:** 459e12 / 2,765e9 ≈ **166 FLOPs/byte**  
- **v6e:** 918e12 / 1,600e9 ≈ **574 FLOPs/byte**

For a GEMM of shape **m×k · k×n** (bf16 in/out), the arithmetic intensity at HBM level is:

$$\text{AI}_{HBM} \approx \frac{2mkn}{(mk + kn + mn)s}$$

where $s$ is bytes/element (bf16: $s=2$). This assumes single pass over inputs, output written once, tiles reused from VMEM while resident.

### 2) If you shard, does the ICI cost swamp the MXU?

Treat cross-chip collectives (all-gather/reduce-scatter) as extra "bytes over a slower pipe":

$$T_{comm} \approx \frac{\text{bytes moved per chip}}{\text{per-axis ICI BW}} \times \text{(effective hop factor)}$$

A torus reduces max distance to ~N/2 along that dimension; missing wraparounds often ~2× the time. Keep per-step ICI traffic well below what would stall the MXU given its compute time.

## TPU Architecture: The Matrix Multiply Machine

TPUs are fundamentally "matrix multiply machines attached to fast memory." Each TPU chip contains one or more MXUs that perform dense matmul operations at extremely high throughput, plus VPUs and support logic. 

**Current chip-level specs (as of 2025-09)**  
_(rounded; per chip; bf16 unless noted)_
- **TPU v5e** – **197 TFLOPs**; **16 GB HBM @ 819 GB/s**; **ICI 1,600 Gb/s (aggregate)**; **2D torus**; **Pod size: 256 chips**
- **TPU v5p** – **459 TFLOPs**; **95 GB HBM @ 2,765 GB/s**; **ICI 4,800 Gb/s (aggregate)**; **3D torus** (wraparound for cube slices); **Pod size: up to 8,960 chips**
- **TPU v6e (Trillium)** – **918 TFLOPs**; **32 GB HBM @ 1,600 GB/s**; **ICI 3,200 Gb/s (aggregate)**; **2D torus**; **Pod size: 256 chips**
- **MXU tile size:** **128×128** on v5e/v5p; **256×256** on v6e

**Key Design Principle:** Structure computations to maximize MXU utilization - feed large matrices or batches of smaller matrices, avoid serialized scalar or small-vector operations.

## Memory Hierarchy and Data Movement

The TPU memory system has three tiers with vastly different bandwidths:

### 1. HBM (High Bandwidth Memory) – Off-chip params/activations
- Capacity (per chip): **v5e 16 GB**, **v5p 95 GB**, **v6e 32 GB**
- Bandwidth (per chip): **v5e 819 GB/s**, **v5p 2,765 GB/s**, **v6e 1,600 GB/s**
- Rule of thumb: HBM ≫ ICI ≫ DCN/PCIe

### 2. VMEM (Vector Memory) – On-chip SRAM scratchpad
- Small capacity (~128 MiB on v5e, varies by generation)
- **~20× HBM bandwidth** order-of-magnitude
- Explicitly managed buffer for staging tiles for MXU/VPU
- Critical for achieving compute-bound operation at lower arithmetic intensities

### 3. System Memory / Host I/O – Host DRAM accessed via PCIe/NIC
- Host↔TPU bandwidth is **O(10^10 B/s)** and deployment-dependent
- Treat this as the slowest tier for training inner loops

**Pipelined Execution:** TPUs hide memory latency through pipelining - while one chunk is being multiplied in the MXU, the next chunk is loading from HBM to VMEM. This overlap keeps the MXU busy continuously, allowing well-optimized matmuls to be compute-bound despite memory transfer needs.

**Optimization Strategy:** 
- Maximize arithmetic intensity by reusing data once loaded to VMEM
- Multiply a weight matrix chunk with multiple activation batches before evicting it
- Use compiler features like operation fusion to keep intermediate results on-chip
- Prefetch next layer's weights into VMEM during unrelated compute (attention ↔ FFN)

## Hardware Constraints and Padding

The MXU's systolic array operates on fixed tiles:
- **128×128** on v5e/v5p
- **256×256** on v6e

**Alignment requirement:** Choose dimensions that are multiples of the MXU tile size.

Any matrix with dimensions not divisible by the MXU tile gets padded automatically by the compiler, performing extra work on padded zeros.

**Padding overhead examples:**
- `K=100` on a **128-wide** MXU → pad to 128 ⇒ overhead = (128−100)/128 = **21.9%**
- `K=6300` on a **256-wide** MXU → pad to 6400 ⇒ overhead = (6400−6300)/6400 = **1.56%**

**Best Practice:** Choose model dimensions (hidden sizes, attention heads) as multiples of 128/256 to avoid wasted compute on padding. Prefer batch/blocking that forms multiples of these tiles.

## Precision and Throughput Scaling

TPUs support multiple precision levels with corresponding throughput gains:
- **bfloat16:** Baseline matmul performance (spec TFLOPs are quoted in bf16)
- **int8:** Higher matmul peak on MXUs (e.g., v5e ~393 TOPS, v6e ~1,836 TOPS)
- **int4:** Even higher matmul peak possible on some generations/frameworks

**Important caveat:** VPU operations (elementwise adds, nonlinearities, reductions) typically use **fp32 accumulation**, so not everything scales with lower precision. Speedups primarily apply to **large, MXU-bound** matmuls.

## Network Topology and Communication

### Within a Pod
Topology differs by generation:
- **v5e/v6e:** **2D torus** slices
  - v5e: ICI ~200 GB/s aggregate (1,600 Gb/s)
  - v6e: ICI ~400 GB/s aggregate (3,200 Gb/s)
- **v5p:** **3D torus** with wraparound links for cube slices
  - ICI ~600 GB/s aggregate (4,800 Gb/s)
  - ~90 GB/s per axis × 3 axes

### Between Pods/Hosts
Communication beyond directly-connected chips uses slower paths:
- **PCIe/host egress:** O(10^10 B/s), host- and VM-shape-dependent
- **DCN between hosts:** Typically below ICI; treat as a scarce resource

### Communication Patterns
Since chips only connect to neighbors, collective operations occur via multi-hop routing:
- Data passes through intermediate chips in ring or tree patterns
- Latency grows with distance (each hop adds delay)
- Missing wraparounds can ~2× communication time
- Throughput can be maintained through pipelining

**Optimization Strategies:**
1. Structure algorithms to communicate primarily over ICI (within-pod) rather than DCN
2. Use reduce-scatter-first patterns to keep cross-chip bytes proportional to ICI bandwidth
3. Avoid shapes without wraparounds when collectives are hot
4. Keep per-step ICI traffic well below MXU compute time

## Balancing Communication Across Links

**Bandwidth Hierarchy (fastest → slowest):** HBM ≫ ICI ≫ DCN ≈ PCIe

**Per-gen bandwidth ratios (per chip):**
- **v5e:** HBM ~819 GB/s; ICI ~200 GB/s → HBM ≈ **4×** ICI
- **v5p:** HBM ~2,765 GB/s; ICI ~600 GB/s → HBM ≈ **4.6×** ICI
- **v6e:** HBM ~1,600 GB/s; ICI ~400 GB/s → HBM ≈ **4×** ICI

**Key Principle:** No single link should saturate well before compute completes. Budget ICI per step: list all collectives, approximate bytes, and ensure $T_{comm}$ stays comfortably under your MXU compute time.

## Practical Examples and Calculations

### Worked Example A: Big FFN GEMM (likely compute-bound)
v5e per-chip numbers, let $m=n=k=4096$:

- FLOPs = $2mkn = 2 \cdot 4096^3 = 137,438,953,472$
- HBM bytes (min) = $(mk+kn+mn)s = 3 \cdot 16,777,216 \cdot 2 = 100,663,296$ bytes
- **AI$_{HBM}$** = $137,438,953,472 / 100,663,296 ≈ 1,365.3$ FLOPs/byte ≫ 240
- **Result:** Compute-bound on HBM
- Compute time ≈ 137.44e9/197e12 ≈ 0.70 ms (if fully utilized)

### Worked Example B: Tiny GEMM (often HBM-bound unless in VMEM)
Let $m=n=k=128$:

- FLOPs = $2 \cdot 128^3 = 4,194,304$
- HBM bytes = $(16,384+16,384+16,384) \cdot 2 = 98,304$
- **AI$_{HBM}$** = $4,194,304 / 98,304 ≈ 42.67$ FLOPs/byte < 240
- **Result:** HBM-bound at HBM level
- If served from VMEM (threshold ~10-20 FLOPs/byte), can become compute-bound

### Worked Example C: Sharding sanity check (v5p)
Suppose each step needs a **64 MB** all-gather along one axis of length 8:

- Per-axis ICI ≈ **90 GB/s**
- Time per chip ≈ 64/90 ≈ 0.71 ms
- Lack of wraparounds can ~2× that
- If per-step MXU compute is ~0.8 ms, this comm will hurt utilization

### Generational Considerations
Different TPU versions require different strategies:
- **v5e** (16 GB HBM): Smaller per-chip capacity → more aggressive sharding/checkpointing
- **v5p** (95 GB HBM): Larger per-chip capacity → more replication or fuller layers per chip
- **v6e** (574 FLOPs/byte threshold): Demands higher arithmetic intensity → push reuse/fusion more aggressively

## How to Reason About Optimal Operations & Mechanisms

### Shape to the Array
- Pad matmul dims to **128** (v2–v5) or **256** (v6e) to fill the MXU
- Small or ragged dims waste lanes
- Prefer batch/blocking that forms multiples of these tiles

### Chase Arithmetic Intensity Up the Memory Hierarchy
- **HBM level:** Make GEMMs big enough so AI$_{HBM}$ ≥ Peak/HBM BW
- **VMEM level:** Tile so each weight/activation is reused many times while resident
- Prefetch next layer's weights into VMEM during unrelated compute

### Favor MXU-Heavy Kernels
- Put as much math as possible under a matmul/conv umbrella
- Leave only unavoidable pointwise ops to the VPU
- When VPUs dominate time in profiles, you're almost surely bandwidth-bound

### Shard to the Network You Actually Have
- **Within a slice:** Use data parallel or reduce-scatter-first patterns
- **Across slices (DCN):** Assume "slow lane" - stream checkpoints and offloads
- Never put DCN/PCIe on the critical path

### Exploit Precision and Core Features
- Lower precision matmuls (int8/int4) run faster (often ~2–4× vs bf16)
- Use bf16 accumulators for training; int8 for inference when accuracy allows

## Pitfalls & Trade-offs

- **Ragged dimensions** → padding overhead; underutilized MXU lanes
- **Microbatches too small** → GEMMs that fit MXU but not roofline (HBM-bound)
- **Communication first, compute second** → sharding with large all-gathers makes ICI dominate
- **Host traffic on critical path** → PCIe/DCN copies throttle by ~10–100× vs HBM

## Mini-Toolkit for Practice

1. **Compute your thresholds:** Memorize Peak/HBM BW for your chip
   - v5e ≈ **240** FLOPs/byte
   - v5p ≈ **166** FLOPs/byte
   - v6e ≈ **574** FLOPs/byte

2. **Estimate AI** of hot kernels with $2mkn/((mk+kn+mn)s)$; compare to threshold

3. **Budget ICI** per step: list all collectives, ensure $T_{comm}$ < MXU compute time

4. **If AI is too low, try:**
   - Bigger tiles/batches
   - Fusing ops around the GEMM
   - VMEM prefetch/tiling
   - Different sharding to keep reuse local

## Acceptance Tests (Quick Verification)

1. **Roofline check:** Given your three largest GEMMs, compute AI$_{HBM}$. Each should exceed the chip's Peak/HBM threshold or show VMEM tiling that raises effective AI above that.

2. **Profile verification:** MXU utilization should be high when AI ≥ threshold; when lower, HBM throughput should be near spec while MXU stalls rise.

3. **Communication check:** For any hot collective, $T_{comm}$ (bytes / per-axis ICI) should be ≪ the compute time of surrounding MXU work.

## Summary

TPUs excel at dense matrix multiplication through specialized hardware (MXUs) and multi-tier memory systems. The key insight is to think of them as **matmul appliances** with three pipes of different speeds. Effective utilization requires:

1. **Aligning computations** with hardware dimensions (multiples of 128/256)
2. **Maximizing arithmetic intensity** through data reuse and VMEM staging
3. **Using appropriate precision** for the task
4. **Understanding and respecting** the bandwidth hierarchy (HBM ≫ ICI ≫ DCN)
5. **Adapting communication patterns** to the mesh topology
6. **Balancing computation and communication** loads across all available links

The mental model is simple: **fill the array, raise AI (preferably in VMEM), and keep ICI off the critical path.**

## References

- JAX/ML *How to Think About TPUs* (2025): Clean mental model; VMEM/array sizing/prefetch; torus notes. https://jax-ml.github.io/scaling-book/tpus/
- Cloud TPU **v5e** product page: https://cloud.google.com/tpu/docs/v5e
- Cloud TPU **v5p** product page: https://cloud.google.com/tpu/docs/v5p
- Cloud TPU **v6e (Trillium)** product page: https://cloud.google.com/tpu/docs/v6e
- TPU system architecture overview: https://cloud.google.com/tpu/docs/system-architecture-tpu-vm
- Jouppi et al., "**TPU v4**: An Optically Reconfigurable Supercomputer" (2023): https://arxiv.org/abs/2304.01433
- Jouppi et al., "**In-Datacenter Performance Analysis of a TPU**" (ISCA 2017): https://dl.acm.org/doi/10.1145/3079856.3080246
