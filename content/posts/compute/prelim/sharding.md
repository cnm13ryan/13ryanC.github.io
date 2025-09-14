---
date: "2025-09-14"
title: "Sharded Matrix Multiplication on TPUs - Theory, Cost Models, and Practice"
summary: "How to break up big matrices and run them on many chips at once?"
lastmod: "2025-09-14"
category: "Notes"
series: ["Compute"]
author: "Bryan Chan"
hero: /assets/images/hero3.png
image: /assets/images/card3.png
---

## Goal and Scope
- This guide provides a **complete mental model** for matrix multiplication with sharded arrays on TPUs, grounded in the cost of communication primitives. 
- You'll understand not just *what* happens, but *why* it happens, *how much it costs*, and *how to optimize* your decisions based on quantitative analysis.

**Key Promise**: After reading this, you can predict exactly what collectives will be inserted, calculate their time cost to microsecond precision, and make optimal sharding decisions based on byte-math rather than trial and error.

## 0. Notation, Assumptions, and Ground Rules

### Array and Mesh Notation
- **Arrays**: C = A·B where A[I,J] and B[J,K] produce C[I,K]. The **contracting dimension** is J.
- **Device mesh**: Logical grid with named axes (X, Y, Z). Written as A[I_X, J_Y] meaning:
  - I is sharded across mesh axis X
  - J is sharded across mesh axis Y  
  - Unsubscripted dimensions are replicated
- **Multi-axis sharding**: A[I_XY] shards I over flattened XY. Order matters for layout (I_XY ≠ I_YX in terms of tiling).

### Fundamental Sharding Constraint
**Critical Rule**: You cannot shard two different array dimensions along the same mesh axis. A[I_X, J_X] is **invalid**. Each mesh axis can only be "spent" once per tensor.

### Communication Primitive Cost Model (Throughput-Bound)

For volume V bytes on a bidirectional ring with ICI bandwidth W_ici:

$$T_{AllGather} = T_{ReduceScatter} = \frac{V}{W_{ici}}$$

This is **independent of shard count** because rings pipeline. AllReduce is implemented as ReduceScatter + AllGather:

$$T_{AllReduce} = 2 \cdot \frac{V}{W_{ici}}$$

AllToAll (resharding) in bidirectional rings is remarkably efficient:

$$T_{AllToAll} \approx \frac{1}{4} \cdot \frac{V}{W_{ici}}$$

**Why factor of 4?** Shards only travel part of the ring, and traffic splits left/right. The detailed derivation shows each shard travels distance X/4 on average in a bidirectional topology.

### Multi-Axis Collective Bandwidth

When gathering over n mesh axes simultaneously, effective bandwidth scales:

$$T = \frac{V}{n \cdot W_{ici}}$$

This is why 2D and 3D meshes are powerful - you get additive bandwidth across independent axes.

### Latency Regime Boundary

**Critical threshold**: On v5e, any buffer < 45 KB per link is latency-bound (~1μs per hop). 

Time in latency regime:
$$T = n_{hops} \cdot T_{hop} + \frac{V}{W_{ici}}$$

where n_hops depends on ring length and topology (wrap-arounds matter here).

## 1. TPU Hardware Reality Check

### Bandwidth Hierarchy (v5p Reference)
| Component | Bandwidth | Relative Speed | Use For |
|-----------|-----------|----------------|---------|
| **HBM** | ~2.5 TB/s | 1x (baseline) | Local compute |
| **ICI** | ~90 GB/s/axis | 1/28x of HBM | Intra-slice collectives |
| **DCN** | ~6.25 GB/s | 1/14x of ICI | Last resort only |

**Topology Gotchas**:
- Smaller requested topologies may lack wrap-arounds → ring paths double → 2x communication time
- DCN is catastrophically slow - a single DCN hop can dominate your entire communication budget
- Full torus axes have wrap-arounds that halve hop distances

## 2. The Four Canonical Cases - Complete Analysis

### Case 1: Neither Multiplicand Shards the Contracting Dimension J

**Patterns**: 
- A[I_X, J], B[J, K_Y] → C[I_X, K_Y]
- A[I, J], B[J, K] → C[I, K] (fully replicated)

**Why it works**: Every device sees a complete local contracting slice of J. Block multiplication proceeds without communication.

**Communication cost**: **ZERO**

**Mental model**: This is embarrassingly parallel - each device computes its block independently.

### Case 2: Exactly One Multiplicand Shards J

**Pattern**: A[I, J_X], B[J, K] (or symmetric case)

**The fundamental choice** - two mathematically equivalent strategies:

#### Strategy 2a: Gather-then-Matmul
```
1. AllGather A along X → A_full[I, J]
2. C = A_full @ B (local matmul)
```
**Cost**: $T = \frac{V_A}{W_{ici}}$ where $V_A = I \cdot J \cdot \text{bytes_per_element}$

#### Strategy 2b: Matmul-then-ReduceScatter  
```
1. C_partial = local_matmul(A, B)  # Unreduced partial sums
2. ReduceScatter C_partial along X → C[I, K_X]
```
**Cost**: $T = \frac{V_C}{W_{ici}}$ where $V_C = I \cdot K \cdot \text{bytes_per_element}$

#### The M/K Decision Rule

The communication ratio is:
$$\frac{\text{Gather cost}}{\text{ReduceScatter cost}} = \frac{V_A}{V_C} = \frac{I \cdot J}{I \cdot K} = \frac{J}{K} = \frac{M}{K}$$

**Decision algorithm**:
- If M << K: AllGather the input (moving less data)
- If K << M: ReduceScatter the output (moving less data)
- If M ≈ K: Consider memory pressure and downstream needs

### Case 3: Both Multiplicands Shard J Identically

**Pattern**: A[I, J_X], B[J_X, K]

**What happens**: Local matmul produces unreduced partial sums. Each device has computed part of each output element.

**Two collective options**:

1. **AllReduce**: Everyone gets full C (replicated)
   - Cost: $T = 2 \cdot \frac{V_C}{W_{ici}}$
   - Memory: Full C on every device
   - Use when: Downstream needs replicated data

2. **ReduceScatter**: Distributed C[I, K_X] or C[I_X, K]
   - Cost: $T = \frac{V_C}{W_{ici}}$ (half of AllReduce!)
   - Memory: Only 1/X of C per device
   - Use when: Downstream accepts sharded data

**Critical decision**: Choose ReduceScatter axis to match next layer's expected layout:
- RS over K if next layer expects column sharding
- RS over I if next layer expects row/batch sharding

### Case 4: Same Non-Contracting Dimension on Same Axis (The Diagonal Trap)

**Pattern**: A[I_X, J], B[J, K_X]

**Why it's invalid**: Device x has rows I_x and columns K_x. It can only compute C[I_x, K_x] - the block diagonal! To compute C[I_x, K_≠x] it needs columns of B it doesn't have.

**Solution**: AllGather one input first
```
B_full = AllGather(B, axis='X')  # Cost: V_B / W_ici
C = A @ B_full  # Now valid, produces C[I_X, K]
```

## 3. Detailed Worked Example with Real Numbers

**Setup**: 
- A ∈ BF16^{128×8192}, B ∈ BF16^{8192×32768}
- Single axis ring, W_ici = 90 GB/s bidirectional
- Contracting dimension J = 8192

### Option A: Gather-then-Matmul (Case 2a)
```
Volume: V_A = 128 × 8192 × 2 bytes = 2,097,152 bytes
Time: T_AG = 2,097,152 / (90 × 10^9) = 23.3 μs
```

### Option B: Matmul-then-ReduceScatter (Case 2b)
```
Volume: V_C = 128 × 32768 × 2 bytes = 8,388,608 bytes  
Time: T_RS = 8,388,608 / (90 × 10^9) = 93.2 μs
```

### Option C: If we mistakenly used AllReduce
```
Time: T_AR = 2 × 8,388,608 / (90 × 10^9) = 186.4 μs
```

**Analysis**: 
- M/K = 8192/32768 = 1/4
- Gather is 4× cheaper than ReduceScatter (exactly as predicted!)
- AllReduce would be 8× more expensive than optimal choice

## 4. Advanced Cost Optimizations

### Memory vs. Bandwidth Trade-offs

**Memory pressure scenario**: Even if M/K suggests AllGather, memory constraints may force ReduceScatter:

```python
# Memory accounting for replication
if array.sharded_over(S) and replicated_over(R):
    per_device_bytes = global_bytes / ∏(a∈S)|a|
    total_cluster_bytes = global_bytes × ∏(a∈R)|a|
```

**Rule**: Replication across n axes multiplies memory usage by n even when compute is unchanged.

### AllToAll: The Secret Weapon

AllToAll is 4× cheaper than AllGather for the same volume:

**Use cases**:
- Transposes between layouts: A[I_X, J_Y] → A[I_Y, J_X]
- Resharding for algorithm changes
- Switching between data and model parallelism

**Why it's cheaper**: In a bidirectional ring, each shard only travels ~1/4 of the ring distance vs. full ring for AllGather.

### Exploiting Multi-Axis Bandwidth

```python
# Single axis: 90 GB/s
P_x = PartitionSpec('X', None)  # T = V / 90GB/s

# Two axes: 180 GB/s effective
P_xy = PartitionSpec(('X','Y'), None)  # T = V / 180GB/s

# But beware layout implications!
P_xy ≠ P_yx  # Different tiling patterns
```

## 5. Design Playbook with Quantitative Rules

### Decision Tree for Every Matmul

```
1. Can you avoid sharding J?
   YES → Case 1, zero communication, YOU WIN
   NO → Continue to 2

2. Can only one side shard J?
   YES → Case 2, compute M/K ratio:
         M/K < 0.5 → Strong preference for AllGather
         0.5 < M/K < 2 → Consider memory and downstream
         M/K > 2 → Strong preference for ReduceScatter
   NO → Continue to 3

3. Both must shard J?
   → Case 3, prefer ReduceScatter over AllReduce (2× cheaper)
   → Choose RS axis based on downstream layout needs

4. Verify no Case 4 (diagonal trap)
   If A[I_X,J] and B[J,K_X] → Must AllGather one first
```

### Acceptance Checks (Do These Before Running)

1. **Case identification**: Label each matmul 1-4. If unclear, your spec is ambiguous.

2. **Byte tally**: 
   ```
   V_moved = array_elements × bytes_per_element × axes_involved
   ```

3. **Time estimate**:
   ```
   T_comm = V_moved / (n_axes × W_ici × topology_factor)
   where topology_factor = 0.5 if missing wrap-arounds
   ```

4. **Memory bound**:
   ```
   peak_memory = max(input_memory + output_memory + intermediate_memory)
   Verify: peak_memory < 0.8 × HBM_capacity (leave headroom)
   ```

5. **Topology fit**: 
   - Verify axes have wrap-arounds (check topology specs)
   - Confirm no DCN crossings (14× penalty)

## 6. Common Pitfalls with Quantitative Impact

| Pitfall | Quantitative Impact | Detection | Solution |
|---------|-------------------|-----------|----------|
| **Memory blowup from AllGather** | OOM when array > HBM/n_devices | Check V_gathered > HBM capacity | Use ReduceScatter path |
| **Latency trap (small tensors)** | 10-100× slower for V < 45KB | Time doesn't match V/W formula | Batch to > 45KB chunks |
| **Missing wrap-arounds** | 2× communication time | Topology has open edges | Request full torus |
| **Resharding churn** | n×(V/4W) for n reshards | Multiple AllToAlls in profile | Align layouts across layers |
| **DCN crossings** | 14× slower (6.25 vs 90 GB/s) | Cross-slice communication | Restructure to stay intra-slice |
| **Wrong RS axis choice** | Extra V/4W for immediate reshard | AllToAll right after matmul | Match RS axis to next op |

## 7. Minimal Implementation Recipes

### JAX/XLA Pseudocode for Each Case

```python
# Case 1: No J sharding (zero communication)
# A: [I_X, J], B: [J, K_Y] → C: [I_X, K_Y]
C = jnp.dot(A, B)  # No collective inserted

# Case 2a: AllGather approach
# A: [I, J_X], B: [J, K]
A_full = lax.all_gather(A, axis='X', axis_index_groups=...)
C = jnp.dot(A_full, B)  # Time: V_A / W_ici

# Case 2b: ReduceScatter approach  
# A: [I, J_X], B: [J, K]
C_partial = jnp.dot(A_local, B_local)  # Local partial
C = lax.psum_scatter(C_partial, axis='X', scatter_dimension=1)  # Time: V_C / W_ici

# Case 3: Both shard J
# A: [I, J_X], B: [J_X, K]
C_unreduced = jnp.dot(A_local, B_local)
C = lax.psum_scatter(C_unreduced, axis='X', scatter_dimension=1)  # Time: V_C / W_ici
# Or: C = lax.psum(C_unreduced, axis='X')  # Time: 2 * V_C / W_ici

# Case 4: Diagonal trap resolution
# A: [I_X, J], B: [J, K_X] (invalid as-is)
B_fixed = lax.all_gather(B, axis='X')  # Time: V_B / W_ici
C = jnp.dot(A, B_fixed)
```

## 8. Why This Model Is Trustworthy (And When It Breaks)

### Why It Works

1. **Ring algorithms pipeline**: Time depends on bytes and bandwidth, not device count (when throughput-bound)
2. **Bidirectional links double bandwidth**: Each device can send and receive simultaneously
3. **AllReduce = RS + AG**: This is the actual implementation, so 2× cost is exact
4. **AllToAll's 4× advantage**: Mathematical result from ring distance analysis

### Model Breakdown Conditions

1. **Latency-bound regime** (V < 45KB): Add hop latency term
2. **Asymmetric bandwidth**: Some TPU configs have asymmetric X/Y/Z speeds
3. **Network contention**: Multiple jobs sharing DCN
4. **Compiler optimizations**: Sometimes XLA finds clever fusion opportunities
5. **Tree algorithms**: For certain patterns, compilers might choose trees over rings

**Mitigation**: Always profile to verify, but use model for design decisions.

## 9. Advanced Topics and Subtleties

### Multi-Axis Sharding Order Matters

```python
# These are different!
I_XY: I is sharded X-major (rows of X, columns of Y)
I_YX: I is sharded Y-major (rows of Y, columns of X)

# Impact: Different cache patterns, different resharding costs
```

### The Unreduced Dimension Choice in Case 3

When both shard J, after local matmul you have unreduced C. You must reduce, but can choose where to place the result:

```python
# Option 1: Reduce and scatter to K axis
C[I, K_X] = reduce_scatter(C_unreduced, axis='X', dim=K)

# Option 2: Reduce and scatter to I axis  
C[I_X, K] = reduce_scatter(C_unreduced, axis='X', dim=I)

# Option 3: Reduce to replicated (costs 2×)
C[I, K] = all_reduce(C_unreduced, axis='X')
```

Choose based on what the next operation expects.

### Communication-Computation Overlap

Overlap is possible but requires careful orchestration:

```python
# Conceptual pipeline (actual implementation is compiler-dependent)
for tile in range(n_tiles):
    future = start_all_gather(A[tile+1])  # Non-blocking
    C[tile] = matmul(A[tile], B[tile])   # Compute current
    A[tile+1] = wait(future)              # Block for next
```

**Reality check**: Verify overlap in timeline profiler; don't assume it happens.

### Topology Awareness for Performance

```python
# Full torus (wrap-arounds on all axes): Optimal
# Time = V / W_ici

# Missing wrap-arounds: ~2× penalty
# Time = 2 * V / W_ici

# DCN crossing: ~14× penalty
# Time = V / W_dcn where W_dcn = W_ici / 14
```

## 10. Complete Verification Checklist

Before running:
- [ ] Each tensor axis maps to at most one mesh axis
- [ ] No A[I_X, J_X] style reuse
- [ ] Each matmul classified as Case 1-4
- [ ] M/K ratio computed for Case 2 decisions
- [ ] ReduceScatter preferred over AllReduce where applicable
- [ ] RS axis matches downstream layout needs
- [ ] Byte volumes calculated for all collectives
- [ ] Time estimates sum to less than compute budget
- [ ] Peak memory < 80% HBM capacity
- [ ] Topology has required wrap-arounds
- [ ] No unnecessary DCN crossings

After running (profile verification):
- [ ] Collectives match predictions (AG before compute, RS/AR after)
- [ ] No unexpected AllToAlls from resharding
- [ ] Communication overlaps with compute where expected
- [ ] No back-to-back AG↔A2A↔RS patterns
- [ ] Timeline shows expected throughput (V/T ≈ W_ici)

## Summary: The Mental Model

Sharded matrix multiplication on TPUs is governed by simple byte arithmetic:
- **Time = Bytes / Bandwidth** (when throughput-bound)
- **Four cases** determine which collectives are needed
- **M/K ratio** decides between gather-input vs reduce-output
- **ReduceScatter > AllReduce** when you don't need replication (2× savings)
- **AllToAll is 4× cheaper** than AllGather for resharding
- **Multi-axis meshes** give additive bandwidth
- **Stay intra-slice** to avoid 14× DCN penalty

With this model, you can predict communication patterns before writing code, estimate performance before running experiments, and optimize layouts based on quantitative analysis rather than trial and error. The key insight is that TPU communication is predictable and follows simple rules—master these rules and you master distributed performance.
