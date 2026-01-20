# Feature Planning for ttnn-visualizer-cli from TTNN Optimization Perspective

This document outlines the required features for ttnn-visualizer-cli based on the [tt-metal/skills/optimizing-ttnn-models](https://github.com/tenstorrent/tt-metal) TTNN optimization skills.

## Current Features

| Category | Features | Status |
|----------|----------|--------|
| Basic Info | Device info, operation list, tensor list | ✅ Implemented |
| Memory | Memory usage summary, L1 visualization | ✅ Implemented |
| Performance | Bottleneck detection, core efficiency analysis, Matmul/Conv analysis | ✅ Implemented |

---

## Required Features Derived from TTNN Optimization Skills

### 1. Data Format Analysis (Step 1)

Skills reference: `step-01-data-formats.md`

| Feature | Status | Priority | Description |
|---------|--------|----------|-------------|
| dtype distribution analysis | ⚠️ Partial | High | Distribution of bfloat16/bfloat8_b/float32 usage |
| Math Fidelity settings list | ✅ Implemented | - | Distribution of LoFi/HiFi2/HiFi3/HiFi4 |
| Weights vs activations format comparison | ❌ Not implemented | Medium | Check if bfloat8_b is used for weights |
| bfloat8_b usage optimization suggestions | ❌ Not implemented | Medium | Show potential 2x memory reduction |

**Optimization Rules:**
- Use `bfloat8_b` for weights (2x memory reduction)
- Use `bfloat16` for activations
- Start with LoFi Math Fidelity and increase only if PCC is insufficient

---

### 2. Tensor Layout Analysis (Step 2)

Skills reference: `step-02-tensor-layouts.md`

| Feature | Status | Priority | Description |
|---------|--------|----------|-------------|
| Layout distribution | ⚠️ Partial | High | Distribution of TILE vs ROW_MAJOR usage |
| Padding overhead analysis | ❌ Not implemented | Medium | Amount of padding for 32x32 tiles |
| Layout conversion cost analysis | ❌ Not implemented | Low | Detection of to_layout calls |

**Optimization Rules:**
- Use `TILE_LAYOUT` for compute operations
- Make shapes multiples of 32 when possible
- Minimize layout conversions

---

### 3. Memory & Sharding Analysis (Step 3)

Skills reference: `step-03-memory-sharding.md`

| Feature | Status | Priority | Description |
|---------|--------|----------|-------------|
| Sharding strategy distribution | ✅ Implemented | **Highest** | Distribution of HEIGHT/WIDTH/BLOCK/INTERLEAVED |
| L1 vs DRAM usage patterns | ✅ Implemented | - | Memory placement verification |
| Reshard operation detection | ✅ Implemented | High | Detect sharding changes between consecutive operations |
| Double buffering status check | ❌ Not implemented | Medium | Usage of enable_act_double_buffer |
| act_block_h setting analysis | ❌ Not implemented | High | Block height settings for Conv operations |

**Optimization Rules:**
- Keep hot data in L1, cold data in DRAM
- Maintain consistent sharding strategy across operation chains (minimize reshards)
- Enable double buffering for compute-bound operations
- Maximize `act_block_h_override` within L1 capacity

---

### 4. Metal Trace Analysis (Step 4)

Skills reference: `step-04-metal-trace.md`

| Feature | Status | Priority | Description |
|---------|--------|----------|-------------|
| Device vs E2E performance gap | ⚠️ Partial | **Highest** | Calculate host overhead from op_to_op_gap |
| Host overhead ratio | ⚠️ Partial | High | Ratio of host time to total execution time |
| Trace applicability check | ❌ Not implemented | Medium | Check if shapes are static |
| Static shape check | ❌ Not implemented | Medium | Detect operations with dynamic shapes |

**Optimization Rules:**
- Use Metal Trace when model is host-bound
- Trace is effective when Device vs E2E gap is large
- All tensor shapes must be static

---

### 5. Multi-CQ Analysis (Step 5)

Skills reference: `step-05-multi-cq.md`

| Feature | Status | Priority | Description |
|---------|--------|----------|-------------|
| Command queue configuration check | ❌ Not implemented | Medium | Usage of 1CQ vs 2CQ |
| I/O overlap efficiency | ❌ Not implemented | Medium | Overlap between transfers and computation |
| Transfer time vs compute time comparison | ❌ Not implemented | Medium | Detection of I/O bottlenecks |

**Optimization Rules:**
- Use 2CQ when I/O time is significant
- Input writes can run in parallel with computation
- Use event synchronization to prevent deadlocks

---

### 6. Conv2d Optimization Analysis (Step 6)

Skills reference: `step-06-conv2d-optimization.md`

| Feature | Status | Priority | Description |
|---------|--------|----------|-------------|
| Conv operation detailed analysis | ✅ Implemented | - | Efficiency analysis of Conv operations |
| Sharding strategy consistency check | ✅ Implemented | High | Detect sharding changes between consecutive Convs |
| act_block_h_override recommended values | ❌ Not implemented | High | Recommended values based on L1 capacity |
| BN folding status | ❌ Not implemented | Low | Check if BatchNorm is folded into Conv |

**Optimization Rules:**
- Use HEIGHT_SHARDED in most cases
- `act_block_h_override` should be a multiple of 32 and maximum value that fits in L1
- Fold BN into Conv
- Use `bfloat8_b` for weights

---

### 7. Multi-Device Analysis (Step 7)

Skills reference: `step-07-multi-device.md`

| Feature | Status | Priority | Description |
|---------|--------|----------|-------------|
| Device mesh configuration | ❌ Not implemented | Low | Mesh shape verification |
| Tensor distribution pattern analysis | ❌ Not implemented | Low | Usage of Replicate vs Shard |

**Optimization Rules:**
- Data parallelism: shard along batch dimension
- Replicate weights to all devices
- Batch size should be divisible by device count

---

## Recommended Implementation Priority

### Phase 1: High Priority (Directly impacts optimization decisions)

1. **Sharding analysis command** ✅ Implemented
   - Visualize sharding strategy for each operation
   - Display HEIGHT/WIDTH/BLOCK/INTERLEAVED distribution

2. **Host vs Device time analysis**
   - Calculate host overhead ratio from op_to_op_gap
   - Decision criteria for Metal Trace application

3. **Data Format distribution report**
   - Distribution of dtype/Math Fidelity
   - Detect weights not using bfloat8_b and suggest optimization

4. **Reshard detection** ✅ Implemented
   - Detect sharding changes between consecutive operations
   - Visualize reshard costs

### Phase 2: Medium Priority (For detailed optimization)

5. **act_block_h analysis**
   - Block settings and recommended values for Conv operations
   - Suggest optimal values based on L1 capacity

6. **Padding overhead analysis**
   - Memory increase due to tile padding
   - Suggest shape optimizations

7. **Metal Trace applicability check**
   - Static shape check
   - Device vs E2E gap analysis

8. **Optimization checklist output**
   - Automated diagnostics based on skills checklist
   - Display achievement status for each step

### Phase 3: Future Extensions

9. Multi-CQ analysis
10. Multi-Device analysis
11. Automated optimization suggestion system

---

## Proposed New Commands

```bash
# Sharding analysis
uv run ttnn-vis-cli sharding --profiler /path/to/db.sqlite

# Data Format analysis
uv run ttnn-vis-cli dtype-analysis --profiler /path/to/db.sqlite

# Optimization checklist
uv run ttnn-vis-cli optimization-check --profiler /path/to/db.sqlite --performance /path/to/perf

# Host overhead analysis
uv run ttnn-vis-cli host-overhead --performance /path/to/perf

# Reshard detection
uv run ttnn-vis-cli reshard-detect --profiler /path/to/db.sqlite
```

---

## References

### TTNN Optimization Skills
- `step-01-data-formats.md` - Data format optimization
- `step-02-tensor-layouts.md` - Tensor layout optimization
- `step-03-memory-sharding.md` - Memory, sharding, double buffering
- `step-04-metal-trace.md` - Metal Trace guide
- `step-05-multi-cq.md` - Multiple command queues
- `step-06-conv2d-optimization.md` - Conv2d tuning
- `step-07-multi-device.md` - Multi-device scaling

### Tech Reports
- `tech_reports/data_formats/data_formats.md`
- `tech_reports/tensor_layouts/tensor_layouts.md`
- `tech_reports/tensor_sharding/tensor_sharding.md`
- `tech_reports/memory/allocator.md`
- `tech_reports/AdvancedPerformanceOptimizationsForModels/`
