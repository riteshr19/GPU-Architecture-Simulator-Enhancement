# RFC-001: GPU Architecture Simulator Enhancement Roadmap

| Field       | Value                                         |
|-------------|-----------------------------------------------|
| **Title**   | Enhancement Roadmap for GPU Architecture Simulator |
| **Status**  | Implemented                                   |
| **Authors** | GPU Architecture Team                         |
| **Created** | 2026-02-27                                    |
| **Updated** | 2026-02-27                                    |

---

## Abstract

This RFC describes a comprehensive set of enhancements to the GPU Architecture
Simulator, organized into four pillars:

1. **Core Execution Model** — GTO warp scheduling and Tensor Core MMA
2. **Memory Hierarchy & Coherence** — Sector cache with bypassing and DRAM timing
3. **Simulation Performance** — Checkpointing and SimPoint-style sampling
4. **Infrastructure & Validation** — Python regression suite and structured testing

Each section provides motivation, technical design, implementation references
(headers and source files), and guidance for future work.

---

## Table of Contents

- [1. Core Execution Model](#1-core-execution-model)
  - [1.1 Greedy-Then-Oldest (GTO) Warp Scheduler](#11-greedy-then-oldest-gto-warp-scheduler)
  - [1.2 Tensor Core Abstraction for MMA Operations](#12-tensor-core-abstraction-for-mma-operations)
- [2. Memory Hierarchy & Coherence](#2-memory-hierarchy--coherence)
  - [2.1 Sector Cache with Bypassing](#21-sector-cache-with-bypassing)
  - [2.2 DRAM Timing Model](#22-dram-timing-model)
- [3. Simulation Performance](#3-simulation-performance)
  - [3.1 Checkpointing and SimPoint Sampling](#31-checkpointing-and-simpoint-sampling)
  - [3.2 Multi-Threading Strategies (Future Work)](#32-multi-threading-strategies-future-work)
- [4. Infrastructure & Validation](#4-infrastructure--validation)
  - [4.1 Validation Suite Design](#41-validation-suite-design)
  - [4.2 Python Regression Script](#42-python-regression-script)
- [5. File Inventory](#5-file-inventory)
- [6. References](#6-references)

---

## 1. Core Execution Model

### 1.1 Greedy-Then-Oldest (GTO) Warp Scheduler

**Files:** `include/warp_scheduler.h`, `src/warp_scheduler.cpp`

#### Motivation

The baseline simulator dispatches threads uniformly across shader cores without
a fine-grained warp scheduling policy. Real GPU SM (Streaming Multiprocessor)
units use a warp scheduler to select which warp issues an instruction each
cycle. The scheduling policy has a significant impact on:

- **Cache locality**: Greedily executing one warp exhausts its temporal locality
  before switching, reducing L1 thrashing.
- **Memory-level parallelism (MLP)**: Stalling on a long-latency memory
  operation frees the pipeline for another warp.
- **Fairness**: Oldest-first fallback ensures no warp is starved.

#### Design

The GTO scheduler works as follows:

1. **Greedy phase**: Continue issuing instructions from the *current active warp*
   as long as it remains in the `READY` state.
2. **Fallback (Oldest)**: When the active warp stalls (e.g., on a cache miss,
   moving to `WAITING`), select the ready warp with the earliest arrival time
   (`age` field).

A Two-Level scheduler variant is also provided, where warps are partitioned
into a *fetch group* (high priority) and a *ready group* (low priority) to
limit the number of concurrently competing warps.

#### Key C++ Snippet — GTO Scheduling

```cpp
int32_t WarpScheduler::schedule_gto(uint64_t current_cycle) {
    // GREEDY: continue issuing from the current active warp if it is READY
    if (current_active_warp_ >= 0) {
        for (auto& w : warps_) {
            if (static_cast<int32_t>(w.warp_id) == current_active_warp_ &&
                w.state == WarpState::READY &&
                w.instructions_remaining > 0) {
                w.last_issued_cycle = current_cycle;
                w.instructions_remaining--;
                stats_.instructions_issued++;
                if (w.instructions_remaining == 0) {
                    w.state = WarpState::COMPLETED;
                    stats_.warps_completed++;
                    current_active_warp_ = -1;
                }
                return current_active_warp_;
            }
        }
        // Current warp is no longer ready — fall through to OLDEST
        stats_.policy_switches++;
    }

    // OLDEST: pick the ready warp with the smallest age (earliest arrival)
    int32_t oldest = find_oldest_ready_warp();
    if (oldest >= 0) {
        current_active_warp_ = oldest;
        // ... issue instruction from oldest warp ...
    }
    return oldest;
}
```

#### Warp States

| State     | Meaning                                      |
|-----------|----------------------------------------------|
| READY     | Warp can issue an instruction this cycle      |
| WAITING   | Blocked on long-latency operation (memory)    |
| BLOCKED   | Blocked on barrier or data dependency         |
| COMPLETED | All instructions retired                      |

#### Statistics Tracked

- `total_cycles`, `idle_cycles` (no ready warp)
- `instructions_issued`, `warps_completed`
- `policy_switches` (GTO → oldest fallback count)
- `throughput_ipc` (computed from instructions / cycles)

---

### 1.2 Tensor Core Abstraction for MMA Operations

**Files:** `include/tensor_core.h`, `src/tensor_core.cpp`

#### Motivation

Modern NVIDIA GPUs (Volta and later) include Tensor Cores that accelerate
matrix operations used in deep learning and HPC. Supporting a Tensor Core
abstraction enables the simulator to:

- Model mixed-precision throughput (FP16 input, FP32 accumulation).
- Estimate the performance benefit of Tensor Cores vs. CUDA cores.
- Explore different tile sizes and precision modes (BF16, TF32, INT8, INT4).

#### Design

The `TensorCore` class implements:

```
D[M×N] = A[M×K] × B[K×N] + C[M×N]
```

Key features:
- **MatrixFragment**: In-register tile descriptor (rows, cols, precision, data).
- **Precision simulation**: Bit-accurate truncation for BF16 and TF32 via
  integer bit manipulation of IEEE-754 representation.
- **Latency model**: Per-instruction cycle count varies by precision:

| Precision   | Latency (cycles) |
|-------------|-------------------|
| FP16→FP16   | 8                 |
| FP16→FP32   | 8                 |
| BF16→FP32   | 8                 |
| TF32→FP32   | 16                |
| INT8→INT32  | 4                 |
| INT4→INT32  | 4                 |

#### Key C++ Snippet — Mixed-Precision Multiply

```cpp
float TensorCore::precision_mul(float a, float b, PrecisionMode mode) const {
    switch (mode) {
        case PrecisionMode::BF16_FP32: {
            // BF16: same exponent range as FP32 but only 8 mantissa bits
            auto to_bf16 = [](float v) -> float {
                uint32_t bits;
                std::memcpy(&bits, &v, sizeof(bits));
                bits &= 0xFFFF0000u;  // Zero low 16 bits of mantissa
                float result;
                std::memcpy(&result, &bits, sizeof(result));
                return result;
            };
            return to_bf16(a) * to_bf16(b);
        }
        // ... other precision modes ...
    }
}
```

---

## 2. Memory Hierarchy & Coherence

### 2.1 Sector Cache with Bypassing

**Files:** `include/sector_cache.h`, `src/sector_cache.cpp`

#### Motivation

The baseline L1/L2 cache model uses full-line fetch with simple LRU eviction.
This causes two problems for GPU workloads:

1. **Over-fetch**: A 128-byte cache line is fetched even when only 32 bytes are
   needed, wasting bandwidth in memory-intensive kernels.
2. **Thrashing**: Streaming accesses pollute the cache, evicting data with
   better reuse.

#### Design: Sector Cache

A 128-byte cache line is divided into 4 × 32-byte **sectors**. Each sector
has independent valid and dirty bits:

```
┌──────────────────────────────────────┐
│           128-byte Cache Line         │
│ ┌────────┬────────┬────────┬────────┐│
│ │Sector 0│Sector 1│Sector 2│Sector 3││
│ │V=1 D=0 │V=0 D=0 │V=1 D=1 │V=0 D=0││
│ └────────┴────────┴────────┴────────┘│
│ Tag: 0xABCD   LRU timestamp: 42      │
└──────────────────────────────────────┘
```

On a miss, **only the requested sector** is fetched from the next level,
reducing bandwidth by up to 75%.

#### Design: Bypassing

Four bypass hints are supported:

| Hint              | Behavior                                    |
|-------------------|---------------------------------------------|
| `NO_BYPASS`       | Normal caching path                         |
| `BYPASS_L1`       | Skip L1, go directly to L2                  |
| `BYPASS_ALL_CACHE`| Skip all caches, access DRAM directly       |
| `STREAMING`       | Write-evict: don't allocate on miss         |

**Adaptive bypass**: When enabled, the cache monitors recent access addresses.
If a sequential (streaming) pattern is detected (≥3 consecutive forward
accesses within 2× line size), the `STREAMING` hint is automatically applied.

#### Key C++ Snippet — Sector Read

```cpp
bool SectorCache::read(uint64_t address, void* data, size_t size,
                       BypassHint hint) {
    access_counter_++;
    record_access(address);

    // Apply adaptive bypass if enabled
    if (adaptive_bypass_enabled_ && hint == BypassHint::NO_BYPASS) {
        hint = get_adaptive_hint(address);
    }

    if (hint == BypassHint::BYPASS_L1 || hint == BypassHint::BYPASS_ALL_CACHE) {
        stats_.bypass_count++;
        stats_.misses++;
        return false; // Caller should go to next level
    }

    SectorCacheLine* line = find_line(address);
    size_t sector_idx = get_sector_index(address);

    if (line && sector_idx < line->sectors.size() &&
        line->sectors[sector_idx].valid) {
        // Sector hit
        stats_.hits++;
        stats_.sector_hits++;
        // ... copy data from sector ...
        return true;
    }

    // Sector miss or full miss
    stats_.misses++;
    stats_.sector_misses++;
    return false;
}
```

#### Bandwidth Savings Estimation

The `SectorCacheStats` reports `bandwidth_saved_pct`:

```
saved = 1 − (sector_misses × sector_size) / (sector_misses × line_size)
      = 1 − sector_size / line_size
      = 1 − 32/128 = 75%  (maximum theoretical saving)
```

---

### 2.2 DRAM Timing Model

**Files:** `include/dram_controller.h`, `src/dram_controller.cpp`

#### Motivation

The baseline memory hierarchy uses a flat latency of 100 cycles for VRAM
access. This does not capture:

- **Row buffer locality**: Accessing the same DRAM row is much cheaper than
  opening a new one.
- **Bank-level parallelism**: Requests to different banks can proceed
  concurrently.
- **Timing constraints**: Real DRAM imposes minimum delays between commands
  (tRCD, tRP, tCL, etc.).

#### Design

The `DRAMController` models:

| Feature                | Implementation                              |
|------------------------|---------------------------------------------|
| **Address mapping**    | Row-interleaved across channels and banks    |
| **Bank state**         | Tracks open row, last activate/access times  |
| **Row buffer hit**     | Latency = tCL only                          |
| **Row buffer miss**    | Latency = tRCD + tCL (empty row buffer)     |
| **Row buffer conflict**| Latency = tRP + tRCD + tCL (close + open)   |
| **FR-FCFS scheduling** | Row-buffer-hit requests prioritized, then FCFS|
| **Timing profiles**    | HBM2 and GDDR6 pre-configured               |

#### HBM2 Default Timing Parameters

| Parameter | Value | Description                    |
|-----------|-------|--------------------------------|
| tCL       | 14    | CAS Latency                    |
| tRCD      | 14    | RAS-to-CAS Delay               |
| tRP       | 14    | Row Precharge                  |
| tRAS      | 34    | Row Active time                |
| tRC       | 48    | Row Cycle time                 |
| tCCD      | 2     | Column-to-Column Delay         |
| Channels  | 8     | Number of memory channels      |
| Banks/Ch  | 16    | Banks per channel              |
| Bus width | 128b  | Per-channel bus width           |

#### Key C++ Snippet — Row Buffer Hit/Miss/Conflict

```cpp
uint64_t DRAMController::compute_access_latency(
    const DRAMRequest& req, uint64_t current_cycle) const {
    const auto& bank = bank_states_[req.channel_id][req.bank_id];
    uint64_t latency = 0;

    if (!bank.row_open) {
        // Row buffer empty — need ACT + CAS
        latency = params_.tRCD + params_.tCL;
    } else if (bank.open_row == req.row_id) {
        // Row buffer hit — CAS only
        latency = params_.tCL;
    } else {
        // Row buffer conflict — PRE + ACT + CAS
        latency = params_.tRP + params_.tRCD + params_.tCL;
    }
    return latency;
}
```

---

## 3. Simulation Performance

### 3.1 Checkpointing and SimPoint Sampling

**Files:** `include/sim_checkpoint.h`, `src/sim_checkpoint.cpp`

#### Motivation

Simulating large-scale CUDA/OpenCL kernels instruction-by-instruction is
extremely slow (wall-clock hours for a few billion instructions). Two
techniques address this:

1. **Checkpointing**: Save the full simulation state so that re-runs can skip
   initialization phases.
2. **Sampling (SimPoint-style)**: Simulate only a representative subset of
   execution intervals and extrapolate full-run metrics via weighted
   combination.

#### Design: Checkpointing

```
SimulationCheckpoint:
  - checkpoint_id, cycle, instruction_count
  - Metrics snapshot: IPC, cache_hit_rate, active_warps
  - Serialization: CSV format for portability
```

Checkpoints are saved to and loaded from files, enabling:
- Fast restart of interrupted simulations
- Comparison of metrics at the same execution point across configurations

#### Design: SimPoint Sampling

1. **Divide** execution into fixed-size intervals (default: 10M instructions).
2. **Characterize** each interval by a Basic Block Vector (BBV).
3. **Cluster** BBVs using k-means (simplified 5-iteration implementation).
4. **Select** the interval closest to each cluster centroid as the
   representative.
5. **Simulate** only the representative intervals.
6. **Estimate** full-run metrics as a weighted sum:

```
IPC_estimated = Σ (weight_i × IPC_sample_i)
```

where `weight_i` = (cluster size) / (total intervals).

#### Estimated Performance

| Metric              | Full Sim | Sampled (10 clusters) |
|---------------------|----------|-----------------------|
| Intervals simulated | 100      | 10                    |
| Estimated speedup   | 1×       | ~10×                  |
| IPC error           | 0%       | <5% (typical)         |

### 3.2 Multi-Threading Strategies (Future Work)

For further wall-clock reduction, the following multi-threading strategies
are recommended:

1. **Functional/Timing decoupling**: Run the functional simulation (instruction
   execution) in one thread and timing simulation (pipeline state) in another,
   communicating via a shared trace buffer.

2. **SM-level parallelism**: Each SM (Streaming Multiprocessor) can be simulated
   on a separate host thread, with a barrier synchronization at memory fence
   instructions.

3. **Speculative parallel simulation**: Optimistically simulate multiple epochs
   in parallel, rolling back on misspeculation (e.g., global memory
   synchronization).

These are non-trivial to implement correctly and are flagged for a future phase.

---

## 4. Infrastructure & Validation

### 4.1 Validation Suite Design

**Files:** `tests/test_suite.cpp` (original), `tests/test_enhancements.cpp` (new)

#### Test Organization

| Test Group                | Assertions | Component                |
|---------------------------|------------|--------------------------|
| GTO Warp Scheduler        | 7          | `WarpScheduler` (GTO)   |
| Two-Level Scheduler       | 2          | `WarpScheduler` (TL)    |
| Scheduler Stalls          | 3          | Stall/resume behavior    |
| Tensor Core MMA           | 8          | `TensorCore` FP16→FP32  |
| Tensor Core Accumulation  | 4          | D = A×B + C              |
| Tensor Core INT8          | 2          | `TensorCore` INT8→INT32 |
| Sector Cache Basic        | 4          | Write/Read/Match         |
| Sector Cache Bypass       | 2          | Bypass hint path         |
| Sector Cache Partial      | 2          | Sector-level validity    |
| Sector Cache Adaptive     | 2          | Streaming detection      |
| DRAM HBM2                 | 6          | HBM2 timing model       |
| DRAM GDDR6                | 4          | GDDR6 timing model      |
| DRAM Row Buffer Conflict  | 2          | Conflict detection       |
| Checkpoint Save/Load      | 8          | Serialization round-trip |
| SimPoint Sampling         | 10         | K-means + estimation     |
| Integration               | 8          | All components together  |
| **Total**                 | **74**     |                          |

#### Benchmark Suite Recommendation

For validation against real hardware, run the following standard suites
through the simulator:

| Suite      | Kernels                        | Focus Area           |
|------------|--------------------------------|----------------------|
| **Rodinia**| BFS, Hotspot, SRAD, NW         | Diverse GPU patterns |
| **Parboil**| SGEMM, Stencil, MRI-Q          | Compute-intensive    |
| **PolyBench/GPU** | 2mm, 3mm, GEMM, ATAX    | Linear algebra       |

### 4.2 Python Regression Script

**File:** `scripts/validation_regression.py`

#### Usage

```bash
# Build the project
cd build && cmake .. && make -j4 && cd ..

# Run validation (first run creates baseline)
python3 scripts/validation_regression.py --build-dir build

# Subsequent runs compare against baseline
python3 scripts/validation_regression.py --build-dir build \
    --baseline scripts/baseline.json \
    --output validation_report.md
```

#### Output Format

The script produces a Markdown report with:

1. **Summary table**: Total benchmarks, pass/fail count, pass rate.
2. **Detailed results**: Per-benchmark IPC, cache hit rate, wall time, with
   delta comparison against baseline.
3. **Regression analysis**: Flags any benchmark with >5% IPC regression.
4. **Test assertion summary**: Per-binary assertion pass/fail counts.

---

## 5. File Inventory

| File | Type | Pillar |
|------|------|--------|
| `include/warp_scheduler.h` | Header | 1 — Execution |
| `src/warp_scheduler.cpp` | Source | 1 — Execution |
| `include/tensor_core.h` | Header | 1 — Execution |
| `src/tensor_core.cpp` | Source | 1 — Execution |
| `include/sector_cache.h` | Header | 2 — Memory |
| `src/sector_cache.cpp` | Source | 2 — Memory |
| `include/dram_controller.h` | Header | 2 — Memory |
| `src/dram_controller.cpp` | Source | 2 — Memory |
| `include/sim_checkpoint.h` | Header | 3 — Simulation |
| `src/sim_checkpoint.cpp` | Source | 3 — Simulation |
| `tests/test_enhancements.cpp` | Test | 4 — Validation |
| `scripts/validation_regression.py` | Script | 4 — Validation |
| `docs/rfc_enhancement_roadmap.md` | Doc | All |

---

## 6. References

1. Rogers, T. G., O'Connor, M., & Aamodt, T. M. (2012). *Cache-Conscious
   Wavefront Scheduling*. MICRO-45.
2. Gebhart, M., et al. (2012). *Unifying Primary Cache, Scratch, and Register
   File Memories in a Throughput Processor*. MICRO-45.
3. Narasiman, V., et al. (2011). *Improving GPU Performance via Large Warps
   and Two-Level Warp Scheduling*. MICRO-44.
4. Sherwood, T., Perelman, E., Hamerly, G., & Calder, B. (2002).
   *Automatically Characterizing Large Scale Program Behavior*. ASPLOS.
5. NVIDIA. (2017). *NVIDIA Tesla V100 GPU Architecture* (Volta whitepaper).
6. Bakhoda, A., et al. (2009). *Analyzing CUDA Workloads Using a Detailed GPU
   Simulator*. ISPASS.

---

*End of RFC-001*
