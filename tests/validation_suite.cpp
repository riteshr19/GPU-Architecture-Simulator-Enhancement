#include "warp_scheduler.h"
#include "tensor_core.h"
#include "sector_cache.h"
#include "dram_timing.h"
#include "simulation_engine.h"
#include "gpu_core.h"
#include "memory_hierarchy.h"
#include "performance_monitor.h"

#include <cassert>
#include <cmath>
#include <iostream>
#include <numeric>
#include <vector>
#include <string>
#include <memory>
#include <cstring>

using namespace gpu_sim;

// ---------------------------------------------------------------------------
// Minimal test framework (matches existing test_suite.cpp style)
// ---------------------------------------------------------------------------

static int g_total_tests  = 0;
static int g_passed_tests = 0;

static void check(bool condition, const std::string& msg) {
    ++g_total_tests;
    if (condition) {
        ++g_passed_tests;
        std::cout << "  ✓ " << msg << "\n";
    } else {
        std::cerr << "  ✗ FAIL: " << msg << "\n";
    }
}

// ---------------------------------------------------------------------------
// 1. Warp Scheduler – GTO policy
// ---------------------------------------------------------------------------

void test_gto_scheduler() {
    std::cout << "\n=== Pillar 1A: GTO Warp Scheduler ===\n";

    // --- Basic warp management ---
    WarpScheduler sched(8, SchedulerPolicy::GTO);
    for (uint32_t i = 0; i < 8; ++i) sched.add_warp(i, 0);

    check(sched.num_ready_warps() == 8,
          "All 8 warps ready after add_warp");
    check(sched.num_stalled_warps() == 0,
          "No stalled warps initially");

    // --- Greedy behaviour: same warp issued repeatedly ---
    Warp* first  = sched.select_next_warp(0);
    check(first != nullptr, "First select returns a warp");
    Warp* second = sched.select_next_warp(1);
    check(second != nullptr && second->warp_id == first->warp_id,
          "GTO: second issue from same (greedy) warp");

    // --- Stall switches to oldest ---
    sched.stall_warp(first->warp_id);
    check(sched.num_stalled_warps() == 1, "One stalled warp after stall_warp");
    Warp* after_stall = sched.select_next_warp(2);
    check(after_stall != nullptr && after_stall->warp_id != first->warp_id,
          "GTO: stall causes switch to different warp");

    // --- Ready restores warp ---
    sched.ready_warp(first->warp_id, 10);
    check(sched.num_stalled_warps() == 0, "Zero stalled after ready_warp");

    // --- Complete removes warp ---
    sched.complete_warp(0);
    check(sched.num_ready_warps() == 7, "Completion reduces ready count");

    // --- Issue count tracking ---
    check(sched.total_issues() > 0, "Issues recorded");

    // --- Stats string ---
    auto stats = sched.get_stats_summary();
    check(!stats.empty(), "get_stats_summary not empty");
}

// ---------------------------------------------------------------------------
// 2. Warp Scheduler – Two-Level policy
// ---------------------------------------------------------------------------

void test_two_level_scheduler() {
    std::cout << "\n=== Pillar 1A: Two-Level Warp Scheduler ===\n";

    WarpScheduler sched(16, SchedulerPolicy::TWO_LEVEL, /*fetch_group=*/4);
    for (uint32_t i = 0; i < 16; ++i) sched.add_warp(i, i); // different birth cycles

    check(sched.num_ready_warps() == 16, "All 16 warps ready");

    // Issue several instructions – should stay within the fetch group.
    std::vector<uint32_t> issued_ids;
    for (int i = 0; i < 8; ++i) {
        sched.tick(static_cast<uint64_t>(i));
        Warp* w = sched.select_next_warp(static_cast<uint64_t>(i));
        if (w) issued_ids.push_back(w->warp_id);
    }
    check(!issued_ids.empty(), "Two-Level: issued at least one instruction");

    // Stall all warps in the current fetch group, forcing promotion from pending.
    for (uint32_t i = 0; i < 4; ++i) sched.stall_warp(i);
    sched.tick(20);
    Warp* promoted = sched.select_next_warp(20);
    check(promoted != nullptr, "Two-Level: promotes from pending group when fetch group stalls");
}

// ---------------------------------------------------------------------------
// 3. Tensor Core – MMA correctness
// ---------------------------------------------------------------------------

void test_tensor_core_mma() {
    std::cout << "\n=== Pillar 1B: Tensor Core MMA ===\n";

    TensorCore tc(0);

    // Build A (identity) and B (identity) in FP16.
    std::array<fp16_t, WMMA_M * WMMA_K> a_raw{};
    std::array<fp16_t, WMMA_K * WMMA_N> b_raw{};
    for (uint32_t i = 0; i < WMMA_M; ++i) {
        a_raw[i * WMMA_K + i] = float_to_fp16(1.0f);  // A = I
    }
    for (uint32_t i = 0; i < WMMA_K; ++i) {
        b_raw[i * WMMA_N + i] = float_to_fp16(1.0f);  // B = I (row-major)
    }

    FragmentA fa; FragmentB fb; FragmentC fc, fd;
    tc.load_matrix_a(fa, a_raw.data(), FragmentLayout::ROW_MAJOR);
    tc.load_matrix_b(fb, b_raw.data(), FragmentLayout::ROW_MAJOR);
    tc.fill_zero(fc);

    // D = I × I + 0 = I
    tc.mma_sync(fd, fa, fb, fc, /*cycle=*/0);

    // Verify diagonal is 1.0, off-diagonal is 0.0.
    bool diag_ok = true, offdiag_ok = true;
    for (uint32_t m = 0; m < WMMA_M; ++m) {
        for (uint32_t n = 0; n < WMMA_N; ++n) {
            float v = fd.data[m * WMMA_N + n];
            if (m == n) { if (std::fabs(v - 1.0f) > 1e-3f) diag_ok = false; }
            else        { if (std::fabs(v)         > 1e-3f) offdiag_ok = false; }
        }
    }
    check(diag_ok,    "MMA I×I: diagonal elements are 1.0");
    check(offdiag_ok, "MMA I×I: off-diagonal elements are 0.0");

    // --- Accumulate: D = I×I + I should be 2*I ---
    std::array<float, WMMA_M * WMMA_N> c2_raw{};
    for (uint32_t i = 0; i < WMMA_M; ++i) c2_raw[i * WMMA_N + i] = 1.0f;
    FragmentC fc2, fd2;
    tc.load_accumulator(fc2, c2_raw.data());
    tc.mma_sync(fd2, fa, fb, fc2, /*cycle=*/100);
    bool acc_ok = true;
    for (uint32_t i = 0; i < WMMA_M; ++i) {
        if (std::fabs(fd2.data[i * WMMA_N + i] - 2.0f) > 1e-3f) acc_ok = false;
    }
    check(acc_ok, "MMA I×I + I = 2*I (accumulation correct)");

    // --- FP16 ↔ FP32 round-trip ---
    float orig = 3.14f;
    fp16_t h = float_to_fp16(orig);
    float back = fp16_to_float(h);
    check(std::fabs(back - orig) < 0.01f, "FP16 round-trip within 1% for 3.14");

    // --- Timing model ---
    check(tc.mma_count() == 2,   "Two MMA operations counted");
    check(tc.mma_cycles() == 2 * TensorCore::MMA_LATENCY_CYCLES,
          "MMA cycle count matches 2 × latency");
    check(tc.flops() == 2ULL * 2 * WMMA_M * WMMA_N * WMMA_K,
          "FLOP count = 2 × 2 × M × N × K");
    check(!tc.is_available(tc.busy_until() - 1), "Unit busy before busy_until");
    check(tc.is_available(tc.busy_until()),       "Unit available at busy_until");
}

// ---------------------------------------------------------------------------
// 4. Sector Cache – hit/miss/bypass
// ---------------------------------------------------------------------------

void test_sector_cache() {
    std::cout << "\n=== Pillar 2A: Sector Cache ===\n";

    // 128 KB, 4-way, divergent bypass (threshold 0.5)
    SectorCache sc(128 * 1024, 4, BypassPolicy::DIVERGENT, 0.5f);

    uint8_t write_buf[32];
    for (int i = 0; i < 32; ++i) write_buf[i] = static_cast<uint8_t>(i);

    // Fill a sector.
    sc.fill_sector(0x1000, write_buf, /*cycle=*/1);

    // Read back – should hit.
    uint8_t read_buf[32] = {};
    bool hit = sc.read(0x1000, read_buf, 32, 0xFFFFFFFF, /*cycle=*/2);
    check(hit, "Sector cache hit after fill");
    check(std::memcmp(write_buf, read_buf, 32) == 0, "Read data matches filled sector");

    // Miss on different address.
    bool miss = sc.read(0x2000, read_buf, 32, 0xFFFFFFFF, /*cycle=*/3);
    check(!miss, "Cache miss on unfilled address");

    // Bypass test: only 4/32 threads active (< 0.5 threshold).
    uint32_t low_mask = 0x0000000F; // 4 active threads
    bool bypass_read = sc.read(0x3000, read_buf, 32, low_mask, /*cycle=*/4);
    check(!bypass_read, "Divergent access bypasses cache (low active mask)");

    auto stats = sc.get_stats();
    check(stats.sector_hits   >= 1, "At least one sector hit recorded");
    check(stats.sector_misses >= 1, "At least one sector miss recorded");
    check(stats.bypasses      >= 1, "At least one bypass recorded");

    // Invalidate.
    sc.invalidate(0x1000);
    bool after_inv = sc.read(0x1000, read_buf, 32, 0xFFFFFFFF, /*cycle=*/5);
    check(!after_inv, "Miss after invalidation");

    // Stats summary non-empty.
    auto summary = sc.get_stats_summary();
    check(!summary.empty(), "SectorCache stats summary not empty");
}

// ---------------------------------------------------------------------------
// 5. DRAM Timing Model
// ---------------------------------------------------------------------------

void test_dram_timing() {
    std::cout << "\n=== Pillar 2B: DRAM Timing Model ===\n";

    // Use HBM2 params.
    DRAMTimingModel dram(DRAMTimingParams::hbm2(), /*ranks=*/1, /*banks=*/16);

    // Enqueue two reads to the same row (should produce a row-buffer hit).
    uint64_t row_stride = 1ULL << (6 + 4); // 6 byte-offset bits + 4 bank bits
    uint64_t addr0 = 0;
    uint64_t addr1 = row_stride; // same row, different bank
    (void)addr1;

    uint64_t id0 = dram.enqueue(addr0, DRAMCommandType::READ, /*cycle=*/0);

    // Run enough cycles to service the first request (tRCD + tCL + burst).
    for (uint64_t c = 0; c < 200; ++c) dram.tick(c);
    check(dram.is_complete(id0), "First DRAM read completes within 200 cycles");

    // Row-buffer hit: second request to same bank & row.
    uint64_t id1 = dram.enqueue(addr0, DRAMCommandType::READ, /*cycle=*/200);
    for (uint64_t c = 200; c < 400; ++c) dram.tick(c);
    check(dram.is_complete(id1), "Second DRAM read (same row) completes");

    auto stats = dram.get_stats();
    check(stats.total_requests >= 2, "At least 2 requests tracked");
    check(stats.avg_latency_cycles > 0.0, "Average latency > 0");

    // Row-buffer conflict: same bank (0) but row 1 instead of row 0.
    // With the default address map: bank = (addr>>6) % 16, row = (addr>>10) % 65536
    // bank 0, row 1: addr = 1<<10 = 1024
    uint64_t conflict_addr = 1024ULL;
    uint64_t idc = dram.enqueue(conflict_addr, DRAMCommandType::READ, /*cycle=*/400);
    for (uint64_t c = 400; c < 700; ++c) dram.tick(c);
    check(dram.is_complete(idc), "Row-buffer conflict request eventually completes");
    check(dram.get_stats().row_buffer_conflicts >= 1,
          "At least one row-buffer conflict recorded");

    auto summary = dram.get_stats_summary();
    check(!summary.empty(), "DRAM stats summary not empty");

    // GDDR6 preset sanity.
    DRAMTimingParams gddr6 = DRAMTimingParams::gddr6();
    check(gddr6.tCL > 0 && gddr6.tRCD > 0, "GDDR6 preset has valid timing parameters");
}

// ---------------------------------------------------------------------------
// 6. Simulation Engine – single-threaded with sampling & checkpointing
// ---------------------------------------------------------------------------

void test_simulation_engine() {
    std::cout << "\n=== Pillar 3: Simulation Engine ===\n";

    // Build a minimal system.
    auto gpu   = std::make_shared<GPUCore>(4);
    auto mem   = std::make_shared<MemoryHierarchy>();
    auto perf  = std::make_shared<PerformanceMonitor>();
    gpu->initialize(mem, perf);

    auto sched = std::make_shared<WarpScheduler>(8, SchedulerPolicy::GTO);
    for (uint32_t i = 0; i < 8; ++i) sched->add_warp(i, 0);

    auto dram  = std::make_shared<DRAMTimingModel>(DRAMTimingParams::hbm2());

    SimulationConfig cfg;
    cfg.max_cycles           = 5000;
    cfg.max_instructions     = 0;
    cfg.enable_sampling      = true;
    cfg.warmup_cycles        = 100;
    cfg.sample_length        = 500;
    cfg.skip_cycles          = 1000;
    cfg.max_samples          = 3;
    cfg.enable_checkpointing = false;  // no disk I/O in tests

    SimulationEngine engine(cfg);
    engine.attach_gpu_core(gpu);
    engine.attach_memory(mem);
    engine.attach_scheduler(sched);
    engine.attach_dram(dram);

    uint64_t cb_calls = 0;
    engine.run([&](uint64_t /*cycle*/, uint64_t /*instr*/) -> bool {
        ++cb_calls;
        return true;
    });

    auto stats = engine.get_stats();
    check(stats.total_cycles > 0,       "Engine ran at least one cycle");
    check(stats.total_instructions > 0, "Engine issued at least one instruction");
    check(cb_calls > 0,                 "Instruction callback was invoked");

    const auto& ipcs = engine.get_sample_ipcs();
    check(!ipcs.empty(), "Sampling: at least one IPC sample collected");
    for (double ipc : ipcs) {
        check(ipc >= 0.0, "Sampled IPC is non-negative");
    }

    // Checkpoint round-trip (in-memory only).
    auto ckpt = engine.create_checkpoint("test");
    check(ckpt.cycle == stats.total_cycles,
          "Checkpoint cycle matches engine cycle");
    check(ckpt.label == "test", "Checkpoint label preserved");

    engine.restore_checkpoint(ckpt);
    auto stats2 = engine.get_stats();
    check(stats2.total_cycles == ckpt.cycle, "Restore sets cycle correctly");
}

// ---------------------------------------------------------------------------
// 7. Integration: Warp Scheduler drives GPUCore
// ---------------------------------------------------------------------------

void test_scheduler_gpu_integration() {
    std::cout << "\n=== Integration: Scheduler ↔ GPUCore ===\n";

    auto mem  = std::make_shared<MemoryHierarchy>();
    auto perf = std::make_shared<PerformanceMonitor>();
    auto gpu  = std::make_shared<GPUCore>(4);
    gpu->initialize(mem, perf);

    WarpScheduler sched(4, SchedulerPolicy::GTO);
    for (uint32_t i = 0; i < 4; ++i) sched.add_warp(i, i * 10);

    // Simulate 100 cycles: select a warp each cycle, occasionally stall/ready.
    uint64_t issues = 0;
    for (uint64_t cycle = 0; cycle < 100; ++cycle) {
        sched.tick(cycle);
        Warp* w = sched.select_next_warp(cycle);
        if (w) {
            ++issues;
            // Simulate occasional memory stall.
            if (cycle % 7 == 0) {
                sched.stall_warp(w->warp_id);
                sched.ready_warp(w->warp_id, cycle + 5);
            }
        }
    }

    check(issues > 0, "Integration: warps issued instructions over 100 cycles");
    check(sched.total_stalls() > 0, "Integration: stalls were recorded");
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

int main() {
    std::cout << "GPU Architecture Simulator Enhancement – Validation Suite\n";
    std::cout << "==========================================================\n";

    test_gto_scheduler();
    test_two_level_scheduler();
    test_tensor_core_mma();
    test_sector_cache();
    test_dram_timing();
    test_simulation_engine();
    test_scheduler_gpu_integration();

    std::cout << "\n==========================================================\n";
    std::cout << "Results: " << g_passed_tests << " / " << g_total_tests
              << " tests passed.\n";

    if (g_passed_tests == g_total_tests) {
        std::cout << "✓ ALL VALIDATION TESTS PASSED\n";
        return 0;
    } else {
        std::cerr << "✗ " << (g_total_tests - g_passed_tests)
                  << " TESTS FAILED\n";
        return 1;
    }
}
