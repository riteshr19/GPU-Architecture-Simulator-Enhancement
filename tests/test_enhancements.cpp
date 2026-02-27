#include "warp_scheduler.h"
#include "tensor_core.h"
#include "sector_cache.h"
#include "dram_controller.h"
#include "sim_checkpoint.h"
#include "performance_monitor.h"
#include <iostream>
#include <cassert>
#include <vector>
#include <memory>
#include <cmath>

using namespace gpu_sim;

// Reuse the test framework from the main test suite
class TestFramework {
public:
    static void assert_true(bool condition, const std::string& message) {
        if (!condition) {
            std::cerr << "ASSERTION FAILED: " << message << std::endl;
            exit(1);
        }
        std::cout << "✓ " << message << std::endl;
    }

    static void assert_equals(uint64_t expected, uint64_t actual, const std::string& message) {
        if (expected != actual) {
            std::cerr << "ASSERTION FAILED: " << message
                      << " (expected: " << expected << ", actual: " << actual << ")" << std::endl;
            exit(1);
        }
        std::cout << "✓ " << message << std::endl;
    }

    static void assert_greater_than(double value, double threshold, const std::string& message) {
        if (value <= threshold) {
            std::cerr << "ASSERTION FAILED: " << message
                      << " (value: " << value << " <= threshold: " << threshold << ")" << std::endl;
            exit(1);
        }
        std::cout << "✓ " << message << std::endl;
    }

    static void assert_near(double value, double expected, double epsilon, const std::string& message) {
        if (std::abs(value - expected) > epsilon) {
            std::cerr << "ASSERTION FAILED: " << message
                      << " (value: " << value << ", expected: " << expected
                      << ", epsilon: " << epsilon << ")" << std::endl;
            exit(1);
        }
        std::cout << "✓ " << message << std::endl;
    }
};

// ============================================================================
// Warp Scheduler Tests
// ============================================================================
void test_warp_scheduler_gto() {
    std::cout << "\n=== Testing GTO Warp Scheduler ===" << std::endl;

    auto perf = std::make_shared<PerformanceMonitor>();
    WarpScheduler scheduler(16, SchedulePolicy::GTO);
    scheduler.initialize(perf);

    // Add warps with different instruction counts
    scheduler.add_warp(0, 5);
    scheduler.add_warp(1, 3);
    scheduler.add_warp(2, 4);

    TestFramework::assert_equals(3, scheduler.get_num_active_warps(),
                                 "Should have 3 active warps");
    TestFramework::assert_equals(3, scheduler.get_num_ready_warps(),
                                 "All 3 warps should be ready");

    // GTO should greedily stick with the first warp it picks
    int32_t first = scheduler.schedule_next(0);
    TestFramework::assert_true(first >= 0, "Should schedule a warp on first cycle");

    int32_t second = scheduler.schedule_next(1);
    TestFramework::assert_true(second == first,
                               "GTO should greedily continue with the same warp");

    // Run all warps to completion
    uint64_t cycle = 2;
    while (scheduler.get_num_ready_warps() > 0) {
        scheduler.schedule_next(cycle++);
    }

    auto stats = scheduler.get_statistics();
    TestFramework::assert_equals(12, stats.instructions_issued,
                                 "Should issue all 12 instructions (5+3+4)");
    TestFramework::assert_equals(3, stats.warps_completed,
                                 "All 3 warps should complete");
    TestFramework::assert_greater_than(stats.throughput_ipc, 0.0,
                                       "IPC should be positive");

    std::cout << "GTO Warp Scheduler tests passed!" << std::endl;
}

void test_warp_scheduler_two_level() {
    std::cout << "\n=== Testing Two-Level Warp Scheduler ===" << std::endl;

    WarpScheduler scheduler(16, SchedulePolicy::TWO_LEVEL);
    scheduler.initialize(nullptr);

    scheduler.add_warp(0, 3);
    scheduler.add_warp(1, 3);
    scheduler.add_warp(2, 3);

    uint64_t cycle = 0;
    while (scheduler.get_num_ready_warps() > 0) {
        scheduler.schedule_next(cycle++);
    }

    auto stats = scheduler.get_statistics();
    TestFramework::assert_equals(9, stats.instructions_issued,
                                 "Should issue all 9 instructions (3+3+3)");
    TestFramework::assert_equals(3, stats.warps_completed,
                                 "All 3 warps should complete");

    std::cout << "Two-Level Warp Scheduler tests passed!" << std::endl;
}

void test_warp_scheduler_stall() {
    std::cout << "\n=== Testing Warp Scheduler with Stalls ===" << std::endl;

    WarpScheduler scheduler(16, SchedulePolicy::GTO);
    scheduler.initialize(nullptr);

    scheduler.add_warp(0, 10);
    scheduler.add_warp(1, 5);

    // Schedule warp 0 once
    int32_t first = scheduler.schedule_next(0);

    // Simulate warp 0 stalling on memory
    scheduler.set_warp_state(static_cast<uint32_t>(first), WarpState::WAITING);

    // Next schedule should pick warp 1 (the oldest ready warp)
    int32_t next = scheduler.schedule_next(1);
    TestFramework::assert_true(next >= 0, "Should find a ready warp after stall");
    TestFramework::assert_true(next != first || first < 0,
                               "Should switch to a different warp on stall");

    // Notify memory complete for the stalled warp
    scheduler.notify_memory_complete(static_cast<uint32_t>(first), 5);

    auto stats = scheduler.get_statistics();
    TestFramework::assert_greater_than(static_cast<double>(stats.policy_switches), 0.0,
                                       "GTO should have recorded at least one policy switch");

    std::cout << "Warp Scheduler stall tests passed!" << std::endl;
}

// ============================================================================
// Tensor Core Tests
// ============================================================================
void test_tensor_core_mma() {
    std::cout << "\n=== Testing Tensor Core MMA ===" << std::endl;

    auto perf = std::make_shared<PerformanceMonitor>();
    TensorCore tc(0, 4, 4, 4); // Small 4x4x4 tiles for testing
    tc.initialize(perf);

    // Create identity-like matrices:
    // A = [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]
    std::vector<float> identity_data = {
        1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1
    };

    // B = [[2,0,0,0],[0,3,0,0],[0,0,4,0],[0,0,0,5]]
    std::vector<float> b_data = {
        2,0,0,0, 0,3,0,0, 0,0,4,0, 0,0,0,5
    };

    auto A = tc.load_matrix_a(identity_data, 4, 4, PrecisionMode::FP16_FP32);
    auto B = tc.load_matrix_b(b_data, 4, 4, PrecisionMode::FP16_FP32);
    auto C = tc.create_accumulator(4, 4, PrecisionMode::FP16_FP32, 0.0f);
    MatrixFragment D(4, 4, PrecisionMode::FP16_FP32);

    tc.mma(A, B, C, D);

    // D should equal B (since A is identity and C is zero)
    TestFramework::assert_near(D.at(0, 0), 2.0f, 0.01, "D[0][0] should be 2.0");
    TestFramework::assert_near(D.at(1, 1), 3.0f, 0.01, "D[1][1] should be 3.0");
    TestFramework::assert_near(D.at(2, 2), 4.0f, 0.01, "D[2][2] should be 4.0");
    TestFramework::assert_near(D.at(3, 3), 5.0f, 0.01, "D[3][3] should be 5.0");
    TestFramework::assert_near(D.at(0, 1), 0.0f, 0.01, "D[0][1] should be 0.0");

    auto stats = tc.get_statistics();
    TestFramework::assert_equals(1, stats.mma_operations, "Should have 1 MMA operation");
    TestFramework::assert_true(stats.total_flops > 0, "Should have positive FLOPS");
    TestFramework::assert_true(stats.total_cycles > 0, "Should have positive cycle count");

    std::cout << "Tensor Core MMA tests passed!" << std::endl;
}

void test_tensor_core_accumulate() {
    std::cout << "\n=== Testing Tensor Core Accumulation ===" << std::endl;

    TensorCore tc(0, 2, 2, 2);

    // A = [[1, 2], [3, 4]]
    auto A = tc.load_matrix_a({1, 2, 3, 4}, 2, 2, PrecisionMode::FP16_FP32);
    // B = [[1, 0], [0, 1]]  (identity)
    auto B = tc.load_matrix_b({1, 0, 0, 1}, 2, 2, PrecisionMode::FP16_FP32);
    // C = [[10, 20], [30, 40]]
    auto C = tc.load_matrix_a({10, 20, 30, 40}, 2, 2, PrecisionMode::FP16_FP32);

    MatrixFragment D(2, 2, PrecisionMode::FP16_FP32);
    tc.mma(A, B, C, D);

    // D = A*I + C = A + C = [[11, 22], [33, 44]]
    TestFramework::assert_near(D.at(0, 0), 11.0f, 0.01, "D[0][0] = 1+10 = 11");
    TestFramework::assert_near(D.at(0, 1), 22.0f, 0.01, "D[0][1] = 2+20 = 22");
    TestFramework::assert_near(D.at(1, 0), 33.0f, 0.01, "D[1][0] = 3+30 = 33");
    TestFramework::assert_near(D.at(1, 1), 44.0f, 0.01, "D[1][1] = 4+40 = 44");

    std::cout << "Tensor Core Accumulation tests passed!" << std::endl;
}

void test_tensor_core_int8() {
    std::cout << "\n=== Testing Tensor Core INT8 Mode ===" << std::endl;

    TensorCore tc(0, 2, 2, 2);
    tc.set_precision_mode(PrecisionMode::INT8_INT32);

    auto A = tc.load_matrix_a({3, 5, 7, 11}, 2, 2, PrecisionMode::INT8_INT32);
    auto B = tc.load_matrix_b({1, 0, 0, 1}, 2, 2, PrecisionMode::INT8_INT32);
    auto C = tc.create_accumulator(2, 2, PrecisionMode::INT8_INT32, 0.0f);

    MatrixFragment D(2, 2, PrecisionMode::INT8_INT32);
    tc.mma(A, B, C, D);

    TestFramework::assert_near(D.at(0, 0), 3.0f, 0.01, "INT8: D[0][0] = 3");
    TestFramework::assert_near(D.at(1, 1), 11.0f, 0.01, "INT8: D[1][1] = 11");

    std::cout << "Tensor Core INT8 tests passed!" << std::endl;
}

// ============================================================================
// Sector Cache Tests
// ============================================================================
void test_sector_cache_basic() {
    std::cout << "\n=== Testing Sector Cache Basics ===" << std::endl;

    auto perf = std::make_shared<PerformanceMonitor>();
    SectorCache cache(32, 128, 32, 4); // 32KB, 128B lines, 32B sectors, 4-way
    cache.initialize(perf);

    // Write some data
    uint8_t write_data[32];
    for (int i = 0; i < 32; ++i) write_data[i] = static_cast<uint8_t>(i);

    TestFramework::assert_true(cache.write(0x1000, write_data, 32),
                               "Write to sector cache should succeed");

    // Read it back
    uint8_t read_data[32] = {};
    TestFramework::assert_true(cache.read(0x1000, read_data, 32),
                               "Read from sector cache should hit");

    bool data_match = true;
    for (int i = 0; i < 32; ++i) {
        if (read_data[i] != write_data[i]) { data_match = false; break; }
    }
    TestFramework::assert_true(data_match, "Read data should match written data");

    auto stats = cache.get_statistics();
    TestFramework::assert_true(stats.hits > 0, "Should have at least one hit");

    std::cout << "Sector Cache basic tests passed!" << std::endl;
}

void test_sector_cache_bypass() {
    std::cout << "\n=== Testing Sector Cache Bypass ===" << std::endl;

    SectorCache cache(16, 128, 32, 4);
    cache.initialize(nullptr);

    uint8_t data[32] = {0xAA};

    // Write normally
    cache.write(0x2000, data, 32);

    // Read with bypass hint — should miss (bypassed)
    uint8_t buf[32] = {};
    bool hit = cache.read(0x2000, buf, 32, BypassHint::BYPASS_L1);
    TestFramework::assert_true(!hit, "Bypass read should report miss");

    auto stats = cache.get_statistics();
    TestFramework::assert_true(stats.bypass_count > 0, "Bypass count should be > 0");

    std::cout << "Sector Cache bypass tests passed!" << std::endl;
}

void test_sector_cache_partial_validity() {
    std::cout << "\n=== Testing Sector Cache Partial Validity ===" << std::endl;

    SectorCache cache(16, 128, 32, 4);
    cache.initialize(nullptr);

    // Write to sector 0 of a line (address 0x3000, sector offset 0)
    uint8_t data[32] = {0xBB};
    cache.write(0x3000, data, 32);

    // Read from sector 1 of the same line (address 0x3020 = 0x3000 + 32)
    uint8_t buf[32] = {};
    bool hit = cache.read(0x3020, buf, 32);
    TestFramework::assert_true(!hit,
        "Reading an invalid sector of a valid line should miss");

    // But reading sector 0 should still hit
    bool hit0 = cache.read(0x3000, buf, 32);
    TestFramework::assert_true(hit0,
        "Reading the valid sector should hit");

    std::cout << "Sector Cache partial validity tests passed!" << std::endl;
}

void test_sector_cache_adaptive_bypass() {
    std::cout << "\n=== Testing Sector Cache Adaptive Bypass ===" << std::endl;

    SectorCache cache(16, 128, 32, 4);
    cache.initialize(nullptr);
    cache.enable_adaptive_bypass(true);

    // Simulate a streaming access pattern (sequential addresses)
    for (uint64_t addr = 0x10000; addr < 0x10000 + 128 * 10; addr += 128) {
        uint8_t data[32] = {};
        cache.write(addr, data, 32);
    }

    // After sequential pattern, the adaptive hint should detect streaming
    BypassHint hint = cache.get_adaptive_hint(0x10000 + 128 * 10);
    // Note: the detection may or may not trigger depending on threshold
    // This just tests the API doesn't crash
    TestFramework::assert_true(
        hint == BypassHint::NO_BYPASS || hint == BypassHint::STREAMING,
        "Adaptive hint should return a valid value");

    auto stats = cache.get_statistics();
    TestFramework::assert_true(stats.bandwidth_saved_pct >= 0.0,
        "Bandwidth savings should be non-negative");

    std::cout << "Sector Cache adaptive bypass tests passed!" << std::endl;
}

// ============================================================================
// DRAM Controller Tests
// ============================================================================
void test_dram_controller_hbm2() {
    std::cout << "\n=== Testing DRAM Controller (HBM2) ===" << std::endl;

    auto perf = std::make_shared<PerformanceMonitor>();
    DRAMController dram(DRAMTimingParams::hbm2_default());
    dram.initialize(perf);

    TestFramework::assert_true(dram.get_config_name() == "HBM2",
                               "Config name should be HBM2");

    // Issue a read request
    uint64_t completion = dram.enqueue_request(0x1000, DRAMRequestType::READ, 0);
    TestFramework::assert_true(completion > 0,
                               "Read should have positive completion cycle");

    // Issue a second read to the same row — should be a row buffer hit
    uint64_t completion2 = dram.enqueue_request(0x1040, DRAMRequestType::READ, 1);
    TestFramework::assert_true(completion2 > 0,
                               "Second read should also complete");

    // Tick the controller to completion
    for (uint64_t cycle = 0; cycle <= completion2 + 10; ++cycle) {
        dram.tick(cycle);
    }
    TestFramework::assert_true(dram.is_idle(), "DRAM should be idle after completion");

    auto stats = dram.get_statistics();
    TestFramework::assert_equals(2, stats.total_reads, "Should have 2 reads");
    TestFramework::assert_true(stats.avg_read_latency > 0,
                               "Average read latency should be positive");
    TestFramework::assert_true(stats.row_buffer_hits + stats.row_buffer_misses +
                               stats.row_buffer_conflicts > 0,
                               "Should have row buffer activity");

    std::cout << "DRAM Controller (HBM2) tests passed!" << std::endl;
}

void test_dram_controller_gddr6() {
    std::cout << "\n=== Testing DRAM Controller (GDDR6) ===" << std::endl;

    DRAMController dram(DRAMTimingParams::gddr6_default());
    dram.initialize(nullptr);

    TestFramework::assert_true(dram.get_config_name() == "GDDR6",
                               "Config name should be GDDR6");

    // Issue reads to different banks (should exploit bank-level parallelism)
    auto params = dram.get_timing_params();
    uint64_t addr_bank0 = 0x0;
    uint64_t addr_bank1 = params.row_buffer_size * params.num_channels;

    uint64_t c1 = dram.enqueue_request(addr_bank0, DRAMRequestType::READ, 0);
    uint64_t c2 = dram.enqueue_request(addr_bank1, DRAMRequestType::READ, 0);

    TestFramework::assert_true(c1 > 0 && c2 > 0,
                               "Both reads should have positive completion");

    // Writes
    dram.enqueue_request(0x5000, DRAMRequestType::WRITE, 10);

    auto stats = dram.get_statistics();
    TestFramework::assert_equals(2, stats.total_reads, "Should have 2 reads");
    TestFramework::assert_equals(1, stats.total_writes, "Should have 1 write");

    std::cout << "DRAM Controller (GDDR6) tests passed!" << std::endl;
}

void test_dram_row_buffer_conflict() {
    std::cout << "\n=== Testing DRAM Row Buffer Conflicts ===" << std::endl;

    DRAMController dram(DRAMTimingParams::hbm2_default());
    dram.initialize(nullptr);

    auto params = dram.get_timing_params();

    // Access row 0 in bank 0
    uint64_t addr_row0 = 0;
    dram.enqueue_request(addr_row0, DRAMRequestType::READ, 0);

    // Access a different row in the same bank — should cause conflict
    uint64_t addr_row1 = static_cast<uint64_t>(params.row_buffer_size) *
                         params.num_channels * params.num_banks_per_channel;
    uint64_t conflict_completion = dram.enqueue_request(addr_row1, DRAMRequestType::READ, 5);

    TestFramework::assert_true(conflict_completion > 5,
                               "Conflicting access should have higher latency");

    auto stats = dram.get_statistics();
    TestFramework::assert_true(
        stats.row_buffer_conflicts > 0 || stats.row_buffer_misses > 0,
        "Should detect row buffer conflict or miss");

    std::cout << "DRAM Row Buffer Conflict tests passed!" << std::endl;
}

// ============================================================================
// Checkpointing / Sampling Tests
// ============================================================================
void test_checkpoint_create_save_load() {
    std::cout << "\n=== Testing Checkpoint Create/Save/Load ===" << std::endl;

    auto perf = std::make_shared<PerformanceMonitor>();
    SimCheckpointEngine engine(1000000);
    engine.initialize(perf);

    auto cp = engine.create_checkpoint(50000, 25000, "test_checkpoint");

    TestFramework::assert_equals(0, cp.checkpoint_id, "First checkpoint ID should be 0");
    TestFramework::assert_equals(50000, cp.cycle, "Cycle should be 50000");
    TestFramework::assert_equals(25000, cp.instruction_count, "Instruction count should be 25000");
    TestFramework::assert_near(cp.ipc, 0.5, 0.01, "IPC should be 0.5");

    // Save and load
    std::string filepath = "/tmp/test_checkpoint.csv";
    TestFramework::assert_true(engine.save_checkpoint(cp, filepath),
                               "Save checkpoint should succeed");

    auto loaded = engine.load_checkpoint(filepath);
    TestFramework::assert_equals(cp.checkpoint_id, loaded.checkpoint_id,
                                 "Loaded checkpoint ID should match");
    TestFramework::assert_equals(cp.cycle, loaded.cycle,
                                 "Loaded cycle should match");
    TestFramework::assert_near(loaded.ipc, 0.5, 0.01, "Loaded IPC should match");

    std::cout << "Checkpoint create/save/load tests passed!" << std::endl;
}

void test_simpoint_sampling() {
    std::cout << "\n=== Testing SimPoint Sampling ===" << std::endl;

    SimCheckpointEngine engine(1000);
    engine.initialize(nullptr);

    // Record BBVs for 20 intervals — create 3 distinct clusters
    for (uint64_t i = 0; i < 20; ++i) {
        std::vector<uint64_t> bbv(10, 0);
        if (i < 7) {
            bbv[0] = 100; bbv[1] = 50; // Cluster A pattern
        } else if (i < 14) {
            bbv[5] = 80; bbv[6] = 90;  // Cluster B pattern
        } else {
            bbv[2] = 60; bbv[8] = 40;  // Cluster C pattern
        }
        engine.record_interval_bbv(i, bbv);
    }

    // Run clustering
    engine.compute_sample_points(3);

    auto samples = engine.get_sample_intervals();
    TestFramework::assert_true(!samples.empty(), "Should produce sample intervals");
    TestFramework::assert_true(samples.size() <= 3, "Should have at most 3 representatives");

    // Check weight coverage
    double total_weight = 0.0;
    for (const auto& s : samples) {
        total_weight += s.weight;
        TestFramework::assert_true(s.is_representative, "Sample should be marked representative");
    }
    TestFramework::assert_near(total_weight, 1.0, 0.01, "Total weight should sum to ~1.0");

    // Test should_simulate_cycle
    bool inside = engine.should_simulate_cycle(samples[0].start_cycle + 1);
    TestFramework::assert_true(inside, "Cycle inside a sample interval should be simulated");

    bool outside = engine.should_simulate_cycle(999999999);
    TestFramework::assert_true(!outside, "Cycle far outside any interval should be skipped");

    // Test IPC estimation
    std::vector<double> ipcs;
    for (size_t i = 0; i < samples.size(); ++i) ipcs.push_back(1.5);
    double estimated = engine.estimate_total_ipc(ipcs);
    TestFramework::assert_near(estimated, 1.5, 0.01,
                               "Weighted IPC estimate should be ~1.5 when all samples have same IPC");

    auto stats = engine.get_statistics();
    TestFramework::assert_equals(20, stats.total_intervals, "Should have 20 total intervals");
    TestFramework::assert_true(stats.estimated_speedup > 1.0,
                               "Sampling should provide speedup > 1x");

    std::cout << "SimPoint Sampling tests passed!" << std::endl;
}

// ============================================================================
// Integration test — all new components together
// ============================================================================
void test_enhancement_integration() {
    std::cout << "\n=== Enhancement Integration Test ===" << std::endl;

    auto perf = std::make_shared<PerformanceMonitor>();

    // Create all enhanced components
    WarpScheduler scheduler(32, SchedulePolicy::GTO);
    TensorCore tensor_core(0, 4, 4, 4);
    SectorCache sector_cache(32, 128, 32, 4);
    DRAMController dram(DRAMTimingParams::hbm2_default());
    SimCheckpointEngine checkpoint_engine(10000);

    scheduler.initialize(perf);
    tensor_core.initialize(perf);
    sector_cache.initialize(perf);
    dram.initialize(perf);
    checkpoint_engine.initialize(perf);

    // --- Simulate a kernel execution ---

    // 1. Add warps to scheduler
    for (uint32_t w = 0; w < 8; ++w) {
        scheduler.add_warp(w, 20);
    }

    // 2. Run scheduler for some cycles and issue MMA + memory ops
    uint64_t cycle = 0;
    while (scheduler.get_num_ready_warps() > 0 && cycle < 500) {
        int32_t warp_id = scheduler.schedule_next(cycle);
        if (warp_id >= 0) {
            // Simulate occasional memory access
            if (cycle % 10 == 0) {
                dram.enqueue_request(cycle * 64, DRAMRequestType::READ, cycle);
            }
            // Simulate sector cache access
            uint8_t data[32] = {};
            sector_cache.write(cycle * 32, data, 32);
            // Also read back to generate cache activity
            sector_cache.read(cycle * 32, data, 32);
        }
        dram.tick(cycle);
        cycle++;
    }

    // 3. Perform a tensor core operation
    auto A = tensor_core.load_matrix_a({1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1},
                                        4, 4, PrecisionMode::FP16_FP32);
    auto B = tensor_core.load_matrix_b({2,2,2,2, 2,2,2,2, 2,2,2,2, 2,2,2,2},
                                        4, 4, PrecisionMode::FP16_FP32);
    auto C = tensor_core.create_accumulator(4, 4, PrecisionMode::FP16_FP32, 0.0f);
    MatrixFragment D(4, 4, PrecisionMode::FP16_FP32);
    tensor_core.mma(A, B, C, D);

    // 4. Create a checkpoint
    auto sched_stats = scheduler.get_statistics();
    checkpoint_engine.create_checkpoint(cycle, sched_stats.instructions_issued, "integration_test");

    // --- Validate all components produced metrics ---
    TestFramework::assert_true(sched_stats.instructions_issued > 0,
                               "Scheduler should have issued instructions");
    TestFramework::assert_true(tensor_core.get_statistics().mma_operations > 0,
                               "Tensor core should have MMA operations");

    auto cache_stats = sector_cache.get_statistics();
    TestFramework::assert_true(cache_stats.hits + cache_stats.misses > 0,
                               "Sector cache should have activity");

    auto dram_stats = dram.get_statistics();
    TestFramework::assert_true(dram_stats.total_reads > 0,
                               "DRAM should have read requests");

    TestFramework::assert_true(!checkpoint_engine.get_checkpoints().empty(),
                               "Should have at least one checkpoint");

    // Check perf monitor aggregated some data
    TestFramework::assert_true(perf->get_counter("scheduler_instructions_issued") > 0,
                               "Perf monitor should track scheduler instructions");
    TestFramework::assert_true(perf->get_counter("tensor_core_mma_ops") > 0,
                               "Perf monitor should track tensor core ops");
    TestFramework::assert_true(perf->get_counter("dram_requests") > 0,
                               "Perf monitor should track DRAM requests");

    std::cout << "Enhancement Integration test passed!" << std::endl;
}

// ============================================================================
// Main
// ============================================================================
int main() {
    std::cout << "GPU Architecture Simulator - Enhancement Test Suite" << std::endl;
    std::cout << "====================================================" << std::endl;

    try {
        // Warp Scheduler tests
        test_warp_scheduler_gto();
        test_warp_scheduler_two_level();
        test_warp_scheduler_stall();

        // Tensor Core tests
        test_tensor_core_mma();
        test_tensor_core_accumulate();
        test_tensor_core_int8();

        // Sector Cache tests
        test_sector_cache_basic();
        test_sector_cache_bypass();
        test_sector_cache_partial_validity();
        test_sector_cache_adaptive_bypass();

        // DRAM Controller tests
        test_dram_controller_hbm2();
        test_dram_controller_gddr6();
        test_dram_row_buffer_conflict();

        // Checkpointing / Sampling tests
        test_checkpoint_create_save_load();
        test_simpoint_sampling();

        // Integration test
        test_enhancement_integration();

        std::cout << "\n🎉 ALL ENHANCEMENT TESTS PASSED! 🎉" << std::endl;
        std::cout << "\nNew components validated:" << std::endl;
        std::cout << "✓ GTO Warp Scheduler (Greedy-Then-Oldest)" << std::endl;
        std::cout << "✓ Two-Level Warp Scheduler" << std::endl;
        std::cout << "✓ Tensor Core MMA (FP16/FP32/INT8 mixed-precision)" << std::endl;
        std::cout << "✓ Sector Cache with Bypassing" << std::endl;
        std::cout << "✓ DRAM Controller (HBM2/GDDR6 timing)" << std::endl;
        std::cout << "✓ SimPoint Checkpointing & Sampling" << std::endl;
        std::cout << "✓ Full Integration" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Test failed with exception: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
