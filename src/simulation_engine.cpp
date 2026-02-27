#include "simulation_engine.h"
#include "gpu_core.h"
#include "memory_hierarchy.h"
#include "warp_scheduler.h"
#include "dram_timing.h"

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>

namespace gpu_sim {

// ---------------------------------------------------------------------------
// Construction / destruction
// ---------------------------------------------------------------------------

SimulationEngine::SimulationEngine(const SimulationConfig& config)
    : config_(config)
{
    // Initialise sampling/skip schedule.
    next_event_cycle_ = config_.enable_sampling
                        ? config_.warmup_cycles
                        : config_.max_cycles;
}

SimulationEngine::~SimulationEngine() {
    stop();
}

// ---------------------------------------------------------------------------
// Attach components
// ---------------------------------------------------------------------------

void SimulationEngine::attach_gpu_core(std::shared_ptr<GPUCore> core) {
    gpu_core_ = std::move(core);
}

void SimulationEngine::attach_memory(std::shared_ptr<MemoryHierarchy> memory) {
    memory_ = std::move(memory);
}

void SimulationEngine::attach_scheduler(std::shared_ptr<WarpScheduler> scheduler) {
    scheduler_ = std::move(scheduler);
}

void SimulationEngine::attach_dram(std::shared_ptr<DRAMTimingModel> dram) {
    dram_ = std::move(dram);
}

// ---------------------------------------------------------------------------
// Run
// ---------------------------------------------------------------------------

void SimulationEngine::run(InstructionCallback callback) {
    running_ = true;
    paused_  = false;
    stats_.total_cycles = 0;
    stats_.total_instructions = 0;

    auto wall_start = std::chrono::high_resolution_clock::now();

    if (config_.num_timing_threads > 1 || config_.num_functional_threads > 1) {
        run_multi_threaded(callback);
    } else {
        run_single_threaded(callback);
    }

    auto wall_end = std::chrono::high_resolution_clock::now();
    stats_.wall_clock_seconds =
        std::chrono::duration<double>(wall_end - wall_start).count();

    // Compute aggregate IPC.
    if (stats_.total_cycles > 0) {
        stats_.ipc = static_cast<double>(stats_.total_instructions) /
                     static_cast<double>(stats_.total_cycles);
    }
    running_ = false;
}

// ---------------------------------------------------------------------------
// Single-threaded simulation loop
// ---------------------------------------------------------------------------

void SimulationEngine::run_single_threaded(InstructionCallback& cb) {
    for (uint64_t cycle = 0; cycle < config_.max_cycles && running_; ++cycle) {

        while (paused_ && running_) {
            std::this_thread::sleep_for(std::chrono::microseconds(100));
        }

        // --- Functional tick ---
        bool keep_running = tick_functional(cycle);
        if (!keep_running) break;

        // --- Timing tick ---
        tick_timing(cycle);

        // --- Update stats ---
        {
            std::lock_guard<std::mutex> lock(stats_mutex_);
            stats_.total_cycles = cycle + 1;
        }

        // --- Sampling ---
        if (config_.enable_sampling) {
            if (cycle == next_event_cycle_) {
                if (!in_sample_) {
                    start_sample(cycle);
                } else {
                    end_sample(cycle);
                    if (samples_collected_ >= config_.max_samples) break;
                }
            }
        }

        // --- Checkpointing ---
        if (config_.enable_checkpointing) {
            maybe_checkpoint(cycle);
        }

        // --- User callback ---
        if (cb) {
            std::lock_guard<std::mutex> lock(stats_mutex_);
            if (!cb(cycle, stats_.total_instructions)) break;
        }

        // --- Instruction limit ---
        if (config_.max_instructions > 0 &&
            stats_.total_instructions >= config_.max_instructions) {
            break;
        }
    }
}

// ---------------------------------------------------------------------------
// Multi-threaded simulation loop
// ---------------------------------------------------------------------------

/*
 * Decoupled functional/timing simulation:
 *
 *  Thread 0 (functional): advances by one cycle at a time producing an
 *    instruction stream.  It runs at most `window` cycles ahead of the timing
 *    thread.  A simple atomic cycle counter provides back-pressure.
 *
 *  Thread 1 (timing): consumes the instruction stream, updates DRAM/cache
 *    timing models, and advances the timing cycle counter.
 *
 * For simplicity the current implementation uses a single functional thread
 * and a single timing thread; the config allows extending to N threads by
 * partitioning the warp space.
 */
void SimulationEngine::run_multi_threaded(InstructionCallback& cb) {
    constexpr uint64_t WINDOW = 1000; // functional runs at most WINDOW ahead

    std::atomic<uint64_t> functional_cycle{0};
    std::atomic<uint64_t> timing_cycle{0};

    // Functional thread.
    workers_.emplace_back([&]() {
        for (uint64_t cycle = 0; cycle < config_.max_cycles && running_; ++cycle) {
            // Back-pressure: don't run more than WINDOW ahead of timing.
            while ((cycle - timing_cycle.load()) > WINDOW && running_) {
                std::this_thread::yield();
            }

            bool keep = tick_functional(cycle);
            functional_cycle.store(cycle);

            if (!keep) { running_ = false; break; }

            if (config_.max_instructions > 0) {
                std::lock_guard<std::mutex> lock(stats_mutex_);
                if (stats_.total_instructions >= config_.max_instructions) {
                    running_ = false; break;
                }
            }
        }
        running_ = false;
    });

    // Timing thread.
    workers_.emplace_back([&]() {
        for (uint64_t cycle = 0; cycle < config_.max_cycles && running_; ++cycle) {
            // Wait until the functional thread has processed this cycle.
            while (functional_cycle.load() < cycle && running_) {
                std::this_thread::yield();
            }

            tick_timing(cycle);
            timing_cycle.store(cycle);

            {
                std::lock_guard<std::mutex> lock(stats_mutex_);
                stats_.total_cycles = cycle + 1;
            }

            if (config_.enable_sampling && cycle == next_event_cycle_) {
                if (!in_sample_) start_sample(cycle);
                else {
                    end_sample(cycle);
                    if (samples_collected_ >= config_.max_samples) {
                        running_ = false; break;
                    }
                }
            }

            if (config_.enable_checkpointing) maybe_checkpoint(cycle);

            if (cb) {
                std::lock_guard<std::mutex> lock(stats_mutex_);
                if (!cb(cycle, stats_.total_instructions)) { running_ = false; break; }
            }
        }
        running_ = false;
    });

    for (auto& t : workers_) {
        if (t.joinable()) t.join();
    }
    workers_.clear();
}

// ---------------------------------------------------------------------------
// Per-cycle functional tick
// ---------------------------------------------------------------------------

bool SimulationEngine::tick_functional(uint64_t cycle) {
    // Advance the warp scheduler.
    if (scheduler_) {
        scheduler_->tick(cycle);
        Warp* w = scheduler_->select_next_warp(cycle);
        if (w) {
            // Count one instruction issued this cycle.
            std::lock_guard<std::mutex> lock(stats_mutex_);
            stats_.total_instructions++;
        }
    } else {
        // No scheduler attached: count every cycle as one instruction.
        std::lock_guard<std::mutex> lock(stats_mutex_);
        stats_.total_instructions++;
    }

    // Update occupancy (fraction of ready warps).
    if (scheduler_) {
        uint32_t ready   = scheduler_->num_ready_warps();
        uint32_t stalled = scheduler_->num_stalled_warps();
        uint32_t total   = ready + stalled;
        if (total > 0) {
            float occ = static_cast<float>(ready) / static_cast<float>(total);
            std::lock_guard<std::mutex> lock(stats_mutex_);
            stats_.warp_occupancy = occ;
            if (stalled > 0) stats_.memory_stall_cycles++;
        }
    }

    return true; // continue simulation
}

// ---------------------------------------------------------------------------
// Per-cycle timing tick
// ---------------------------------------------------------------------------

void SimulationEngine::tick_timing(uint64_t cycle) {
    if (dram_) {
        dram_->tick(cycle);
    }
}

// ---------------------------------------------------------------------------
// Control
// ---------------------------------------------------------------------------

void SimulationEngine::pause()  { paused_  = true; }
void SimulationEngine::resume() { paused_  = false; }

void SimulationEngine::stop() {
    running_ = false;
    paused_  = false;
    for (auto& t : workers_) {
        if (t.joinable()) t.join();
    }
    workers_.clear();
}

// ---------------------------------------------------------------------------
// Checkpointing
// ---------------------------------------------------------------------------

SimulationCheckpoint SimulationEngine::create_checkpoint(const std::string& label) {
    SimulationCheckpoint ckpt;
    {
        std::lock_guard<std::mutex> lock(stats_mutex_);
        ckpt.cycle = stats_.total_cycles;
        ckpt.instructions_retired = stats_.total_instructions;
        ckpt.ipc  = stats_.ipc;
        ckpt.occupancy = stats_.warp_occupancy;
    }
    ckpt.label = label.empty()
                 ? ("ckpt_" + std::to_string(ckpt.cycle))
                 : label;
    return ckpt;
}

void SimulationEngine::restore_checkpoint(const SimulationCheckpoint& ckpt) {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    stats_.total_cycles       = ckpt.cycle;
    stats_.total_instructions = ckpt.instructions_retired;
    stats_.ipc                = ckpt.ipc;
    stats_.warp_occupancy     = ckpt.occupancy;
}

void SimulationEngine::maybe_checkpoint(uint64_t cycle) {
    if (cycle > 0 && (cycle % config_.checkpoint_interval) == 0) {
        auto ckpt = create_checkpoint();
        // Write a minimal text checkpoint to disk.
        std::string fname = checkpoint_filename(cycle);
        std::ofstream f(fname);
        if (f.is_open()) {
            f << "cycle=" << ckpt.cycle << "\n"
              << "instructions=" << ckpt.instructions_retired << "\n"
              << "ipc=" << ckpt.ipc << "\n"
              << "occupancy=" << ckpt.occupancy << "\n"
              << "label=" << ckpt.label << "\n";
        }
    }
}

std::string SimulationEngine::checkpoint_filename(uint64_t cycle) const {
    return config_.checkpoint_dir + "/ckpt_" + std::to_string(cycle) + ".txt";
}

std::vector<std::string> SimulationEngine::list_checkpoints() const {
    std::vector<std::string> result;
    try {
        for (auto& entry : std::filesystem::directory_iterator(config_.checkpoint_dir)) {
            auto name = entry.path().filename().string();
            if (name.rfind("ckpt_", 0) == 0 &&
                name.size() > 5 && name.back() == 't') {
                result.push_back(entry.path().string());
            }
        }
    } catch (...) {
        // Directory may not exist yet.
    }
    std::sort(result.begin(), result.end());
    return result;
}

// ---------------------------------------------------------------------------
// Sampling
// ---------------------------------------------------------------------------

void SimulationEngine::start_sample(uint64_t cycle) {
    in_sample_          = true;
    sample_start_cycle_ = cycle;
    {
        std::lock_guard<std::mutex> lock(stats_mutex_);
        sample_start_instr_ = stats_.total_instructions;
    }
    next_event_cycle_ = cycle + config_.sample_length;
}

void SimulationEngine::end_sample(uint64_t cycle) {
    in_sample_ = false;
    samples_collected_++;

    uint64_t sample_cycles = cycle - sample_start_cycle_;
    uint64_t sample_instr;
    {
        std::lock_guard<std::mutex> lock(stats_mutex_);
        sample_instr = stats_.total_instructions - sample_start_instr_;
    }
    double sample_ipc = sample_cycles > 0
                        ? static_cast<double>(sample_instr) / sample_cycles
                        : 0.0;

    {
        std::lock_guard<std::mutex> lock(stats_mutex_);
        stats_.sample_ipcs.push_back(sample_ipc);
        stats_.sample_cycles.push_back(cycle);
    }

    // Schedule next sample after skip_cycles fast-forward.
    next_event_cycle_ = cycle + config_.skip_cycles + config_.warmup_cycles;
}

const std::vector<double>& SimulationEngine::get_sample_ipcs() const {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    return stats_.sample_ipcs;
}

// ---------------------------------------------------------------------------
// Statistics
// ---------------------------------------------------------------------------

SimulationStats SimulationEngine::get_stats() const {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    return stats_;
}

void SimulationEngine::print_summary() const {
    auto s = get_stats();
    std::cout << "=== SimulationEngine Summary ===\n"
              << "  Total cycles      : " << s.total_cycles << "\n"
              << "  Total instructions: " << s.total_instructions << "\n"
              << "  IPC               : " << s.ipc << "\n"
              << "  Warp occupancy    : " << (s.warp_occupancy * 100.0f) << "%\n"
              << "  Memory stalls     : " << s.memory_stall_cycles << "\n"
              << "  Wall clock        : " << s.wall_clock_seconds << " s\n";
    if (!s.sample_ipcs.empty()) {
        std::cout << "  Sample IPCs       :";
        for (double ipc : s.sample_ipcs) std::cout << " " << ipc;
        std::cout << "\n";
    }
}

} // namespace gpu_sim
