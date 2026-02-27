#pragma once

#include <atomic>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>
#include <cstdint>

namespace gpu_sim {

// Forward declarations
class GPUCore;
class MemoryHierarchy;
class WarpScheduler;
class DRAMTimingModel;

// ---------------------------------------------------------------------------
// Simulation state (used for checkpointing / sampling)
// ---------------------------------------------------------------------------

/**
 * A lightweight snapshot of the simulator state at a specific cycle.
 * Sufficient to resume simulation from this point for sampling workflows.
 */
struct SimulationCheckpoint {
    uint64_t    cycle;                  // snapshot cycle
    uint64_t    instructions_retired;  // total ISAs retired
    double      ipc;                   // IPC at checkpoint
    float       occupancy;             // active warps / max warps
    std::string label;                 // human-readable tag
    std::vector<uint8_t> serialised;   // opaque blobs for future restore
};

// ---------------------------------------------------------------------------
// Simulation configuration
// ---------------------------------------------------------------------------

struct SimulationConfig {
    // Parallelism
    uint32_t num_timing_threads   = 1;  // threads for timing model (0 = auto)
    uint32_t num_functional_threads = 1; // threads for functional simulation
    bool     decouple_functional  = false; // run func sim ahead of timing sim

    // Checkpointing
    bool     enable_checkpointing = false;
    uint64_t checkpoint_interval  = 1'000'000; // cycles between checkpoints
    std::string checkpoint_dir    = ".";       // directory for checkpoint files

    // Sampling (SimPoint-style)
    bool     enable_sampling      = false;
    uint64_t warmup_cycles        = 100'000;  // cycles to warm caches before sample
    uint64_t sample_length        = 500'000;  // cycles per sample window
    uint64_t skip_cycles          = 5'000'000; // cycles skipped between samples
    uint32_t max_samples          = 10;

    // Termination
    uint64_t max_cycles           = 100'000'000; // hard limit
    uint64_t max_instructions     = 0;           // 0 = unlimited
};

// ---------------------------------------------------------------------------
// Simulation statistics collected each cycle
// ---------------------------------------------------------------------------

struct SimulationStats {
    uint64_t total_cycles         = 0;
    uint64_t total_instructions   = 0;
    uint64_t total_stall_cycles   = 0;
    uint64_t memory_stall_cycles  = 0;
    double   ipc                  = 0.0;
    float    warp_occupancy       = 0.0;   // fraction of warps active
    double   wall_clock_seconds   = 0.0;

    // Per-sample IPC (sampling mode)
    std::vector<double> sample_ipcs;
    std::vector<uint64_t> sample_cycles;
};

// ---------------------------------------------------------------------------
// SimulationEngine
// ---------------------------------------------------------------------------

/**
 * SimulationEngine orchestrates the GPU simulation loop.
 *
 * Architecture
 * ------------
 *  ┌────────────────┐      shared memory      ┌──────────────────┐
 *  │  Functional    │ ─── instruction stream ──► Timing           │
 *  │  Simulation    │                          │  Simulation      │
 *  │  Thread(s)     │ ◄── back-pressure ───── │  Thread(s)       │
 *  └────────────────┘                          └──────────────────┘
 *           │                                           │
 *           └──────────── checkpoints ─────────────────┘
 *
 * Decoupled mode
 * --------------
 * The functional simulator runs ahead of the timing model (up to a configurable
 * window).  This hides functional instruction latency and lets the timing model
 * operate on a pre-computed instruction stream — a common technique in academic
 * simulators (MARSS, ZSim, gem5 TimingSimpleCPU).
 *
 * Checkpointing
 * -------------
 * Every checkpoint_interval cycles the engine serialises the current
 * SimulationCheckpoint to <checkpoint_dir>/ckpt_<cycle>.bin.  The engine can
 * resume from the latest checkpoint (useful for long-running kernel traces).
 *
 * Sampling (SimPoint-style)
 * -------------------------
 * After the warmup period the engine executes sample_length cycles, records
 * IPC and occupancy, then fast-forwards skip_cycles before the next sample.
 * The resulting sample_ipcs vector is used by the regression script.
 */
class SimulationEngine {
public:
    explicit SimulationEngine(const SimulationConfig& config = SimulationConfig{});
    ~SimulationEngine();

    // ---- Attach components ------------------------------------------------

    void attach_gpu_core(std::shared_ptr<GPUCore> core);
    void attach_memory(std::shared_ptr<MemoryHierarchy> memory);
    void attach_scheduler(std::shared_ptr<WarpScheduler> scheduler);
    void attach_dram(std::shared_ptr<DRAMTimingModel> dram);

    // ---- Simulation control -----------------------------------------------

    /**
     * Run the simulation.
     * @param instruction_callback  Called once per simulated instruction.
     *        Return false to stop the simulation early.
     */
    using InstructionCallback = std::function<bool(uint64_t cycle,
                                                    uint64_t instructions)>;
    void run(InstructionCallback callback = nullptr);

    /** Pause a running simulation (safe to call from another thread). */
    void pause();

    /** Resume a paused simulation. */
    void resume();

    /** Stop the simulation and wait for threads to finish. */
    void stop();

    // ---- Checkpointing ----------------------------------------------------

    /** Save a checkpoint at the current simulation cycle. */
    SimulationCheckpoint create_checkpoint(const std::string& label = "");

    /**
     * Restore state from a checkpoint.
     * Currently only restores cycle counter and statistics; a full restore
     * of architectural state requires hooking into the attached components.
     */
    void restore_checkpoint(const SimulationCheckpoint& ckpt);

    /** List checkpoints saved to checkpoint_dir. */
    std::vector<std::string> list_checkpoints() const;

    // ---- Sampling ---------------------------------------------------------

    /** Returns collected per-sample IPC values (populated after run()). */
    const std::vector<double>& get_sample_ipcs() const;

    // ---- Statistics -------------------------------------------------------

    SimulationStats get_stats() const;

    /** Print a human-readable simulation summary. */
    void print_summary() const;

private:
    // ---- Internal simulation loop variants --------------------------------
    void run_single_threaded(InstructionCallback& cb);
    void run_multi_threaded(InstructionCallback& cb);

    // ---- Per-cycle work ---------------------------------------------------
    bool tick_functional(uint64_t cycle);   // returns false to stop
    void tick_timing(uint64_t cycle);

    // ---- Checkpointing helpers --------------------------------------------
    void maybe_checkpoint(uint64_t cycle);
    std::string checkpoint_filename(uint64_t cycle) const;

    // ---- Sampling helpers -------------------------------------------------
    void start_sample(uint64_t cycle);
    void end_sample(uint64_t cycle);

    // ---- State ------------------------------------------------------------
    SimulationConfig            config_;
    SimulationStats             stats_;

    std::shared_ptr<GPUCore>          gpu_core_;
    std::shared_ptr<MemoryHierarchy>  memory_;
    std::shared_ptr<WarpScheduler>    scheduler_;
    std::shared_ptr<DRAMTimingModel>  dram_;

    // Threading
    std::vector<std::thread>    workers_;
    std::atomic<bool>           running_{false};
    std::atomic<bool>           paused_{false};
    mutable std::mutex          stats_mutex_;

    // Sampling state
    bool        in_sample_   = false;
    uint64_t    sample_start_cycle_      = 0;
    uint64_t    sample_start_instr_      = 0;
    uint32_t    samples_collected_       = 0;
    uint64_t    next_event_cycle_        = 0;   // next sample/skip boundary
};

} // namespace gpu_sim
