#pragma once

#include <vector>
#include <deque>
#include <cstdint>
#include <string>

namespace gpu_sim {

/**
 * Represents a single GPU warp (32 threads executing in SIMT lockstep).
 */
struct Warp {
    uint32_t warp_id;
    uint32_t thread_count;       // active threads (bitmask popcount)
    uint32_t active_mask;        // 32-bit bitmask of active threads
    uint64_t pc;                 // program counter
    uint64_t birth_cycle;        // cycle when warp became ready
    bool     ready;              // true if warp has a ready instruction
    bool     waiting_on_memory;  // true if stalled on a memory operation

    Warp(uint32_t id, uint64_t cycle)
        : warp_id(id), thread_count(32), active_mask(0xFFFFFFFF),
          pc(0), birth_cycle(cycle), ready(true), waiting_on_memory(false) {}
};

/**
 * Scheduling policy selector.
 */
enum class SchedulerPolicy {
    ROUND_ROBIN,         // baseline round-robin
    GTO,                 // Greedy-Then-Oldest (favors cache reuse)
    TWO_LEVEL,           // Two-Level (inner GTO, outer oldest-first fetch)
    LOOSE_ROUND_ROBIN,   // skips stalled warps, otherwise round-robin
};

/**
 * WarpScheduler – implements the Greedy-Then-Oldest (GTO) and Two-Level
 * scheduling policies described in Narasiman et al. (MICRO 2011).
 *
 * GTO: Keep issuing from the same warp until it stalls, then fall back to
 *      the oldest ready warp.  This preserves L1 cache reuse within a warp
 *      and reduces cache thrashing from over-subscription.
 *
 * Two-Level: Divide the warp pool into a small "fetch group" (inner level)
 *            scheduled by GTO, and a larger "pending group" (outer level)
 *            that replenishes the fetch group in oldest-first order.
 */
class WarpScheduler {
public:
    /**
     * @param num_warps    Total warps managed by this scheduler.
     * @param policy       Scheduling policy to use.
     * @param fetch_group  Size of the inner fetch group (Two-Level only).
     */
    explicit WarpScheduler(uint32_t num_warps,
                           SchedulerPolicy policy = SchedulerPolicy::GTO,
                           uint32_t fetch_group_size = 8);
    ~WarpScheduler() = default;

    // ---- Lifecycle --------------------------------------------------------

    /** Register a warp as ready to be scheduled. */
    void add_warp(uint32_t warp_id, uint64_t current_cycle);

    /** Notify the scheduler that a warp has stalled (e.g., memory latency). */
    void stall_warp(uint32_t warp_id);

    /** Notify the scheduler that a previously stalled warp is now ready. */
    void ready_warp(uint32_t warp_id, uint64_t current_cycle);

    /** Mark a warp as completed (removes it from all queues). */
    void complete_warp(uint32_t warp_id);

    // ---- Core interface ---------------------------------------------------

    /**
     * Select the next warp to issue an instruction.
     * Returns the selected Warp pointer, or nullptr if no warp is ready.
     * @param current_cycle  Simulator cycle (used for age comparisons).
     */
    Warp* select_next_warp(uint64_t current_cycle);

    /** Advance scheduler state by one cycle (updates Two-Level fetch group). */
    void tick(uint64_t current_cycle);

    // ---- Diagnostics ------------------------------------------------------

    uint32_t num_ready_warps()   const;
    uint32_t num_stalled_warps() const;
    bool     has_ready_warps()   const { return num_ready_warps() > 0; }

    /** Returns the warp_id of the currently preferred warp (-1 if none). */
    int32_t  current_greedy_warp() const { return greedy_warp_id_; }

    /** Per-warp issue counts (for occupancy / IPC tracking). */
    uint64_t get_issue_count(uint32_t warp_id) const;

    /** Total instructions issued across all warps. */
    uint64_t total_issues() const { return total_issues_; }

    /** Total stall cycles encountered. */
    uint64_t total_stalls() const { return total_stalls_; }

    /** Human-readable summary of scheduler state. */
    std::string get_stats_summary() const;

private:
    // ---- GTO helpers ------------------------------------------------------
    Warp* gto_select(uint64_t current_cycle);
    Warp* oldest_ready_warp();

    // ---- Two-Level helpers ------------------------------------------------
    void  two_level_refill_fetch_group();
    Warp* two_level_select(uint64_t current_cycle);

    // ---- State ------------------------------------------------------------
    uint32_t                num_warps_;
    SchedulerPolicy         policy_;
    uint32_t                fetch_group_size_;

    std::vector<Warp>       warps_;            // all warp state
    std::vector<bool>       active_;           // warp is still running
    std::vector<bool>       stalled_;          // warp is currently stalled

    int32_t                 greedy_warp_id_;   // GTO: currently preferred warp
    uint32_t                greedy_streak_;    // consecutive issues from greedy warp

    // Two-Level scheduler: warps in the inner fetch group
    std::deque<uint32_t>    fetch_group_;      // inner: GTO among these
    std::deque<uint32_t>    pending_group_;    // outer: FIFO of remaining warps

    // Metrics
    std::vector<uint64_t>   issue_counts_;
    uint64_t                total_issues_;
    uint64_t                total_stalls_;
};

} // namespace gpu_sim
