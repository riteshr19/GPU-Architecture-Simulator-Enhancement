#pragma once

#include <vector>
#include <deque>
#include <memory>
#include <cstdint>
#include <functional>

namespace gpu_sim {

class PerformanceMonitor;

/**
 * Warp states in the execution lifecycle
 */
enum class WarpState {
    READY,      // Warp is ready to issue an instruction
    WAITING,    // Warp is waiting on a long-latency operation (e.g., memory)
    COMPLETED,  // Warp has finished execution
    BLOCKED     // Warp is blocked on a barrier or dependency
};

/**
 * Represents a single warp (a group of threads executing in lockstep)
 */
struct Warp {
    uint32_t warp_id;
    WarpState state;
    uint64_t pc;                // Program counter
    uint32_t active_mask;       // Active thread mask (for divergence)
    uint64_t age;               // Cycle when warp became ready (for oldest-first)
    uint64_t last_issued_cycle; // Cycle of last instruction issue
    uint32_t instructions_remaining;
    bool has_pending_memory_op;

    Warp(uint32_t id, uint32_t num_instructions)
        : warp_id(id), state(WarpState::READY), pc(0),
          active_mask(0xFFFFFFFF), age(0), last_issued_cycle(0),
          instructions_remaining(num_instructions),
          has_pending_memory_op(false) {}
};

/**
 * Scheduling policy enumeration
 */
enum class SchedulePolicy {
    ROUND_ROBIN,
    GTO,        // Greedy-Then-Oldest
    TWO_LEVEL   // Two-Level warp scheduling
};

/**
 * Greedy-Then-Oldest (GTO) Warp Scheduler
 *
 * The GTO policy greedily issues instructions from the same warp until it
 * stalls (e.g., on a cache miss). When a stall occurs, the scheduler picks
 * the oldest ready warp. This maximizes data locality for the active warp
 * while maintaining fairness through the oldest-first fallback.
 *
 * Reference: Rogers et al., "Cache-Conscious Wavefront Scheduling", MICRO 2012
 */
class WarpScheduler {
public:
    explicit WarpScheduler(uint32_t num_warp_slots = 32,
                           SchedulePolicy policy = SchedulePolicy::GTO);
    ~WarpScheduler() = default;

    // Initialization
    void initialize(std::shared_ptr<PerformanceMonitor> perf_monitor);

    // Warp management
    void add_warp(uint32_t warp_id, uint32_t num_instructions);
    void remove_warp(uint32_t warp_id);
    void set_warp_state(uint32_t warp_id, WarpState new_state);

    // Core scheduling interface — returns the warp_id to issue next, or -1
    int32_t schedule_next(uint64_t current_cycle);

    // Notify the scheduler that a warp has completed a memory operation
    void notify_memory_complete(uint32_t warp_id, uint64_t cycle);

    // Policy management
    void set_policy(SchedulePolicy policy) { policy_ = policy; }
    SchedulePolicy get_policy() const { return policy_; }

    // Statistics
    struct SchedulerStats {
        uint64_t total_cycles;
        uint64_t idle_cycles;          // Cycles with no ready warp
        uint64_t warps_completed;
        uint64_t instructions_issued;
        uint64_t stalls_on_memory;
        uint64_t policy_switches;      // GTO: switches from greedy to oldest
        double avg_active_warps;
        double throughput_ipc;
    };

    SchedulerStats get_statistics() const;
    void reset_statistics();

    // Accessors
    uint32_t get_num_active_warps() const;
    uint32_t get_num_ready_warps() const;
    const std::vector<Warp>& get_warps() const { return warps_; }

private:
    // GTO scheduling
    int32_t schedule_gto(uint64_t current_cycle);

    // Two-Level scheduling
    int32_t schedule_two_level(uint64_t current_cycle);

    // Round-Robin scheduling
    int32_t schedule_round_robin(uint64_t current_cycle);

    // Find the oldest ready warp
    int32_t find_oldest_ready_warp() const;

    std::vector<Warp> warps_;
    uint32_t max_warp_slots_;
    SchedulePolicy policy_;
    int32_t current_active_warp_;   // The warp GTO is "greedy" on
    uint32_t rr_index_;             // Round-robin index

    // Two-Level scheduling groups
    std::deque<uint32_t> fetch_group_;
    std::deque<uint32_t> ready_group_;
    uint32_t two_level_group_size_;

    // Statistics
    mutable SchedulerStats stats_;
    std::shared_ptr<PerformanceMonitor> perf_monitor_;
};

} // namespace gpu_sim
