#include "warp_scheduler.h"
#include "performance_monitor.h"
#include <algorithm>
#include <limits>
#include <iostream>

namespace gpu_sim {

WarpScheduler::WarpScheduler(uint32_t num_warp_slots, SchedulePolicy policy)
    : max_warp_slots_(num_warp_slots),
      policy_(policy),
      current_active_warp_(-1),
      rr_index_(0),
      two_level_group_size_(8) {
    warps_.reserve(max_warp_slots_);
    stats_ = SchedulerStats{};
}

void WarpScheduler::initialize(std::shared_ptr<PerformanceMonitor> perf_monitor) {
    perf_monitor_ = perf_monitor;
    if (perf_monitor_) {
        perf_monitor_->set_counter("scheduler_max_warps", max_warp_slots_);
    }
}

void WarpScheduler::add_warp(uint32_t warp_id, uint32_t num_instructions) {
    if (warps_.size() >= max_warp_slots_) return;

    warps_.emplace_back(warp_id, num_instructions);

    // Two-Level: add to fetch group by default
    if (policy_ == SchedulePolicy::TWO_LEVEL) {
        if (fetch_group_.size() < two_level_group_size_) {
            fetch_group_.push_back(warp_id);
        } else {
            ready_group_.push_back(warp_id);
        }
    }
}

void WarpScheduler::remove_warp(uint32_t warp_id) {
    warps_.erase(
        std::remove_if(warps_.begin(), warps_.end(),
                       [warp_id](const Warp& w) { return w.warp_id == warp_id; }),
        warps_.end());

    if (current_active_warp_ == static_cast<int32_t>(warp_id)) {
        current_active_warp_ = -1;
    }
}

void WarpScheduler::set_warp_state(uint32_t warp_id, WarpState new_state) {
    for (auto& w : warps_) {
        if (w.warp_id == warp_id) {
            w.state = new_state;
            break;
        }
    }
}

// ---------------------------------------------------------------------------
// GTO Scheduling
// ---------------------------------------------------------------------------
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
        for (auto& w : warps_) {
            if (static_cast<int32_t>(w.warp_id) == oldest) {
                w.last_issued_cycle = current_cycle;
                w.instructions_remaining--;
                stats_.instructions_issued++;
                if (w.instructions_remaining == 0) {
                    w.state = WarpState::COMPLETED;
                    stats_.warps_completed++;
                    current_active_warp_ = -1;
                }
                break;
            }
        }
    }
    return oldest;
}

// ---------------------------------------------------------------------------
// Two-Level Scheduling
// ---------------------------------------------------------------------------
int32_t WarpScheduler::schedule_two_level(uint64_t current_cycle) {
    // Try fetch group first — these warps have priority
    for (auto it = fetch_group_.begin(); it != fetch_group_.end(); ++it) {
        for (auto& w : warps_) {
            if (w.warp_id == *it && w.state == WarpState::READY &&
                w.instructions_remaining > 0) {
                w.last_issued_cycle = current_cycle;
                w.instructions_remaining--;
                stats_.instructions_issued++;
                if (w.instructions_remaining == 0) {
                    w.state = WarpState::COMPLETED;
                    stats_.warps_completed++;
                    fetch_group_.erase(it);
                    // Promote one from ready group
                    if (!ready_group_.empty()) {
                        fetch_group_.push_back(ready_group_.front());
                        ready_group_.pop_front();
                    }
                }
                return static_cast<int32_t>(w.warp_id);
            }
        }
    }

    // If no fetch-group warp is ready, try ready group
    for (auto it = ready_group_.begin(); it != ready_group_.end(); ++it) {
        for (auto& w : warps_) {
            if (w.warp_id == *it && w.state == WarpState::READY &&
                w.instructions_remaining > 0) {
                w.last_issued_cycle = current_cycle;
                w.instructions_remaining--;
                stats_.instructions_issued++;
                if (w.instructions_remaining == 0) {
                    w.state = WarpState::COMPLETED;
                    stats_.warps_completed++;
                    ready_group_.erase(it);
                }
                return static_cast<int32_t>(w.warp_id);
            }
        }
    }

    return -1;
}

// ---------------------------------------------------------------------------
// Round-Robin Scheduling
// ---------------------------------------------------------------------------
int32_t WarpScheduler::schedule_round_robin(uint64_t current_cycle) {
    if (warps_.empty()) return -1;

    uint32_t start = rr_index_;
    do {
        if (rr_index_ < warps_.size()) {
            auto& w = warps_[rr_index_];
            if (w.state == WarpState::READY && w.instructions_remaining > 0) {
                w.last_issued_cycle = current_cycle;
                w.instructions_remaining--;
                stats_.instructions_issued++;
                int32_t selected = static_cast<int32_t>(w.warp_id);
                if (w.instructions_remaining == 0) {
                    w.state = WarpState::COMPLETED;
                    stats_.warps_completed++;
                }
                rr_index_ = (rr_index_ + 1) % warps_.size();
                return selected;
            }
        }
        rr_index_ = (rr_index_ + 1) % warps_.size();
    } while (rr_index_ != start);

    return -1;
}

// ---------------------------------------------------------------------------
// Public schedule interface
// ---------------------------------------------------------------------------
int32_t WarpScheduler::schedule_next(uint64_t current_cycle) {
    stats_.total_cycles++;
    int32_t result = -1;

    switch (policy_) {
        case SchedulePolicy::GTO:
            result = schedule_gto(current_cycle);
            break;
        case SchedulePolicy::TWO_LEVEL:
            result = schedule_two_level(current_cycle);
            break;
        case SchedulePolicy::ROUND_ROBIN:
            result = schedule_round_robin(current_cycle);
            break;
    }

    if (result < 0) {
        stats_.idle_cycles++;
    }

    if (perf_monitor_) {
        perf_monitor_->increment_counter("scheduler_cycles");
        if (result >= 0) {
            perf_monitor_->increment_counter("scheduler_instructions_issued");
        }
    }

    return result;
}

void WarpScheduler::notify_memory_complete(uint32_t warp_id, uint64_t /*cycle*/) {
    for (auto& w : warps_) {
        if (w.warp_id == warp_id && w.state == WarpState::WAITING) {
            w.state = WarpState::READY;
            w.has_pending_memory_op = false;
            break;
        }
    }
}

int32_t WarpScheduler::find_oldest_ready_warp() const {
    int32_t oldest_id = -1;
    uint64_t min_age = std::numeric_limits<uint64_t>::max();

    for (const auto& w : warps_) {
        if (w.state == WarpState::READY && w.instructions_remaining > 0) {
            if (w.age < min_age) {
                min_age = w.age;
                oldest_id = static_cast<int32_t>(w.warp_id);
            }
        }
    }
    return oldest_id;
}

uint32_t WarpScheduler::get_num_active_warps() const {
    return static_cast<uint32_t>(std::count_if(
        warps_.begin(), warps_.end(),
        [](const Warp& w) { return w.state != WarpState::COMPLETED; }));
}

uint32_t WarpScheduler::get_num_ready_warps() const {
    return static_cast<uint32_t>(std::count_if(
        warps_.begin(), warps_.end(),
        [](const Warp& w) { return w.state == WarpState::READY && w.instructions_remaining > 0; }));
}

WarpScheduler::SchedulerStats WarpScheduler::get_statistics() const {
    SchedulerStats s = stats_;
    if (s.total_cycles > 0) {
        s.throughput_ipc = static_cast<double>(s.instructions_issued) / s.total_cycles;
    }
    uint64_t active_sum = 0;
    for (const auto& w : warps_) {
        if (w.state != WarpState::COMPLETED) active_sum++;
    }
    s.avg_active_warps = warps_.empty()
        ? 0.0
        : static_cast<double>(active_sum);
    return s;
}

void WarpScheduler::reset_statistics() {
    stats_ = SchedulerStats{};
}

} // namespace gpu_sim
