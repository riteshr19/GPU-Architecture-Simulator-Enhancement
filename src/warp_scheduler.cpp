#include "warp_scheduler.h"
#include <algorithm>
#include <sstream>
#include <stdexcept>

namespace gpu_sim {

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

WarpScheduler::WarpScheduler(uint32_t num_warps,
                             SchedulerPolicy policy,
                             uint32_t fetch_group_size)
    : num_warps_(num_warps),
      policy_(policy),
      fetch_group_size_(fetch_group_size),
      greedy_warp_id_(-1),
      greedy_streak_(0),
      total_issues_(0),
      total_stalls_(0)
{
    warps_.reserve(num_warps_);
    active_.assign(num_warps_, false);
    stalled_.assign(num_warps_, false);
    issue_counts_.assign(num_warps_, 0);
}

// ---------------------------------------------------------------------------
// Lifecycle
// ---------------------------------------------------------------------------

void WarpScheduler::add_warp(uint32_t warp_id, uint64_t current_cycle) {
    if (warp_id >= num_warps_) {
        throw std::out_of_range("warp_id out of range");
    }

    if (warps_.size() <= warp_id) {
        warps_.resize(warp_id + 1, Warp(0, 0));
    }
    warps_[warp_id] = Warp(warp_id, current_cycle);
    active_[warp_id]  = true;
    stalled_[warp_id] = false;

    if (policy_ == SchedulerPolicy::TWO_LEVEL) {
        // Place in the pending group; fetch group will be refilled via tick().
        pending_group_.push_back(warp_id);
        two_level_refill_fetch_group();
    }
}

void WarpScheduler::stall_warp(uint32_t warp_id) {
    if (warp_id >= num_warps_ || !active_[warp_id]) return;
    stalled_[warp_id] = true;
    warps_[warp_id].ready = false;
    warps_[warp_id].waiting_on_memory = true;
    total_stalls_++;

    // If the greedy warp just stalled, reset so we pick oldest next cycle.
    if (static_cast<int32_t>(warp_id) == greedy_warp_id_) {
        greedy_warp_id_ = -1;
        greedy_streak_  = 0;
    }
}

void WarpScheduler::ready_warp(uint32_t warp_id, uint64_t /*current_cycle*/) {
    if (warp_id >= num_warps_ || !active_[warp_id]) return;
    stalled_[warp_id] = false;
    warps_[warp_id].ready = true;
    warps_[warp_id].waiting_on_memory = false;
}

void WarpScheduler::complete_warp(uint32_t warp_id) {
    if (warp_id >= num_warps_) return;
    active_[warp_id]  = false;
    stalled_[warp_id] = false;
    if (warps_.size() > warp_id) {
        warps_[warp_id].ready = false;
    }
    if (static_cast<int32_t>(warp_id) == greedy_warp_id_) {
        greedy_warp_id_ = -1;
        greedy_streak_  = 0;
    }
    // Remove from Two-Level queues if present.
    auto remove_from_deque = [&](std::deque<uint32_t>& dq) {
        dq.erase(std::remove(dq.begin(), dq.end(), warp_id), dq.end());
    };
    remove_from_deque(fetch_group_);
    remove_from_deque(pending_group_);
}

// ---------------------------------------------------------------------------
// Core interface
// ---------------------------------------------------------------------------

Warp* WarpScheduler::select_next_warp(uint64_t current_cycle) {
    Warp* selected = nullptr;

    switch (policy_) {
        case SchedulerPolicy::GTO:
        case SchedulerPolicy::LOOSE_ROUND_ROBIN:
            selected = gto_select(current_cycle);
            break;

        case SchedulerPolicy::TWO_LEVEL:
            selected = two_level_select(current_cycle);
            break;

        case SchedulerPolicy::ROUND_ROBIN: {
            // Simple round-robin baseline
            static uint32_t rr_idx = 0;
            for (uint32_t i = 0; i < num_warps_; ++i) {
                uint32_t idx = (rr_idx + i) % num_warps_;
                if (active_[idx] && !stalled_[idx] &&
                    idx < warps_.size() && warps_[idx].ready) {
                    selected = &warps_[idx];
                    rr_idx = (idx + 1) % num_warps_;
                    break;
                }
            }
            break;
        }
    }

    if (selected) {
        issue_counts_[selected->warp_id]++;
        total_issues_++;
    }
    return selected;
}

void WarpScheduler::tick(uint64_t /*current_cycle*/) {
    if (policy_ == SchedulerPolicy::TWO_LEVEL) {
        two_level_refill_fetch_group();
    }
}

// ---------------------------------------------------------------------------
// GTO implementation
// ---------------------------------------------------------------------------

/*
 * Greedy-Then-Oldest (GTO):
 *   1. If the currently preferred (greedy) warp is still ready, keep issuing
 *      from it (maximises temporal locality for registers and L1 cache).
 *   2. When the greedy warp stalls or completes, fall back to the OLDEST
 *      ready warp (the one that became ready earliest, i.e. has the smallest
 *      birth_cycle).  Oldest-first is preferable to longest-waiting because
 *      it respects program order and is deterministic.
 */
Warp* WarpScheduler::gto_select(uint64_t /*current_cycle*/) {
    // Phase 1: greedy — keep the current warp if it is still ready.
    if (greedy_warp_id_ >= 0) {
        uint32_t gid = static_cast<uint32_t>(greedy_warp_id_);
        if (active_[gid] && !stalled_[gid] && gid < warps_.size() && warps_[gid].ready) {
            greedy_streak_++;
            return &warps_[gid];
        }
        // Current greedy warp stalled – fall through to oldest selection.
        greedy_warp_id_ = -1;
        greedy_streak_  = 0;
    }

    // Phase 2: oldest ready warp becomes the new greedy warp.
    Warp* oldest = oldest_ready_warp();
    if (oldest) {
        greedy_warp_id_ = static_cast<int32_t>(oldest->warp_id);
        greedy_streak_  = 1;
    }
    return oldest;
}

Warp* WarpScheduler::oldest_ready_warp() {
    Warp* result = nullptr;
    for (uint32_t i = 0; i < num_warps_; ++i) {
        if (!active_[i] || stalled_[i] || i >= warps_.size() || !warps_[i].ready)
            continue;
        if (!result || warps_[i].birth_cycle < result->birth_cycle) {
            result = &warps_[i];
        }
    }
    return result;
}

// ---------------------------------------------------------------------------
// Two-Level implementation
// ---------------------------------------------------------------------------

/*
 * Two-Level scheduler (Narasiman et al., MICRO 2011):
 *   - Inner level (fetch_group_): a small subset of warps scheduled by GTO.
 *   - Outer level (pending_group_): the remainder, fed in oldest-first order.
 *
 * When the fetch group is exhausted (all stalled or completed) the next batch
 * is moved from pending_group_ into fetch_group_.
 */
void WarpScheduler::two_level_refill_fetch_group() {
    // Remove completed / already-in-fetch-group warps from pending.
    auto is_inactive = [&](uint32_t id) { return !active_[id]; };
    pending_group_.erase(
        std::remove_if(pending_group_.begin(), pending_group_.end(), is_inactive),
        pending_group_.end());
    fetch_group_.erase(
        std::remove_if(fetch_group_.begin(), fetch_group_.end(), is_inactive),
        fetch_group_.end());

    // Refill if fetch group has room.
    while (fetch_group_.size() < fetch_group_size_ && !pending_group_.empty()) {
        fetch_group_.push_back(pending_group_.front());
        pending_group_.pop_front();
    }
}

Warp* WarpScheduler::two_level_select(uint64_t current_cycle) {
    // Apply GTO within the current fetch group.
    if (greedy_warp_id_ >= 0) {
        uint32_t gid = static_cast<uint32_t>(greedy_warp_id_);
        bool in_fetch = std::find(fetch_group_.begin(), fetch_group_.end(), gid)
                        != fetch_group_.end();
        if (in_fetch && active_[gid] && !stalled_[gid] &&
            gid < warps_.size() && warps_[gid].ready) {
            greedy_streak_++;
            return &warps_[gid];
        }
        greedy_warp_id_ = -1;
        greedy_streak_  = 0;
    }

    // Find oldest ready warp within the fetch group.
    Warp* result = nullptr;
    for (uint32_t id : fetch_group_) {
        if (!active_[id] || stalled_[id] || id >= warps_.size() || !warps_[id].ready)
            continue;
        if (!result || warps_[id].birth_cycle < result->birth_cycle) {
            result = &warps_[id];
        }
    }

    // If the fetch group has no ready warps, demote its stalled warps back to
    // the pending group so that pending warps can be promoted.  This matches
    // the Narasiman et al. two-level policy: when the inner group is fully
    // stalled, replenish from the outer group.
    if (!result) {
        auto it = fetch_group_.begin();
        while (it != fetch_group_.end()) {
            if (!active_[*it] || stalled_[*it]) {
                if (active_[*it] && stalled_[*it]) {
                    pending_group_.push_back(*it);
                }
                it = fetch_group_.erase(it);
            } else {
                ++it;
            }
        }
        two_level_refill_fetch_group();
        // Re-check after refill.
        for (uint32_t id : fetch_group_) {
            if (!active_[id] || stalled_[id] || id >= warps_.size() || !warps_[id].ready)
                continue;
            if (!result || warps_[id].birth_cycle < result->birth_cycle) {
                result = &warps_[id];
            }
        }
    }

    if (result) {
        greedy_warp_id_ = static_cast<int32_t>(result->warp_id);
        greedy_streak_  = 1;
        (void)current_cycle;
    }
    return result;
}

// ---------------------------------------------------------------------------
// Diagnostics
// ---------------------------------------------------------------------------

uint32_t WarpScheduler::num_ready_warps() const {
    uint32_t count = 0;
    for (uint32_t i = 0; i < num_warps_; ++i) {
        if (active_[i] && !stalled_[i] && i < warps_.size() && warps_[i].ready)
            ++count;
    }
    return count;
}

uint32_t WarpScheduler::num_stalled_warps() const {
    uint32_t count = 0;
    for (uint32_t i = 0; i < num_warps_; ++i) {
        if (active_[i] && stalled_[i])
            ++count;
    }
    return count;
}

uint64_t WarpScheduler::get_issue_count(uint32_t warp_id) const {
    if (warp_id >= num_warps_) return 0;
    return issue_counts_[warp_id];
}

std::string WarpScheduler::get_stats_summary() const {
    std::ostringstream oss;
    oss << "WarpScheduler stats:\n"
        << "  Policy        : ";
    switch (policy_) {
        case SchedulerPolicy::GTO:              oss << "GTO\n"; break;
        case SchedulerPolicy::TWO_LEVEL:        oss << "Two-Level\n"; break;
        case SchedulerPolicy::ROUND_ROBIN:      oss << "Round-Robin\n"; break;
        case SchedulerPolicy::LOOSE_ROUND_ROBIN: oss << "Loose-Round-Robin\n"; break;
    }
    oss << "  Total issues  : " << total_issues_ << "\n"
        << "  Total stalls  : " << total_stalls_ << "\n"
        << "  Ready warps   : " << num_ready_warps() << "\n"
        << "  Stalled warps : " << num_stalled_warps() << "\n"
        << "  Greedy warp   : " << greedy_warp_id_ << " (streak=" << greedy_streak_ << ")\n";
    return oss.str();
}

} // namespace gpu_sim
