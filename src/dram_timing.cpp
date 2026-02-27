#include "dram_timing.h"
#include <algorithm>
#include <sstream>
#include <numeric>

namespace gpu_sim {

// ---------------------------------------------------------------------------
// DRAMTimingParams presets
// ---------------------------------------------------------------------------

DRAMTimingParams DRAMTimingParams::hbm2() {
    DRAMTimingParams p;
    // HBM2 1 GHz effective DRAM clock (from JEDEC JESD235B)
    p.tCL   = 14; p.tRCD  = 14; p.tRP   = 14; p.tRAS  = 33;
    p.tRC   = 47; p.tWR   = 15; p.tRTP  = 7;  p.tBURST= 4;
    p.tRRD  = 4;  p.tFAW  = 16; p.tCCD  = 4;  p.tWTR  = 8;
    p.tRFC  = 260; p.tREFI = 3900;
    return p;
}

DRAMTimingParams DRAMTimingParams::gddr6() {
    DRAMTimingParams p;
    // GDDR6 1750 MHz DRAM clock (SK Hynix H5AN8G)
    p.tCL   = 16; p.tRCD  = 16; p.tRP   = 16; p.tRAS  = 38;
    p.tRC   = 54; p.tWR   = 18; p.tRTP  = 8;  p.tBURST= 4;
    p.tRRD  = 6;  p.tFAW  = 24; p.tCCD  = 4;  p.tWTR  = 9;
    p.tRFC  = 350; p.tREFI = 4550;
    return p;
}

// ---------------------------------------------------------------------------
// DRAMTimingModel
// ---------------------------------------------------------------------------

DRAMTimingModel::DRAMTimingModel(const DRAMTimingParams& params,
                                 uint32_t num_ranks,
                                 uint32_t banks_per_rank,
                                 uint32_t rows_per_bank)
    : params_(params),
      num_ranks_(num_ranks),
      banks_per_rank_(banks_per_rank),
      rows_per_bank_(rows_per_bank)
{
    uint32_t total_banks = num_ranks_ * banks_per_rank_;
    banks_.resize(total_banks);
    activation_windows_.resize(num_ranks_);
    next_refresh_cycle_.assign(num_ranks_, params_.tREFI);
    reset_stats();
}

// ---------------------------------------------------------------------------
// Address mapping
// Interleaved at cache-line (64B) granularity to maximise bank parallelism:
//   bits [5:0]   → byte offset within cache line (64B)
//   bits [8:6]   → bank index (log2(banks_per_rank) bits)
//   bits [9]     → rank index (if >1 rank)
//   bits [...]   → row index
// ---------------------------------------------------------------------------

uint32_t DRAMTimingModel::rank_index(uint64_t address) const {
    if (num_ranks_ <= 1) return 0;
    return static_cast<uint32_t>((address >> 6) >> __builtin_ctz(banks_per_rank_))
           % num_ranks_;
}

uint32_t DRAMTimingModel::bank_index(uint64_t address) const {
    return static_cast<uint32_t>((address >> 6) % banks_per_rank_);
}

uint32_t DRAMTimingModel::row_index(uint64_t address) const {
    uint32_t bank_bits = 0;
    uint32_t bp = banks_per_rank_;
    while (bp > 1) { bank_bits++; bp >>= 1; }
    uint32_t rank_bits = 0;
    uint32_t rp = num_ranks_;
    while (rp > 1) { rank_bits++; rp >>= 1; }
    return static_cast<uint32_t>(
        (address >> (6 + bank_bits + rank_bits)) % rows_per_bank_);
}

// ---------------------------------------------------------------------------
// Four-activation window (tFAW)
// ---------------------------------------------------------------------------

bool DRAMTimingModel::faw_ok(uint32_t rank, uint64_t cycle) const {
    const auto& win = activation_windows_[rank];
    if (win.size() < 4) return true;
    // The 4th oldest activation must be at least tFAW cycles ago.
    return (cycle - win.front()) >= params_.tFAW;
}

void DRAMTimingModel::record_activation(uint32_t rank, uint64_t cycle) {
    auto& win = activation_windows_[rank];
    win.push_back(cycle);
    if (win.size() > 4) win.pop_front();
}

// ---------------------------------------------------------------------------
// Enqueue
// ---------------------------------------------------------------------------

uint64_t DRAMTimingModel::enqueue(uint64_t address, DRAMCommandType type,
                                  uint64_t current_cycle) {
    uint64_t id = requests_.size();
    requests_.push_back({address, type, current_cycle});
    pending_queue_.push_back(id);
    stats_.total_requests++;
    return id;
}

// ---------------------------------------------------------------------------
// tick – advance by one cycle
// ---------------------------------------------------------------------------

void DRAMTimingModel::tick(uint64_t current_cycle) {
    // Issue refresh if needed (per rank).
    for (uint32_t r = 0; r < num_ranks_; ++r) {
        if (current_cycle >= next_refresh_cycle_[r]) {
            // Simplified refresh: stall any bank in this rank that is active.
            for (uint32_t b = 0; b < banks_per_rank_; ++b) {
                DRAMBank& bank = banks_[r * banks_per_rank_ + b];
                if (bank.state == BankState::ACTIVE) {
                    bank.state    = BankState::REFRESHING;
                    bank.open_row = UINT32_MAX;
                    bank.next_act_cycle = current_cycle + params_.tRFC;
                    stats_.refresh_stalls++;
                }
            }
            next_refresh_cycle_[r] = current_cycle + params_.tREFI;
        }
    }

    // Count simultaneously active banks for parallelism metric.
    uint32_t active_count = 0;
    for (auto& bank : banks_) {
        if (bank.state == BankState::ACTIVE) active_count++;
    }
    if (active_count > 1) stats_.bank_parallelism_events++;

    // Service pending requests in FIFO order (FR-FCFS-like, simplified).
    for (auto it = pending_queue_.begin(); it != pending_queue_.end(); ) {
        DRAMRequest& req = requests_[*it];
        try_service_request(req, current_cycle);
        if (req.completed) {
            stats_.total_latency_cycles +=
                req.completion_cycle - req.arrival_cycle;
            it = pending_queue_.erase(it);
        } else {
            ++it;
        }
    }
}

// ---------------------------------------------------------------------------
// Service one request: issue ACT → PRE (if needed) → READ/WRITE
// ---------------------------------------------------------------------------

void DRAMTimingModel::try_service_request(DRAMRequest& req, uint64_t cycle) {
    uint32_t rank = rank_index(req.address);
    uint32_t bank_idx = bank_index(req.address);
    uint32_t row  = row_index(req.address);
    DRAMBank& bank = banks_[rank * banks_per_rank_ + bank_idx];

    // Handle banks that are currently refreshing.
    if (bank.state == BankState::REFRESHING) {
        if (cycle < bank.next_act_cycle) return;
        bank.state = BankState::IDLE;
    }

    switch (bank.state) {

        case BankState::IDLE: {
            // Check tFAW constraint before activating.
            if (!faw_ok(rank, cycle)) return;
            if (cycle < bank.next_act_cycle) return;

            // Issue ACTIVATE.
            record_activation(rank, cycle);
            bank.state    = BankState::ACTIVE;
            bank.open_row = row;
            bank.next_read_cycle  = cycle + params_.tRCD;
            bank.next_write_cycle = cycle + params_.tRCD;
            bank.next_pre_cycle   = cycle + params_.tRAS;
            stats_.row_buffer_misses++;
            return;   // come back next cycle(s)
        }

        case BankState::ACTIVE: {
            if (bank.open_row == row) {
                // Row buffer hit!
                stats_.row_buffer_hits++;

                if (req.type == DRAMCommandType::READ) {
                    if (cycle < bank.next_read_cycle) return;
                    req.completion_cycle = cycle + params_.tCL + params_.tBURST;
                } else {
                    if (cycle < bank.next_write_cycle) return;
                    req.completion_cycle = cycle + params_.tBURST;
                }
                req.completed = true;

                // Update bank timing for next command.
                bank.next_read_cycle  = cycle + params_.tCCD;
                bank.next_write_cycle = cycle + params_.tCCD;
                bank.next_pre_cycle   = std::max(bank.next_pre_cycle,
                    (req.type == DRAMCommandType::READ)
                        ? cycle + params_.tRTP
                        : cycle + params_.tWR);

            } else {
                // Row buffer conflict: precharge then activate new row.
                stats_.row_buffer_conflicts++;
                if (cycle < bank.next_pre_cycle) return;

                // Issue PRECHARGE.
                bank.state    = BankState::IDLE;
                bank.open_row = UINT32_MAX;
                bank.next_act_cycle = cycle + params_.tRP;
            }
            return;
        }

        default:
            return;
    }
}

// ---------------------------------------------------------------------------
// Query
// ---------------------------------------------------------------------------

bool DRAMTimingModel::is_complete(uint64_t request_id) const {
    if (request_id >= requests_.size()) return false;
    return requests_[request_id].completed;
}

uint64_t DRAMTimingModel::completion_cycle(uint64_t request_id) const {
    if (request_id >= requests_.size()) return 0;
    return requests_[request_id].completion_cycle;
}

// ---------------------------------------------------------------------------
// Statistics
// ---------------------------------------------------------------------------

DRAMTimingModel::DRAMStats DRAMTimingModel::get_stats() const {
    DRAMStats s = stats_;
    s.avg_latency_cycles = s.total_requests > 0
        ? static_cast<double>(s.total_latency_cycles) / s.total_requests
        : 0.0;
    return s;
}

void DRAMTimingModel::reset_stats() {
    stats_ = DRAMStats{};
}

std::string DRAMTimingModel::get_stats_summary() const {
    auto s = get_stats();
    std::ostringstream oss;
    oss << "DRAMTimingModel stats:\n"
        << "  Total requests      : " << s.total_requests << "\n"
        << "  Row buffer hits     : " << s.row_buffer_hits << "\n"
        << "  Row buffer misses   : " << s.row_buffer_misses << "\n"
        << "  Row buffer conflicts: " << s.row_buffer_conflicts << "\n"
        << "  Avg latency (cycles): " << s.avg_latency_cycles << "\n"
        << "  Bank parallelism ev : " << s.bank_parallelism_events << "\n"
        << "  Refresh stalls      : " << s.refresh_stalls << "\n";
    return oss.str();
}

} // namespace gpu_sim
