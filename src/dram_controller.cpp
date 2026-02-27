#include "dram_controller.h"
#include "performance_monitor.h"
#include <algorithm>
#include <cmath>
#include <limits>

namespace gpu_sim {

DRAMController::DRAMController(const DRAMTimingParams& params)
    : params_(params),
      total_read_latency_(0),
      total_write_latency_(0),
      active_bank_cycles_(0) {

    // Initialize bank states: [channel][bank]
    bank_states_.resize(params_.num_channels);
    for (auto& ch : bank_states_) {
        ch.resize(params_.num_banks_per_channel);
        for (uint32_t b = 0; b < params_.num_banks_per_channel; ++b) {
            ch[b].bank_id = b;
        }
    }

    stats_ = DRAMStats{};
}

void DRAMController::initialize(std::shared_ptr<PerformanceMonitor> perf_monitor) {
    perf_monitor_ = perf_monitor;
    if (perf_monitor_) {
        perf_monitor_->set_counter("dram_channels", params_.num_channels);
        perf_monitor_->set_counter("dram_banks_per_channel", params_.num_banks_per_channel);
    }
}

// ---------------------------------------------------------------------------
// Address mapping (row-interleaved across channels and banks)
// ---------------------------------------------------------------------------
void DRAMController::decode_address(uint64_t address, uint32_t& channel,
                                     uint32_t& bank, uint32_t& row,
                                     uint32_t& col) const {
    uint64_t cache_line = address / params_.row_buffer_size;

    col     = address % params_.row_buffer_size;
    channel = cache_line % params_.num_channels;
    bank    = (cache_line / params_.num_channels) % params_.num_banks_per_channel;
    row     = static_cast<uint32_t>(
                  (cache_line / params_.num_channels / params_.num_banks_per_channel)
                  % params_.num_rows_per_bank);
}

// ---------------------------------------------------------------------------
// Compute access latency based on bank state
// ---------------------------------------------------------------------------
uint64_t DRAMController::compute_access_latency(const DRAMRequest& req,
                                                  uint64_t current_cycle) const {
    uint32_t ch = req.channel_id;
    uint32_t bk = req.bank_id;
    const auto& bank = bank_states_[ch][bk];

    uint64_t latency = 0;

    if (!bank.row_open) {
        // Row buffer empty — need ACT + CAS
        latency = params_.tRCD + params_.tCL;
        stats_.row_buffer_misses++;
    } else if (bank.open_row == req.row_id) {
        // Row buffer hit — CAS only
        latency = params_.tCL;
        stats_.row_buffer_hits++;
    } else {
        // Row buffer conflict — PRE + ACT + CAS
        latency = params_.tRP + params_.tRCD + params_.tCL;
        stats_.row_buffer_conflicts++;
    }

    // Ensure timing constraints relative to the last access on this bank
    if (bank.last_access > 0 && current_cycle < bank.last_access + params_.tCCD) {
        latency += (bank.last_access + params_.tCCD) - current_cycle;
    }

    return latency;
}

// ---------------------------------------------------------------------------
// Enqueue a request
// ---------------------------------------------------------------------------
uint64_t DRAMController::enqueue_request(uint64_t address, DRAMRequestType type,
                                          uint64_t current_cycle, size_t size) {
    DRAMRequest req(address, type, current_cycle, size);

    // Decode address into channel/bank/row/col
    uint32_t col_unused;
    decode_address(address, req.channel_id, req.bank_id, req.row_id, col_unused);

    // Compute completion time
    uint64_t latency = compute_access_latency(req, current_cycle);
    req.completion_cycle = current_cycle + latency;

    // Update bank state
    auto& bank = bank_states_[req.channel_id][req.bank_id];
    bank.open_row = req.row_id;
    bank.row_open = true;
    bank.last_activate = current_cycle;
    bank.last_access = req.completion_cycle;

    // Track statistics
    if (type == DRAMRequestType::READ) {
        stats_.total_reads++;
        total_read_latency_ += latency;
    } else {
        stats_.total_writes++;
        total_write_latency_ += latency;
    }

    request_queue_.push_back(req);

    if (perf_monitor_) {
        perf_monitor_->increment_counter("dram_requests");
    }

    return req.completion_cycle;
}

void DRAMController::tick(uint64_t current_cycle) {
    stats_.total_cycles = current_cycle;

    // Move completed requests out of the queue
    auto it = std::remove_if(request_queue_.begin(), request_queue_.end(),
        [current_cycle, this](const DRAMRequest& req) {
            if (req.completion_cycle <= current_cycle) {
                completed_requests_.push_back(req);
                return true;
            }
            return false;
        });
    request_queue_.erase(it, request_queue_.end());
}

bool DRAMController::is_idle() const {
    return request_queue_.empty();
}

// ---------------------------------------------------------------------------
// FR-FCFS scheduling (selects row-buffer-hit requests first)
// ---------------------------------------------------------------------------
size_t DRAMController::select_next_request(uint64_t /*current_cycle*/) {
    size_t best_idx = 0;
    bool found_hit = false;
    uint64_t oldest_arrival = std::numeric_limits<uint64_t>::max();

    for (size_t i = 0; i < request_queue_.size(); ++i) {
        const auto& req = request_queue_[i];
        const auto& bank = bank_states_[req.channel_id][req.bank_id];

        bool is_hit = bank.row_open && bank.open_row == req.row_id;

        if (is_hit && !found_hit) {
            // First hit found — prefer it
            found_hit = true;
            best_idx = i;
            oldest_arrival = req.arrival_cycle;
        } else if (is_hit && found_hit && req.arrival_cycle < oldest_arrival) {
            // Among hits, prefer oldest
            best_idx = i;
            oldest_arrival = req.arrival_cycle;
        } else if (!found_hit && req.arrival_cycle < oldest_arrival) {
            // No hits yet — FCFS
            best_idx = i;
            oldest_arrival = req.arrival_cycle;
        }
    }

    return best_idx;
}

DRAMController::DRAMStats DRAMController::get_statistics() const {
    DRAMStats s = stats_;
    if (s.total_reads > 0) {
        s.avg_read_latency = static_cast<double>(total_read_latency_) / s.total_reads;
    }
    if (s.total_writes > 0) {
        s.avg_write_latency = static_cast<double>(total_write_latency_) / s.total_writes;
    }

    // Bandwidth utilization estimate
    uint64_t total_bytes = (s.total_reads + s.total_writes) * params_.row_buffer_size;
    double peak_bw_bytes_per_cycle =
        (params_.bus_width / 8.0) * params_.burst_length * params_.num_channels;
    if (s.total_cycles > 0 && peak_bw_bytes_per_cycle > 0) {
        double peak_total_bytes = peak_bw_bytes_per_cycle * s.total_cycles;
        s.bandwidth_utilization = static_cast<double>(total_bytes) / peak_total_bytes;
        s.bandwidth_utilization = std::min(1.0, s.bandwidth_utilization);
    }

    // Bank-level parallelism estimate
    uint64_t total_requests = s.total_reads + s.total_writes;
    if (total_requests > 0 && params_.num_banks_per_channel > 0) {
        s.bank_level_parallelism = std::min(
            static_cast<double>(params_.num_banks_per_channel),
            static_cast<double>(total_requests) /
                std::max(1.0, static_cast<double>(s.total_cycles) / params_.tCL));
    }

    return s;
}

void DRAMController::reset_statistics() {
    stats_ = DRAMStats{};
    total_read_latency_ = 0;
    total_write_latency_ = 0;
    active_bank_cycles_ = 0;
}

std::string DRAMController::get_config_name() const {
    if (params_.num_channels == 8 && params_.bus_width == 128) {
        return "HBM2";
    } else if (params_.bus_width == 32 && params_.burst_length == 16) {
        return "GDDR6";
    }
    return "Custom";
}

} // namespace gpu_sim
