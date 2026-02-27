#pragma once

#include <vector>
#include <queue>
#include <memory>
#include <cstdint>
#include <string>

namespace gpu_sim {

class PerformanceMonitor;

/**
 * DRAM request type
 */
enum class DRAMRequestType {
    READ,
    WRITE
};

/**
 * A single DRAM request queued in the memory controller
 */
struct DRAMRequest {
    uint64_t address;
    DRAMRequestType type;
    uint64_t arrival_cycle;
    uint64_t completion_cycle;
    uint32_t bank_id;
    uint32_t row_id;
    uint32_t channel_id;
    size_t   size;

    DRAMRequest(uint64_t addr, DRAMRequestType t, uint64_t cycle, size_t sz)
        : address(addr), type(t), arrival_cycle(cycle), completion_cycle(0),
          bank_id(0), row_id(0), channel_id(0), size(sz) {}
};

/**
 * State of a single DRAM bank
 */
struct BankState {
    uint32_t bank_id;
    uint32_t open_row;       // Currently open row (UINT32_MAX = no open row)
    bool     row_open;
    uint64_t last_activate;  // Cycle of last ACT command
    uint64_t last_precharge; // Cycle of last PRE command
    uint64_t last_access;    // Cycle of last RD/WR command

    BankState()
        : bank_id(0), open_row(0), row_open(false),
          last_activate(0), last_precharge(0), last_access(0) {}
};

/**
 * DRAM timing parameters — models HBM2 or GDDR6 timing characteristics
 */
struct DRAMTimingParams {
    // Core timing (in cycles at the memory clock)
    uint32_t tCL;    // CAS Latency
    uint32_t tRCD;   // RAS-to-CAS Delay
    uint32_t tRP;    // Row Precharge time
    uint32_t tRAS;   // Row Active time (ACT to PRE)
    uint32_t tRC;    // Row Cycle time (ACT to ACT same bank)
    uint32_t tCCD;   // Column-to-Column Delay
    uint32_t tRTP;   // Read-to-Precharge
    uint32_t tWR;    // Write Recovery time
    uint32_t tWTR;   // Write-to-Read turnaround
    uint32_t tRRD;   // Row-to-Row Delay (ACT to ACT, different bank)
    uint32_t tFAW;   // Four-Activation Window

    // Configuration
    uint32_t num_channels;
    uint32_t num_banks_per_channel;
    uint32_t num_rows_per_bank;
    uint32_t row_buffer_size;     // Bytes
    uint32_t bus_width;           // Bits
    uint32_t burst_length;
    double   clock_ghz;           // Memory clock in GHz

    // Default HBM2 timing profile
    static DRAMTimingParams hbm2_default() {
        return {
            /* tCL  */ 14,
            /* tRCD */ 14,
            /* tRP  */ 14,
            /* tRAS */ 34,
            /* tRC  */ 48,
            /* tCCD */ 2,
            /* tRTP */ 5,
            /* tWR  */ 16,
            /* tWTR */ 8,
            /* tRRD */ 4,
            /* tFAW */ 16,
            /* channels */ 8,
            /* banks/ch */ 16,
            /* rows/bank */ 16384,
            /* row_buf   */ 2048,
            /* bus_width  */ 128,
            /* burst_len  */ 4,
            /* clock_ghz  */ 1.0
        };
    }

    // Default GDDR6 timing profile
    static DRAMTimingParams gddr6_default() {
        return {
            /* tCL  */ 20,
            /* tRCD */ 20,
            /* tRP  */ 20,
            /* tRAS */ 46,
            /* tRC  */ 66,
            /* tCCD */ 4,
            /* tRTP */ 12,
            /* tWR  */ 20,
            /* tWTR */ 10,
            /* tRRD */ 8,
            /* tFAW */ 32,
            /* channels */ 16,
            /* banks/ch */ 16,
            /* rows/bank */ 32768,
            /* row_buf   */ 4096,
            /* bus_width  */ 32,
            /* burst_len  */ 16,
            /* clock_ghz  */ 1.75
        };
    }
};

/**
 * DRAM Memory Controller with accurate timing model
 *
 * Models:
 *  - Row Buffer hits / misses / conflicts
 *  - Bank-level parallelism: requests to different banks proceed concurrently
 *  - FR-FCFS (First-Ready, First-Come-First-Served) scheduling
 *  - Channel-level interleaving
 *
 * Configurable for HBM2, GDDR6, or custom timing.
 */
class DRAMController {
public:
    explicit DRAMController(const DRAMTimingParams& params = DRAMTimingParams::hbm2_default());
    ~DRAMController() = default;

    // Initialization
    void initialize(std::shared_ptr<PerformanceMonitor> perf_monitor);

    // Enqueue a request and return estimated completion cycle
    uint64_t enqueue_request(uint64_t address, DRAMRequestType type,
                             uint64_t current_cycle, size_t size = 64);

    // Advance the controller by one cycle
    void tick(uint64_t current_cycle);

    // Check if all requests have completed
    bool is_idle() const;

    // Statistics
    struct DRAMStats {
        uint64_t total_reads;
        uint64_t total_writes;
        uint64_t row_buffer_hits;
        uint64_t row_buffer_misses;
        uint64_t row_buffer_conflicts;  // Miss on open row → precharge needed
        uint64_t total_cycles;
        double   avg_read_latency;
        double   avg_write_latency;
        double   bandwidth_utilization; // Fraction of peak bandwidth used
        double   bank_level_parallelism; // Average concurrent bank accesses
    };

    DRAMStats get_statistics() const;
    void reset_statistics();

    // Configuration access
    const DRAMTimingParams& get_timing_params() const { return params_; }
    std::string get_config_name() const;

private:
    // Address mapping: address → (channel, bank, row, column)
    void decode_address(uint64_t address, uint32_t& channel, uint32_t& bank,
                        uint32_t& row, uint32_t& col) const;

    // Compute latency for a request based on bank state
    uint64_t compute_access_latency(const DRAMRequest& req, uint64_t current_cycle) const;

    // FR-FCFS scheduling
    size_t select_next_request(uint64_t current_cycle);

    DRAMTimingParams params_;
    std::vector<std::vector<BankState>> bank_states_; // [channel][bank]
    std::vector<DRAMRequest> request_queue_;
    std::vector<DRAMRequest> completed_requests_;

    // Statistics tracking
    mutable DRAMStats stats_;
    uint64_t total_read_latency_;
    uint64_t total_write_latency_;
    uint64_t active_bank_cycles_;

    std::shared_ptr<PerformanceMonitor> perf_monitor_;
};

} // namespace gpu_sim
