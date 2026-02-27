#pragma once

#include <vector>
#include <deque>
#include <cstdint>
#include <string>

namespace gpu_sim {

// ---------------------------------------------------------------------------
// DRAM timing parameters
// ---------------------------------------------------------------------------

/**
 * DRAM timing parameters for a single rank.
 * Values are in DRAM clock cycles unless noted.
 *
 * Defaults model HBM2 (2 Gbps per pin) running at 1 GHz DRAM clock.
 * A GDDR6 preset is also provided.
 */
struct DRAMTimingParams {
    // Core timing
    uint32_t tCL   = 14;   // CAS latency (READ command → first data)
    uint32_t tRCD  = 14;   // RAS-to-CAS delay (ACT → READ/WRITE allowed)
    uint32_t tRP   = 14;   // Row precharge time (PRE → next ACT allowed)
    uint32_t tRAS  = 33;   // Row active time (minimum ACT → PRE)
    uint32_t tRC   = 47;   // Row cycle time = tRAS + tRP
    uint32_t tWR   = 15;   // Write recovery (last DQ → PRE allowed)
    uint32_t tRTP  = 7;    // Read-to-precharge (last read burst → PRE)
    uint32_t tBURST= 4;    // Burst length / data rate (4 cycles for BL8)

    // Bank / rank timing
    uint32_t tRRD  = 5;    // ACT-to-ACT (different banks, same rank)
    uint32_t tFAW  = 20;   // Four-activation window (max 4 ACTs in this window)
    uint32_t tCCD  = 4;    // CAS-to-CAS delay (same bank group)
    uint32_t tWTR  = 7;    // Write-to-read turnaround (same rank)
    uint32_t tRTW  = 2;    // Read-to-write turnaround (bus turnaround)

    // Refresh
    uint32_t tRFC  = 260;  // Refresh cycle time
    uint32_t tREFI = 3900; // Refresh interval (64 ms / 16K rows ≈ 3.9 µs @ 1 GHz)

    /** Return a preset for HBM2 at 1 GHz DRAM clock. */
    static DRAMTimingParams hbm2();

    /** Return a preset for GDDR6 at 1750 MHz DRAM clock. */
    static DRAMTimingParams gddr6();
};

// ---------------------------------------------------------------------------
// Bank state
// ---------------------------------------------------------------------------

enum class BankState {
    IDLE,           // no row open
    ACTIVATING,     // ACT issued, waiting tRCD
    ACTIVE,         // row open (row buffer loaded)
    PRECHARGING,    // PRE issued, waiting tRP
    REFRESHING,     // under refresh
};

struct DRAMBank {
    BankState state           = BankState::IDLE;
    uint32_t  open_row        = UINT32_MAX;   // currently open row (UINT32_MAX = none)
    uint64_t  next_act_cycle  = 0;            // earliest cycle ACT is allowed
    uint64_t  next_pre_cycle  = 0;            // earliest cycle PRE is allowed
    uint64_t  next_read_cycle = 0;            // earliest cycle READ is allowed
    uint64_t  next_write_cycle= 0;            // earliest cycle WRITE is allowed
};

// ---------------------------------------------------------------------------
// DRAM request
// ---------------------------------------------------------------------------

enum class DRAMCommandType { READ, WRITE };

struct DRAMRequest {
    uint64_t        address;
    DRAMCommandType type;
    uint64_t        arrival_cycle;
    uint64_t        completion_cycle = 0;   // filled in when serviced
    bool            completed        = false;
};

// ---------------------------------------------------------------------------
// DRAMTimingModel
// ---------------------------------------------------------------------------

/**
 * Cycle-accurate DRAM timing model with:
 *   - Per-bank row-buffer management (open-page policy by default).
 *   - Row-buffer hit / miss / conflict classification.
 *   - Four-activation window (tFAW) enforcement.
 *   - Periodic refresh (auto-refresh model).
 *   - Bank-level parallelism: independent banks can be serviced in parallel.
 *
 * The model operates on a single DRAM channel with configurable ranks and
 * banks per rank (default: 1 rank, 16 banks — matching HBM2 pseudo-channel).
 */
class DRAMTimingModel {
public:
    /**
     * @param params         Timing parameters.
     * @param num_ranks      Number of ranks on the channel.
     * @param banks_per_rank Number of banks per rank.
     * @param rows_per_bank  Number of rows per bank (determines row address).
     */
    explicit DRAMTimingModel(const DRAMTimingParams& params = DRAMTimingParams{},
                             uint32_t num_ranks      = 1,
                             uint32_t banks_per_rank = 16,
                             uint32_t rows_per_bank  = 65536);
    ~DRAMTimingModel() = default;

    // ---- Request interface ------------------------------------------------

    /**
     * Enqueue a DRAM request.  Returns a request ID (index into pending list).
     */
    uint64_t enqueue(uint64_t address, DRAMCommandType type,
                     uint64_t current_cycle);

    /**
     * Advance the DRAM timing model by one cycle.
     * Issues commands (ACT / PRE / READ / WRITE) respecting all timing
     * constraints.  Completes requests and updates statistics.
     */
    void tick(uint64_t current_cycle);

    /**
     * Returns true if the request with the given ID has completed.
     */
    bool is_complete(uint64_t request_id) const;

    /**
     * Returns the completion cycle for a completed request.
     * Returns 0 if not yet completed.
     */
    uint64_t completion_cycle(uint64_t request_id) const;

    // ---- Statistics -------------------------------------------------------

    struct DRAMStats {
        uint64_t row_buffer_hits;       // request hits the open row
        uint64_t row_buffer_misses;     // request targets a different (closed) row
        uint64_t row_buffer_conflicts;  // request targets a different row that was open
        uint64_t total_requests;
        uint64_t total_latency_cycles;  // sum of (completion - arrival) for all done
        double   avg_latency_cycles;
        uint64_t bank_parallelism_events; // cycles where >1 bank was active
        uint64_t refresh_stalls;        // requests delayed by refresh
    };

    DRAMStats get_stats() const;
    void      reset_stats();

    /** Human-readable stats string. */
    std::string get_stats_summary() const;

    // ---- Configuration accessors ------------------------------------------
    uint32_t num_banks() const { return num_ranks_ * banks_per_rank_; }

private:
    // ---- Address mapping --------------------------------------------------
    uint32_t rank_index(uint64_t address)  const;
    uint32_t bank_index(uint64_t address)  const;
    uint32_t row_index(uint64_t address)   const;

    // ---- Command scheduling -----------------------------------------------
    void try_service_request(DRAMRequest& req, uint64_t cycle);
    uint64_t activate_latency(uint32_t rank, uint32_t bank,
                              uint32_t row, uint64_t cycle);
    bool     faw_ok(uint32_t rank, uint64_t cycle) const;
    void     record_activation(uint32_t rank, uint64_t cycle);

    // ---- State ------------------------------------------------------------
    DRAMTimingParams            params_;
    uint32_t                    num_ranks_;
    uint32_t                    banks_per_rank_;
    uint32_t                    rows_per_bank_;

    // banks_[rank * banks_per_rank_ + bank]
    std::vector<DRAMBank>       banks_;

    // Four-activation window tracking per rank: last 4 activation cycles.
    std::vector<std::deque<uint64_t>> activation_windows_;

    // Refresh tracking per rank
    std::vector<uint64_t>       next_refresh_cycle_;

    // Request pool
    std::vector<DRAMRequest>    requests_;
    std::deque<uint64_t>        pending_queue_;     // request IDs not yet complete

    // Statistics
    mutable DRAMStats           stats_{};
};

} // namespace gpu_sim
