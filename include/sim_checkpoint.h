#pragma once

#include <vector>
#include <string>
#include <memory>
#include <cstdint>
#include <unordered_map>
#include <fstream>

namespace gpu_sim {

class PerformanceMonitor;

/**
 * Represents a snapshot of the simulation state at a particular point
 */
struct SimulationCheckpoint {
    uint64_t checkpoint_id;
    uint64_t cycle;
    uint64_t instruction_count;
    std::string label;

    // Captured metrics at this point
    double   ipc;
    double   cache_hit_rate;
    uint64_t active_warps;
    double   memory_bandwidth_util;

    // Serialization
    std::string serialize() const;
    static SimulationCheckpoint deserialize(const std::string& data);
};

/**
 * A representative simulation interval identified by sampling analysis
 * (inspired by SimPoint methodology)
 */
struct SampleInterval {
    uint64_t start_cycle;
    uint64_t end_cycle;
    uint64_t instruction_count;
    double   weight;         // Fraction of total execution this interval represents
    uint32_t cluster_id;     // Cluster assignment (SimPoint-style)
    bool     is_representative;

    uint64_t length() const { return end_cycle - start_cycle; }
};

/**
 * Checkpointing and Sampling Engine
 *
 * Provides two complementary strategies for reducing simulation time:
 *
 * 1. Checkpointing: Save/restore full simulation state to skip past
 *    initialization or previously-simulated phases.
 *
 * 2. Sampling (SimPoint-style): Divide execution into fixed-size intervals,
 *    characterize each by a Basic Block Vector (BBV), cluster similar
 *    intervals, and simulate only one representative per cluster.
 *
 * Estimated speedup: 10-50× for large-scale CUDA/OpenCL traces while
 * maintaining <5% IPC error vs. full simulation.
 */
class SimCheckpointEngine {
public:
    explicit SimCheckpointEngine(uint64_t interval_size = 10000000); // 10M instructions
    ~SimCheckpointEngine() = default;

    // Initialization
    void initialize(std::shared_ptr<PerformanceMonitor> perf_monitor);

    // --- Checkpointing API ---

    // Create a checkpoint at the current simulation state
    SimulationCheckpoint create_checkpoint(uint64_t cycle,
                                           uint64_t instruction_count,
                                           const std::string& label = "");

    // Save checkpoint to file
    bool save_checkpoint(const SimulationCheckpoint& cp, const std::string& filepath) const;

    // Load checkpoint from file
    SimulationCheckpoint load_checkpoint(const std::string& filepath) const;

    // List all checkpoints
    const std::vector<SimulationCheckpoint>& get_checkpoints() const { return checkpoints_; }

    // --- Sampling / SimPoint API ---

    // Record a basic-block vector for the current interval
    void record_interval_bbv(uint64_t interval_id,
                             const std::vector<uint64_t>& bbv);

    // Run SimPoint-style clustering to identify representative intervals
    void compute_sample_points(uint32_t num_clusters = 10);

    // Get the representative sample intervals
    const std::vector<SampleInterval>& get_sample_intervals() const {
        return sample_intervals_;
    }

    // Check if a given cycle falls within a sample interval
    bool should_simulate_cycle(uint64_t cycle) const;

    // Estimate total metric from sampled results
    double estimate_total_ipc(const std::vector<double>& sampled_ipcs) const;

    // Statistics
    struct SamplingStats {
        uint64_t total_intervals;
        uint64_t sampled_intervals;
        double   estimated_speedup;
        double   coverage_fraction;   // Fraction of execution represented
    };

    SamplingStats get_statistics() const;

    // Configuration
    void set_interval_size(uint64_t size) { interval_size_ = size; }
    uint64_t get_interval_size() const { return interval_size_; }

private:
    // Simple k-means-style clustering for BBVs
    struct BBVEntry {
        uint64_t interval_id;
        std::vector<double> normalized_bbv;
    };

    void normalize_bbvs();
    double bbv_distance(const std::vector<double>& a, const std::vector<double>& b) const;
    uint32_t assign_cluster(const std::vector<double>& bbv,
                            const std::vector<std::vector<double>>& centroids) const;

    uint64_t interval_size_;
    std::vector<SimulationCheckpoint> checkpoints_;
    std::vector<BBVEntry> bbv_entries_;
    std::vector<SampleInterval> sample_intervals_;

    std::shared_ptr<PerformanceMonitor> perf_monitor_;
};

} // namespace gpu_sim
