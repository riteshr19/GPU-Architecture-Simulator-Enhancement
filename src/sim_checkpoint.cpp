#include "sim_checkpoint.h"
#include "performance_monitor.h"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <sstream>
#include <fstream>
#include <limits>

namespace gpu_sim {

// ---------------------------------------------------------------------------
// SimulationCheckpoint serialization
// ---------------------------------------------------------------------------
std::string SimulationCheckpoint::serialize() const {
    std::ostringstream oss;
    oss << checkpoint_id << ","
        << cycle << ","
        << instruction_count << ","
        << label << ","
        << ipc << ","
        << cache_hit_rate << ","
        << active_warps << ","
        << memory_bandwidth_util;
    return oss.str();
}

SimulationCheckpoint SimulationCheckpoint::deserialize(const std::string& data) {
    SimulationCheckpoint cp{};
    std::istringstream iss(data);
    std::string token;

    if (std::getline(iss, token, ',')) cp.checkpoint_id = std::stoull(token);
    if (std::getline(iss, token, ',')) cp.cycle = std::stoull(token);
    if (std::getline(iss, token, ',')) cp.instruction_count = std::stoull(token);
    if (std::getline(iss, token, ',')) cp.label = token;
    if (std::getline(iss, token, ',')) cp.ipc = std::stod(token);
    if (std::getline(iss, token, ',')) cp.cache_hit_rate = std::stod(token);
    if (std::getline(iss, token, ',')) cp.active_warps = std::stoull(token);
    if (std::getline(iss, token, ',')) cp.memory_bandwidth_util = std::stod(token);

    return cp;
}

// ---------------------------------------------------------------------------
// SimCheckpointEngine
// ---------------------------------------------------------------------------
SimCheckpointEngine::SimCheckpointEngine(uint64_t interval_size)
    : interval_size_(interval_size) {}

void SimCheckpointEngine::initialize(std::shared_ptr<PerformanceMonitor> perf_monitor) {
    perf_monitor_ = perf_monitor;
}

SimulationCheckpoint SimCheckpointEngine::create_checkpoint(
    uint64_t cycle, uint64_t instruction_count, const std::string& label) {

    SimulationCheckpoint cp{};
    cp.checkpoint_id = checkpoints_.size();
    cp.cycle = cycle;
    cp.instruction_count = instruction_count;
    cp.label = label.empty()
        ? "checkpoint_" + std::to_string(cp.checkpoint_id)
        : label;
    cp.ipc = (cycle > 0) ? static_cast<double>(instruction_count) / cycle : 0.0;
    cp.cache_hit_rate = 0.0;
    cp.active_warps = 0;
    cp.memory_bandwidth_util = 0.0;

    checkpoints_.push_back(cp);

    if (perf_monitor_) {
        perf_monitor_->increment_counter("checkpoints_created");
    }

    return cp;
}

bool SimCheckpointEngine::save_checkpoint(const SimulationCheckpoint& cp,
                                           const std::string& filepath) const {
    std::ofstream ofs(filepath);
    if (!ofs.is_open()) return false;
    ofs << cp.serialize() << "\n";
    return true;
}

SimulationCheckpoint SimCheckpointEngine::load_checkpoint(const std::string& filepath) const {
    std::ifstream ifs(filepath);
    std::string line;
    if (ifs.is_open() && std::getline(ifs, line)) {
        return SimulationCheckpoint::deserialize(line);
    }
    return SimulationCheckpoint{};
}

// ---------------------------------------------------------------------------
// BBV recording and SimPoint clustering
// ---------------------------------------------------------------------------
void SimCheckpointEngine::record_interval_bbv(uint64_t interval_id,
                                               const std::vector<uint64_t>& bbv) {
    BBVEntry entry;
    entry.interval_id = interval_id;
    entry.normalized_bbv.resize(bbv.size());

    // Normalize to unit vector
    double sum = 0.0;
    for (auto v : bbv) sum += static_cast<double>(v);
    if (sum > 0) {
        for (size_t i = 0; i < bbv.size(); ++i) {
            entry.normalized_bbv[i] = static_cast<double>(bbv[i]) / sum;
        }
    }

    bbv_entries_.push_back(std::move(entry));
}

double SimCheckpointEngine::bbv_distance(const std::vector<double>& a,
                                          const std::vector<double>& b) const {
    double dist = 0.0;
    size_t len = std::min(a.size(), b.size());
    for (size_t i = 0; i < len; ++i) {
        double diff = a[i] - b[i];
        dist += diff * diff;
    }
    return std::sqrt(dist);
}

uint32_t SimCheckpointEngine::assign_cluster(
    const std::vector<double>& bbv,
    const std::vector<std::vector<double>>& centroids) const {

    uint32_t best = 0;
    double best_dist = std::numeric_limits<double>::max();
    for (uint32_t c = 0; c < centroids.size(); ++c) {
        double d = bbv_distance(bbv, centroids[c]);
        if (d < best_dist) {
            best_dist = d;
            best = c;
        }
    }
    return best;
}

void SimCheckpointEngine::compute_sample_points(uint32_t num_clusters) {
    if (bbv_entries_.empty()) return;

    num_clusters = std::min(num_clusters, static_cast<uint32_t>(bbv_entries_.size()));

    // Initialize centroids (pick first K entries)
    size_t dim = bbv_entries_[0].normalized_bbv.size();
    std::vector<std::vector<double>> centroids(num_clusters);
    for (uint32_t c = 0; c < num_clusters; ++c) {
        centroids[c] = bbv_entries_[c % bbv_entries_.size()].normalized_bbv;
    }

    // Simple k-means (5 iterations)
    std::vector<uint32_t> assignments(bbv_entries_.size(), 0);

    for (int iter = 0; iter < 5; ++iter) {
        // Assignment step
        for (size_t i = 0; i < bbv_entries_.size(); ++i) {
            assignments[i] = assign_cluster(bbv_entries_[i].normalized_bbv, centroids);
        }

        // Update centroids
        for (uint32_t c = 0; c < num_clusters; ++c) {
            std::vector<double> new_centroid(dim, 0.0);
            uint32_t count = 0;
            for (size_t i = 0; i < bbv_entries_.size(); ++i) {
                if (assignments[i] == c) {
                    for (size_t d = 0; d < dim && d < bbv_entries_[i].normalized_bbv.size(); ++d) {
                        new_centroid[d] += bbv_entries_[i].normalized_bbv[d];
                    }
                    count++;
                }
            }
            if (count > 0) {
                for (auto& v : new_centroid) v /= count;
                centroids[c] = new_centroid;
            }
        }
    }

    // Select the representative (closest to centroid) from each cluster
    sample_intervals_.clear();
    for (uint32_t c = 0; c < num_clusters; ++c) {
        double best_dist = std::numeric_limits<double>::max();
        size_t best_idx = 0;
        uint32_t cluster_size = 0;

        for (size_t i = 0; i < bbv_entries_.size(); ++i) {
            if (assignments[i] == c) {
                cluster_size++;
                double d = bbv_distance(bbv_entries_[i].normalized_bbv, centroids[c]);
                if (d < best_dist) {
                    best_dist = d;
                    best_idx = i;
                }
            }
        }

        if (cluster_size == 0) continue;

        SampleInterval si;
        si.start_cycle = bbv_entries_[best_idx].interval_id * interval_size_;
        si.end_cycle = si.start_cycle + interval_size_;
        si.instruction_count = interval_size_; // Approximate
        si.weight = static_cast<double>(cluster_size) / bbv_entries_.size();
        si.cluster_id = c;
        si.is_representative = true;

        sample_intervals_.push_back(si);
    }

    if (perf_monitor_) {
        perf_monitor_->set_counter("simpoint_clusters", num_clusters);
        perf_monitor_->set_counter("simpoint_sample_intervals", sample_intervals_.size());
    }
}

bool SimCheckpointEngine::should_simulate_cycle(uint64_t cycle) const {
    if (sample_intervals_.empty()) return true; // No sampling configured

    for (const auto& si : sample_intervals_) {
        if (cycle >= si.start_cycle && cycle < si.end_cycle) {
            return true;
        }
    }
    return false;
}

double SimCheckpointEngine::estimate_total_ipc(const std::vector<double>& sampled_ipcs) const {
    if (sampled_ipcs.empty() || sample_intervals_.empty()) return 0.0;

    double weighted_ipc = 0.0;
    size_t count = std::min(sampled_ipcs.size(), sample_intervals_.size());

    for (size_t i = 0; i < count; ++i) {
        weighted_ipc += sampled_ipcs[i] * sample_intervals_[i].weight;
    }

    return weighted_ipc;
}

SimCheckpointEngine::SamplingStats SimCheckpointEngine::get_statistics() const {
    SamplingStats s{};
    s.total_intervals = bbv_entries_.size();
    s.sampled_intervals = sample_intervals_.size();

    if (s.total_intervals > 0 && s.sampled_intervals > 0) {
        s.estimated_speedup = static_cast<double>(s.total_intervals) / s.sampled_intervals;

        double total_weight = 0.0;
        for (const auto& si : sample_intervals_) {
            total_weight += si.weight;
        }
        s.coverage_fraction = total_weight;
    }

    return s;
}

} // namespace gpu_sim
