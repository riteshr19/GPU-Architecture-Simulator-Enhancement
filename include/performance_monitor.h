#pragma once

#include <chrono>
#include <vector>
#include <unordered_map>
#include <string>
#include <memory>
#include <cstdint>

namespace gpu_sim {

/**
 * Performance monitoring and profiling system
 */
class PerformanceMonitor {
public:
    PerformanceMonitor();
    ~PerformanceMonitor() = default;

    // Timing measurements
    void start_timer(const std::string& event);
    void end_timer(const std::string& event);
    double get_elapsed_time_ms(const std::string& event) const;
    
    // Counter management
    void increment_counter(const std::string& counter, uint64_t value = 1);
    void set_counter(const std::string& counter, uint64_t value);
    uint64_t get_counter(const std::string& counter) const;
    
    // Performance metrics
    void record_bandwidth_usage(const std::string& component, uint64_t bytes);
    void record_cache_access(const std::string& cache, bool hit);
    void record_frame_metrics(double frame_time_ms, uint32_t triangles, uint32_t fragments);
    
    // Reporting
    struct PerformanceReport {
        std::unordered_map<std::string, double> timing_data;
        std::unordered_map<std::string, uint64_t> counter_data;
        std::unordered_map<std::string, double> bandwidth_data;
        std::unordered_map<std::string, double> cache_hit_rates;
        
        double avg_frame_time_ms;
        double min_frame_time_ms;
        double max_frame_time_ms;
        uint64_t total_triangles;
        uint64_t total_fragments;
        
        // Efficiency metrics
        double memory_efficiency;
        double cache_efficiency;
        double pipeline_utilization;
    };
    
    PerformanceReport generate_report() const;
    void print_report() const;
    void reset_all_metrics();
    
    // Real-time monitoring
    void enable_real_time_monitoring(bool enable) { real_time_monitoring_ = enable; }
    void update_real_time_metrics();
    
    // Performance alerts
    void set_performance_threshold(const std::string& metric, double threshold);
    std::vector<std::string> check_performance_alerts() const;

private:
    // Timing data
    std::unordered_map<std::string, std::chrono::high_resolution_clock::time_point> start_times_;
    std::unordered_map<std::string, std::vector<double>> timing_history_;
    
    // Counters
    std::unordered_map<std::string, uint64_t> counters_;
    
    // Bandwidth tracking
    std::unordered_map<std::string, uint64_t> bandwidth_bytes_;
    std::unordered_map<std::string, std::chrono::high_resolution_clock::time_point> bandwidth_start_times_;
    
    // Cache statistics
    std::unordered_map<std::string, uint64_t> cache_hits_;
    std::unordered_map<std::string, uint64_t> cache_misses_;
    
    // Frame metrics
    std::vector<double> frame_times_;
    std::vector<uint32_t> triangle_counts_;
    std::vector<uint32_t> fragment_counts_;
    
    // Performance thresholds
    std::unordered_map<std::string, double> performance_thresholds_;
    
    // Configuration
    bool real_time_monitoring_;
    size_t max_history_size_;
    
    // Helper functions
    double calculate_average(const std::vector<double>& values) const;
    double calculate_variance(const std::vector<double>& values) const;
    double calculate_bandwidth_mbps(const std::string& component) const;
};

} // namespace gpu_sim