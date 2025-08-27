#include "performance_monitor.h"
#include <algorithm>
#include <numeric>
#include <iostream>
#include <iomanip>

namespace gpu_sim {

PerformanceMonitor::PerformanceMonitor() 
    : real_time_monitoring_(false), max_history_size_(1000) {
}

void PerformanceMonitor::start_timer(const std::string& event) {
    start_times_[event] = std::chrono::high_resolution_clock::now();
}

void PerformanceMonitor::end_timer(const std::string& event) {
    auto end_time = std::chrono::high_resolution_clock::now();
    auto it = start_times_.find(event);
    
    if (it != start_times_.end()) {
        auto duration = std::chrono::duration<double, std::milli>(end_time - it->second).count();
        
        if (timing_history_[event].size() >= max_history_size_) {
            timing_history_[event].erase(timing_history_[event].begin());
        }
        timing_history_[event].push_back(duration);
        
        start_times_.erase(it);
    }
}

double PerformanceMonitor::get_elapsed_time_ms(const std::string& event) const {
    auto it = timing_history_.find(event);
    if (it != timing_history_.end() && !it->second.empty()) {
        return calculate_average(it->second);
    }
    return 0.0;
}

void PerformanceMonitor::increment_counter(const std::string& counter, uint64_t value) {
    counters_[counter] += value;
}

void PerformanceMonitor::set_counter(const std::string& counter, uint64_t value) {
    counters_[counter] = value;
}

uint64_t PerformanceMonitor::get_counter(const std::string& counter) const {
    auto it = counters_.find(counter);
    return (it != counters_.end()) ? it->second : 0;
}

void PerformanceMonitor::record_bandwidth_usage(const std::string& component, uint64_t bytes) {
    auto now = std::chrono::high_resolution_clock::now();
    
    // Initialize start time if this is the first measurement
    if (bandwidth_start_times_.find(component) == bandwidth_start_times_.end()) {
        bandwidth_start_times_[component] = now;
    }
    
    bandwidth_bytes_[component] += bytes;
}

void PerformanceMonitor::record_cache_access(const std::string& cache, bool hit) {
    if (hit) {
        cache_hits_[cache]++;
    } else {
        cache_misses_[cache]++;
    }
}

void PerformanceMonitor::record_frame_metrics(double frame_time_ms, uint32_t triangles, uint32_t fragments) {
    if (frame_times_.size() >= max_history_size_) {
        frame_times_.erase(frame_times_.begin());
        triangle_counts_.erase(triangle_counts_.begin());
        fragment_counts_.erase(fragment_counts_.begin());
    }
    
    frame_times_.push_back(frame_time_ms);
    triangle_counts_.push_back(triangles);
    fragment_counts_.push_back(fragments);
}

double PerformanceMonitor::calculate_average(const std::vector<double>& values) const {
    if (values.empty()) return 0.0;
    return std::accumulate(values.begin(), values.end(), 0.0) / values.size();
}

double PerformanceMonitor::calculate_variance(const std::vector<double>& values) const {
    if (values.size() < 2) return 0.0;
    
    double mean = calculate_average(values);
    double sum_squared_diff = 0.0;
    
    for (double value : values) {
        double diff = value - mean;
        sum_squared_diff += diff * diff;
    }
    
    return sum_squared_diff / (values.size() - 1);
}

double PerformanceMonitor::calculate_bandwidth_mbps(const std::string& component) const {
    auto bytes_it = bandwidth_bytes_.find(component);
    auto start_it = bandwidth_start_times_.find(component);
    
    if (bytes_it == bandwidth_bytes_.end() || start_it == bandwidth_start_times_.end()) {
        return 0.0;
    }
    
    auto now = std::chrono::high_resolution_clock::now();
    auto elapsed_seconds = std::chrono::duration<double>(now - start_it->second).count();
    
    if (elapsed_seconds <= 0) return 0.0;
    
    double megabytes = static_cast<double>(bytes_it->second) / (1024.0 * 1024.0);
    return megabytes / elapsed_seconds;
}

PerformanceMonitor::PerformanceReport PerformanceMonitor::generate_report() const {
    PerformanceReport report;
    
    // Timing data
    for (const auto& [event, times] : timing_history_) {
        if (!times.empty()) {
            report.timing_data[event] = calculate_average(times);
        }
    }
    
    // Counter data
    report.counter_data = counters_;
    
    // Bandwidth data
    for (const auto& [component, _] : bandwidth_bytes_) {
        report.bandwidth_data[component] = calculate_bandwidth_mbps(component);
    }
    
    // Cache hit rates
    for (const auto& [cache, hits] : cache_hits_) {
        uint64_t misses = 0;
        auto miss_it = cache_misses_.find(cache);
        if (miss_it != cache_misses_.end()) {
            misses = miss_it->second;
        }
        
        uint64_t total_accesses = hits + misses;
        if (total_accesses > 0) {
            report.cache_hit_rates[cache] = static_cast<double>(hits) / total_accesses;
        }
    }
    
    // Frame metrics
    if (!frame_times_.empty()) {
        report.avg_frame_time_ms = calculate_average(frame_times_);
        report.min_frame_time_ms = *std::min_element(frame_times_.begin(), frame_times_.end());
        report.max_frame_time_ms = *std::max_element(frame_times_.begin(), frame_times_.end());
        
        report.total_triangles = std::accumulate(triangle_counts_.begin(), triangle_counts_.end(), 0ULL);
        report.total_fragments = std::accumulate(fragment_counts_.begin(), fragment_counts_.end(), 0ULL);
    } else {
        report.avg_frame_time_ms = 0.0;
        report.min_frame_time_ms = 0.0;
        report.max_frame_time_ms = 0.0;
        report.total_triangles = 0;
        report.total_fragments = 0;
    }
    
    // Efficiency metrics
    report.memory_efficiency = 0.0;
    report.cache_efficiency = 0.0;
    report.pipeline_utilization = 0.0;
    
    // Calculate memory efficiency based on cache hit rates
    if (!report.cache_hit_rates.empty()) {
        double total_hit_rate = 0.0;
        for (const auto& [cache, hit_rate] : report.cache_hit_rates) {
            total_hit_rate += hit_rate;
        }
        report.memory_efficiency = total_hit_rate / report.cache_hit_rates.size();
    }
    
    // Calculate cache efficiency (average of all cache hit rates)
    report.cache_efficiency = report.memory_efficiency;
    
    // Calculate pipeline utilization based on frame rate
    if (report.avg_frame_time_ms > 0) {
        double fps = 1000.0 / report.avg_frame_time_ms;
        double target_fps = 60.0; // Assume 60 FPS target
        report.pipeline_utilization = std::min(1.0, fps / target_fps);
    }
    
    return report;
}

void PerformanceMonitor::print_report() const {
    auto report = generate_report();
    
    std::cout << "\n=== GPU Architecture Simulator Performance Report ===" << std::endl;
    std::cout << std::fixed << std::setprecision(3);
    
    // Timing information
    std::cout << "\nTiming Information:" << std::endl;
    for (const auto& [event, avg_time] : report.timing_data) {
        std::cout << "  " << event << ": " << avg_time << " ms" << std::endl;
    }
    
    // Frame metrics
    std::cout << "\nFrame Metrics:" << std::endl;
    std::cout << "  Average frame time: " << report.avg_frame_time_ms << " ms" << std::endl;
    std::cout << "  Min frame time: " << report.min_frame_time_ms << " ms" << std::endl;
    std::cout << "  Max frame time: " << report.max_frame_time_ms << " ms" << std::endl;
    if (report.avg_frame_time_ms > 0) {
        std::cout << "  Average FPS: " << (1000.0 / report.avg_frame_time_ms) << std::endl;
    }
    std::cout << "  Total triangles: " << report.total_triangles << std::endl;
    std::cout << "  Total fragments: " << report.total_fragments << std::endl;
    
    // Cache performance
    std::cout << "\nCache Performance:" << std::endl;
    for (const auto& [cache, hit_rate] : report.cache_hit_rates) {
        std::cout << "  " << cache << " hit rate: " << (hit_rate * 100.0) << "%" << std::endl;
    }
    
    // Bandwidth usage
    std::cout << "\nBandwidth Usage:" << std::endl;
    for (const auto& [component, bandwidth] : report.bandwidth_data) {
        std::cout << "  " << component << ": " << bandwidth << " MB/s" << std::endl;
    }
    
    // Efficiency metrics
    std::cout << "\nEfficiency Metrics:" << std::endl;
    std::cout << "  Memory efficiency: " << (report.memory_efficiency * 100.0) << "%" << std::endl;
    std::cout << "  Cache efficiency: " << (report.cache_efficiency * 100.0) << "%" << std::endl;
    std::cout << "  Pipeline utilization: " << (report.pipeline_utilization * 100.0) << "%" << std::endl;
    
    // Counter information
    std::cout << "\nCounter Information:" << std::endl;
    for (const auto& [counter, value] : report.counter_data) {
        std::cout << "  " << counter << ": " << value << std::endl;
    }
    
    std::cout << "\n=== End of Performance Report ===" << std::endl;
}

void PerformanceMonitor::reset_all_metrics() {
    start_times_.clear();
    timing_history_.clear();
    counters_.clear();
    bandwidth_bytes_.clear();
    bandwidth_start_times_.clear();
    cache_hits_.clear();
    cache_misses_.clear();
    frame_times_.clear();
    triangle_counts_.clear();
    fragment_counts_.clear();
    performance_thresholds_.clear();
}

void PerformanceMonitor::update_real_time_metrics() {
    if (!real_time_monitoring_) return;
    
    // This would update real-time metrics display
    // For simulation, we just check performance alerts
    check_performance_alerts();
}

void PerformanceMonitor::set_performance_threshold(const std::string& metric, double threshold) {
    performance_thresholds_[metric] = threshold;
}

std::vector<std::string> PerformanceMonitor::check_performance_alerts() const {
    std::vector<std::string> alerts;
    
    for (const auto& [metric, threshold] : performance_thresholds_) {
        bool alert_triggered = false;
        std::string alert_message;
        
        if (metric == "frame_time_ms" && !frame_times_.empty()) {
            double current_frame_time = frame_times_.back();
            if (current_frame_time > threshold) {
                alert_triggered = true;
                alert_message = "Frame time exceeded threshold: " + 
                              std::to_string(current_frame_time) + " ms > " + 
                              std::to_string(threshold) + " ms";
            }
        } else if (metric.find("hit_rate") != std::string::npos) {
            // Extract cache name from metric
            std::string cache_name = metric.substr(0, metric.find("_hit_rate"));
            auto hit_it = cache_hits_.find(cache_name);
            auto miss_it = cache_misses_.find(cache_name);
            
            if (hit_it != cache_hits_.end() && miss_it != cache_misses_.end()) {
                uint64_t total = hit_it->second + miss_it->second;
                if (total > 0) {
                    double hit_rate = static_cast<double>(hit_it->second) / total;
                    if (hit_rate < threshold) {
                        alert_triggered = true;
                        alert_message = cache_name + " hit rate below threshold: " +
                                      std::to_string(hit_rate * 100.0) + "% < " +
                                      std::to_string(threshold * 100.0) + "%";
                    }
                }
            }
        }
        
        if (alert_triggered) {
            alerts.push_back(alert_message);
        }
    }
    
    return alerts;
}

} // namespace gpu_sim