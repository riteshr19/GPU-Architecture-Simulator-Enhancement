#pragma once

#include <unordered_map>
#include <queue>
#include <vector>
#include <memory>
#include <chrono>
#include <cstdint>

namespace gpu_sim {

// Forward declarations
class MemoryHierarchy;
class PerformanceMonitor;

/**
 * Texture cache entry with metadata for advanced caching strategies
 */
struct TextureCacheEntry {
    uint64_t texture_id;
    uint32_t mip_level;
    uint64_t address;
    std::vector<uint8_t> data;
    uint64_t last_access_time;
    uint32_t access_count;
    float priority_score;
    bool is_prefetched;
    
    TextureCacheEntry(uint64_t id, uint32_t mip, uint64_t addr, size_t size)
        : texture_id(id), mip_level(mip), address(addr), data(size),
          last_access_time(0), access_count(0), priority_score(0.0f),
          is_prefetched(false) {}
};

/**
 * Advanced texture cache with intelligent prefetching and performance optimization
 * This is the NEW FEATURE that improves graphics pipeline performance
 */
class TextureCache {
public:
    explicit TextureCache(size_t cache_size_mb = 256);
    ~TextureCache() = default;

    // Initialization
    void initialize(std::shared_ptr<MemoryHierarchy> memory,
                   std::shared_ptr<PerformanceMonitor> perf_monitor);

    // Texture operations
    bool read_texture(uint64_t texture_id, uint32_t mip_level, 
                     uint64_t offset, void* data, size_t size);
    
    void prefetch_texture(uint64_t texture_id, uint32_t mip_level);
    void invalidate_texture(uint64_t texture_id);
    
    // Advanced features
    void enable_smart_prefetching(bool enable) { smart_prefetching_enabled_ = enable; }
    void enable_adaptive_caching(bool enable) { adaptive_caching_enabled_ = enable; }
    void set_prefetch_distance(uint32_t distance) { prefetch_distance_ = distance; }

    // Cache management
    void flush();
    void optimize_cache_layout();
    
    // Performance analytics
    struct CacheMetrics {
        uint64_t cache_hits;
        uint64_t cache_misses;
        uint64_t prefetch_hits;
        uint64_t prefetch_misses;
        double hit_rate;
        double prefetch_efficiency;
        uint64_t bytes_transferred;
        double avg_access_latency_ms;
        uint32_t cache_utilization_percent;
    };
    
    CacheMetrics get_metrics() const;
    void reset_metrics();

    // Configuration
    void tune_performance_parameters();

private:
    // Core cache functionality
    TextureCacheEntry* find_entry(uint64_t texture_id, uint32_t mip_level);
    TextureCacheEntry* allocate_entry(uint64_t texture_id, uint32_t mip_level, 
                                     uint64_t address, size_t size);
    void evict_least_valuable_entry();
    
    // Smart prefetching algorithms
    void analyze_access_patterns();
    void predict_future_accesses();
    void schedule_prefetch_operations();
    
    // Cache optimization
    void update_priority_scores();
    void reorder_cache_entries();
    float calculate_priority_score(const TextureCacheEntry& entry) const;
    
    // Adaptive algorithms
    void adapt_cache_parameters();
    void monitor_performance_trends();
    
    // Memory and performance
    std::shared_ptr<MemoryHierarchy> memory_;
    std::shared_ptr<PerformanceMonitor> perf_monitor_;
    
    // Cache storage
    std::unordered_map<uint64_t, std::unique_ptr<TextureCacheEntry>> cache_entries_;
    std::queue<uint64_t> prefetch_queue_;
    
    // Configuration
    size_t max_cache_size_bytes_;
    size_t current_cache_size_bytes_;
    uint32_t prefetch_distance_;
    bool smart_prefetching_enabled_;
    bool adaptive_caching_enabled_;
    
    // Access pattern analysis
    struct AccessPattern {
        uint64_t texture_id;
        uint32_t mip_level;
        uint64_t timestamp;
        float spatial_locality;
        float temporal_locality;
    };
    
    std::vector<AccessPattern> recent_accesses_;
    size_t max_pattern_history_;
    
    // Performance tracking
    mutable CacheMetrics metrics_;
    std::chrono::high_resolution_clock::time_point last_optimization_time_;
    
    // Adaptive parameters
    float prefetch_aggressiveness_;
    float eviction_threshold_;
    uint32_t optimization_interval_ms_;
    
    // Performance constants
    static constexpr float DEFAULT_PREFETCH_AGGRESSIVENESS = 0.7f;
    static constexpr float DEFAULT_EVICTION_THRESHOLD = 0.8f;
    static constexpr uint32_t DEFAULT_OPTIMIZATION_INTERVAL = 100;
    static constexpr size_t DEFAULT_PATTERN_HISTORY_SIZE = 1000;
};

} // namespace gpu_sim