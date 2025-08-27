#include "texture_cache.h"
#include "memory_hierarchy.h"
#include "performance_monitor.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <cstring>

namespace gpu_sim {

TextureCache::TextureCache(size_t cache_size_mb)
    : max_cache_size_bytes_(cache_size_mb * 1024 * 1024),
      current_cache_size_bytes_(0),
      prefetch_distance_(DEFAULT_OPTIMIZATION_INTERVAL),
      smart_prefetching_enabled_(true),
      adaptive_caching_enabled_(true),
      max_pattern_history_(DEFAULT_PATTERN_HISTORY_SIZE),
      prefetch_aggressiveness_(DEFAULT_PREFETCH_AGGRESSIVENESS),
      eviction_threshold_(DEFAULT_EVICTION_THRESHOLD),
      optimization_interval_ms_(DEFAULT_OPTIMIZATION_INTERVAL) {
    
    recent_accesses_.reserve(max_pattern_history_);
    last_optimization_time_ = std::chrono::high_resolution_clock::now();
    
    // Initialize metrics
    reset_metrics();
}

void TextureCache::initialize(std::shared_ptr<MemoryHierarchy> memory,
                             std::shared_ptr<PerformanceMonitor> perf_monitor) {
    memory_ = memory;
    perf_monitor_ = perf_monitor;
    
    if (perf_monitor_) {
        perf_monitor_->set_counter("texture_cache_size_mb", max_cache_size_bytes_ / (1024 * 1024));
    }
}

bool TextureCache::read_texture(uint64_t texture_id, uint32_t mip_level,
                               uint64_t offset, void* data, size_t size) {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Record access pattern for analysis
    if (recent_accesses_.size() >= max_pattern_history_) {
        recent_accesses_.erase(recent_accesses_.begin());
    }
    
    AccessPattern pattern;
    pattern.texture_id = texture_id;
    pattern.mip_level = mip_level;
    pattern.timestamp = std::chrono::duration_cast<std::chrono::microseconds>(
        start_time.time_since_epoch()).count();
    recent_accesses_.push_back(pattern);
    
    // Generate cache key
    uint64_t cache_key = (texture_id << 8) | mip_level;
    
    // Check cache
    auto it = cache_entries_.find(cache_key);
    if (it != cache_entries_.end()) {
        auto& entry = it->second;
        
        // Update access metadata
        entry->last_access_time = pattern.timestamp;
        entry->access_count++;
        
        // Check if requested data is within cached texture
        if (offset + size <= entry->data.size()) {
            memcpy(data, entry->data.data() + offset, size);
            
            // Update metrics
            metrics_.cache_hits++;
            if (entry->is_prefetched) {
                metrics_.prefetch_hits++;
            }
            
            auto end_time = std::chrono::high_resolution_clock::now();
            auto latency = std::chrono::duration<double, std::milli>(end_time - start_time).count();
            
            if (perf_monitor_) {
                perf_monitor_->record_cache_access("texture_cache", true);
                perf_monitor_->increment_counter("texture_cache_bytes_read", size);
            }
            
            // Trigger smart prefetching
            if (smart_prefetching_enabled_) {
                predict_future_accesses();
            }
            
            return true;
        }
    }
    
    // Cache miss - load from memory
    metrics_.cache_misses++;
    
    if (perf_monitor_) {
        perf_monitor_->record_cache_access("texture_cache", false);
        perf_monitor_->start_timer("texture_load_from_memory");
    }
    
    // Calculate texture size (simplified - assume 4 bytes per pixel)
    size_t texture_size = std::max(size, static_cast<size_t>(1024 * 1024)); // At least 1MB
    
    // Allocate memory for full texture
    uint64_t texture_address = memory_->allocate(texture_size);
    if (texture_address == 0) {
        return false; // Memory allocation failed
    }
    
    // Load texture data from memory hierarchy
    std::vector<uint8_t> texture_data(texture_size);
    if (!memory_->read(texture_address, texture_data.data(), texture_size)) {
        memory_->deallocate(texture_address);
        return false;
    }
    
    // Create cache entry
    auto entry = allocate_entry(texture_id, mip_level, texture_address, texture_size);
    if (entry) {
        entry->data = std::move(texture_data);
        entry->last_access_time = pattern.timestamp;
        entry->access_count = 1;
        entry->is_prefetched = false;
        
        // Copy requested data
        if (offset + size <= entry->data.size()) {
            memcpy(data, entry->data.data() + offset, size);
        }
        
        metrics_.bytes_transferred += texture_size;
    }
    
    if (perf_monitor_) {
        perf_monitor_->end_timer("texture_load_from_memory");
    }
    
    // Update performance metrics
    auto end_time = std::chrono::high_resolution_clock::now();
    auto latency = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    
    // Trigger adaptive optimization
    if (adaptive_caching_enabled_) {
        auto now = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
            now - last_optimization_time_).count();
        
        if (elapsed >= optimization_interval_ms_) {
            tune_performance_parameters();
            last_optimization_time_ = now;
        }
    }
    
    return entry != nullptr;
}

void TextureCache::prefetch_texture(uint64_t texture_id, uint32_t mip_level) {
    uint64_t cache_key = (texture_id << 8) | mip_level;
    
    // Don't prefetch if already cached
    if (cache_entries_.find(cache_key) != cache_entries_.end()) {
        return;
    }
    
    // Add to prefetch queue
    prefetch_queue_.push(cache_key);
    
    // Process prefetch queue (simplified - immediate processing)
    if (!prefetch_queue_.empty()) {
        uint64_t prefetch_key = prefetch_queue_.front();
        prefetch_queue_.pop();
        
        uint64_t prefetch_texture_id = prefetch_key >> 8;
        uint32_t prefetch_mip = prefetch_key & 0xFF;
        
        // Simulate prefetch operation
        size_t texture_size = 1024 * 1024; // 1MB default
        uint64_t texture_address = memory_->allocate(texture_size);
        
        if (texture_address != 0) {
            auto entry = allocate_entry(prefetch_texture_id, prefetch_mip, 
                                       texture_address, texture_size);
            if (entry) {
                entry->data.resize(texture_size);
                memory_->read(texture_address, entry->data.data(), texture_size);
                entry->is_prefetched = true;
                
                metrics_.bytes_transferred += texture_size;
                
                if (perf_monitor_) {
                    perf_monitor_->increment_counter("texture_prefetch_operations");
                }
            }
        }
    }
}

TextureCacheEntry* TextureCache::find_entry(uint64_t texture_id, uint32_t mip_level) {
    uint64_t cache_key = (texture_id << 8) | mip_level;
    auto it = cache_entries_.find(cache_key);
    return (it != cache_entries_.end()) ? it->second.get() : nullptr;
}

TextureCacheEntry* TextureCache::allocate_entry(uint64_t texture_id, uint32_t mip_level,
                                               uint64_t address, size_t size) {
    // Check if we need to evict entries
    while (current_cache_size_bytes_ + size > max_cache_size_bytes_ && !cache_entries_.empty()) {
        evict_least_valuable_entry();
    }
    
    uint64_t cache_key = (texture_id << 8) | mip_level;
    auto entry = std::make_unique<TextureCacheEntry>(texture_id, mip_level, address, size);
    
    TextureCacheEntry* entry_ptr = entry.get();
    cache_entries_[cache_key] = std::move(entry);
    current_cache_size_bytes_ += size;
    
    return entry_ptr;
}

void TextureCache::evict_least_valuable_entry() {
    if (cache_entries_.empty()) return;
    
    // Find entry with lowest priority score
    auto least_valuable = std::min_element(cache_entries_.begin(), cache_entries_.end(),
        [this](const auto& a, const auto& b) {
            return calculate_priority_score(*a.second) < calculate_priority_score(*b.second);
        });
    
    if (least_valuable != cache_entries_.end()) {
        current_cache_size_bytes_ -= least_valuable->second->data.size();
        memory_->deallocate(least_valuable->second->address);
        cache_entries_.erase(least_valuable);
    }
}

float TextureCache::calculate_priority_score(const TextureCacheEntry& entry) const {
    auto current_time = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::high_resolution_clock::now().time_since_epoch()).count();
    
    // Factors: recency, frequency, prefetch status
    float recency_score = 1.0f / (1.0f + (current_time - entry.last_access_time) / 1000000.0f);
    float frequency_score = std::log(1.0f + entry.access_count);
    float prefetch_bonus = entry.is_prefetched ? 0.5f : 1.0f;
    
    return recency_score * frequency_score * prefetch_bonus;
}

void TextureCache::predict_future_accesses() {
    if (recent_accesses_.size() < 3) return;
    
    // Simple pattern detection: look for sequential texture accesses
    auto& last_access = recent_accesses_.back();
    
    // Predict next mip level or texture ID
    if (recent_accesses_.size() >= 2) {
        auto& prev_access = recent_accesses_[recent_accesses_.size() - 2];
        
        if (prev_access.texture_id == last_access.texture_id) {
            // Same texture, different mip level - prefetch next mip
            uint32_t next_mip = last_access.mip_level + 1;
            if (next_mip < 16) { // Reasonable mip level limit
                prefetch_texture(last_access.texture_id, next_mip);
            }
        } else if (last_access.texture_id == prev_access.texture_id + 1) {
            // Sequential texture access - prefetch next texture
            prefetch_texture(last_access.texture_id + 1, last_access.mip_level);
        }
    }
}

void TextureCache::tune_performance_parameters() {
    // Analyze cache performance and adjust parameters
    double hit_rate = static_cast<double>(metrics_.cache_hits) / 
                     std::max(static_cast<uint64_t>(1), metrics_.cache_hits + metrics_.cache_misses);
    
    double prefetch_efficiency = static_cast<double>(metrics_.prefetch_hits) /
                                std::max(static_cast<uint64_t>(1), metrics_.prefetch_hits + metrics_.prefetch_misses);
    
    // Adjust prefetch aggressiveness based on efficiency
    if (prefetch_efficiency > 0.7) {
        prefetch_aggressiveness_ = std::min(1.0f, prefetch_aggressiveness_ + 0.1f);
    } else if (prefetch_efficiency < 0.3) {
        prefetch_aggressiveness_ = std::max(0.1f, prefetch_aggressiveness_ - 0.1f);
    }
    
    // Adjust eviction threshold based on hit rate
    if (hit_rate > 0.9) {
        eviction_threshold_ = std::min(0.9f, eviction_threshold_ + 0.05f);
    } else if (hit_rate < 0.7) {
        eviction_threshold_ = std::max(0.5f, eviction_threshold_ - 0.05f);
    }
    
    if (perf_monitor_) {
        perf_monitor_->set_counter("texture_cache_hit_rate_percent", 
                                  static_cast<uint64_t>(hit_rate * 100));
        perf_monitor_->set_counter("prefetch_efficiency_percent",
                                  static_cast<uint64_t>(prefetch_efficiency * 100));
    }
}

void TextureCache::invalidate_texture(uint64_t texture_id) {
    auto it = cache_entries_.begin();
    while (it != cache_entries_.end()) {
        if (it->second->texture_id == texture_id) {
            current_cache_size_bytes_ -= it->second->data.size();
            memory_->deallocate(it->second->address);
            it = cache_entries_.erase(it);
        } else {
            ++it;
        }
    }
}

void TextureCache::flush() {
    for (auto& entry : cache_entries_) {
        memory_->deallocate(entry.second->address);
    }
    cache_entries_.clear();
    current_cache_size_bytes_ = 0;
    
    // Clear prefetch queue
    while (!prefetch_queue_.empty()) {
        prefetch_queue_.pop();
    }
}

TextureCache::CacheMetrics TextureCache::get_metrics() const {
    CacheMetrics metrics = metrics_;
    
    uint64_t total_accesses = metrics.cache_hits + metrics.cache_misses;
    if (total_accesses > 0) {
        metrics.hit_rate = static_cast<double>(metrics.cache_hits) / total_accesses;
    }
    
    uint64_t total_prefetches = metrics.prefetch_hits + metrics.prefetch_misses;
    if (total_prefetches > 0) {
        metrics.prefetch_efficiency = static_cast<double>(metrics.prefetch_hits) / total_prefetches;
    }
    
    metrics.cache_utilization_percent = static_cast<uint32_t>(
        (current_cache_size_bytes_ * 100) / max_cache_size_bytes_);
    
    return metrics;
}

void TextureCache::reset_metrics() {
    metrics_ = CacheMetrics{};
}

} // namespace gpu_sim