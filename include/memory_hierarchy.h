#pragma once

#include <unordered_map>
#include <vector>
#include <memory>
#include <cstdint>

namespace gpu_sim {

/**
 * Cache line structure for GPU memory
 */
struct CacheLine {
    uint64_t address;
    std::vector<uint8_t> data;
    bool valid;
    bool dirty;
    uint64_t access_time;
    
    CacheLine(uint64_t addr, size_t size) 
        : address(addr), data(size), valid(false), dirty(false), access_time(0) {}
};

/**
 * GPU cache implementation with configurable associativity
 */
class GPUCache {
public:
    GPUCache(size_t cache_size, size_t line_size, size_t associativity);
    ~GPUCache() = default;

    // Cache operations
    bool read(uint64_t address, void* data, size_t size);
    bool write(uint64_t address, const void* data, size_t size);
    void invalidate(uint64_t address);
    void flush();

    // Statistics
    uint64_t get_hit_count() const { return hit_count_; }
    uint64_t get_miss_count() const { return miss_count_; }
    double get_hit_rate() const;

private:
    size_t cache_size_;
    size_t line_size_;
    size_t associativity_;
    size_t num_sets_;
    
    std::vector<std::vector<std::unique_ptr<CacheLine>>> cache_sets_;
    
    uint64_t hit_count_;
    uint64_t miss_count_;
    uint64_t access_count_;

    size_t get_set_index(uint64_t address) const;
    uint64_t get_tag(uint64_t address) const;
    CacheLine* find_line(uint64_t address);
    CacheLine* allocate_line(uint64_t address);
};

/**
 * GPU memory hierarchy including L1, L2 caches and VRAM
 */
class MemoryHierarchy {
public:
    MemoryHierarchy();
    ~MemoryHierarchy() = default;

    // Memory operations
    bool read(uint64_t address, void* data, size_t size);
    bool write(uint64_t address, const void* data, size_t size);
    
    // Memory allocation
    uint64_t allocate(size_t size);
    void deallocate(uint64_t address);
    
    // Cache management
    void flush_all_caches();
    
    // Statistics
    struct MemoryStats {
        uint64_t l1_hits, l1_misses;
        uint64_t l2_hits, l2_misses;
        uint64_t vram_accesses;
        double avg_access_latency;
    };
    
    MemoryStats get_statistics() const;

private:
    std::unique_ptr<GPUCache> l1_cache_;
    std::unique_ptr<GPUCache> l2_cache_;
    std::vector<uint8_t> vram_;
    
    uint64_t next_allocation_address_;
    std::unordered_map<uint64_t, size_t> allocations_;
    
    // Timing constants
    static constexpr uint32_t L1_LATENCY = 1;
    static constexpr uint32_t L2_LATENCY = 10;
    static constexpr uint32_t VRAM_LATENCY = 100;
};

} // namespace gpu_sim