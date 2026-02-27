#pragma once

#include <vector>
#include <memory>
#include <cstdint>
#include <cstring>

namespace gpu_sim {

class PerformanceMonitor;

/**
 * A sector within a cache line. Each sector tracks its own validity
 * and dirty state independently, enabling fine-grained data management.
 */
struct CacheSector {
    bool valid;
    bool dirty;
    std::vector<uint8_t> data;

    explicit CacheSector(size_t size)
        : valid(false), dirty(false), data(size, 0) {}
};

/**
 * A cache line composed of multiple sectors.
 * A 128-byte line with 4 sectors of 32 bytes each mirrors NVIDIA's
 * L1 sector cache design (Volta and later architectures).
 */
struct SectorCacheLine {
    uint64_t tag;
    uint64_t last_access_time;
    uint32_t access_count;
    std::vector<CacheSector> sectors;

    SectorCacheLine(uint64_t t, size_t num_sectors, size_t sector_size)
        : tag(t), last_access_time(0), access_count(0) {
        sectors.reserve(num_sectors);
        for (size_t i = 0; i < num_sectors; ++i) {
            sectors.emplace_back(sector_size);
        }
    }

    bool any_sector_valid() const {
        for (const auto& s : sectors) {
            if (s.valid) return true;
        }
        return false;
    }
};

/**
 * Bypass hint — controls whether a request should skip a cache level.
 */
enum class BypassHint {
    NO_BYPASS,         // Use normal caching path
    BYPASS_L1,         // Bypass L1, go directly to L2
    BYPASS_ALL_CACHE,  // Bypass all caches, access DRAM directly
    STREAMING          // Use streaming store (write-evict, don't allocate on miss)
};

/**
 * Sector Cache with Bypassing Support
 *
 * Key design features:
 *  1. Sector-level validity: A 128-byte line is divided into 4 × 32-byte sectors.
 *     Only the requested sector is fetched on a miss, reducing over-fetch.
 *  2. Adaptive bypassing: Memory-intensive loads can bypass the cache to avoid
 *     thrashing working-set data.
 *  3. Streaming access detection: Recognizes sequential patterns and
 *     automatically applies bypass hints.
 *
 * Reference: Gebhart et al., "Unifying Primary Cache, Scratch, and Register
 * File Memories in a Throughput Processor", MICRO 2012
 */
class SectorCache {
public:
    SectorCache(size_t total_size_kb,
                size_t line_size = 128,
                size_t sector_size = 32,
                size_t associativity = 4);
    ~SectorCache() = default;

    // Initialization
    void initialize(std::shared_ptr<PerformanceMonitor> perf_monitor);

    // Core cache operations with bypass support
    bool read(uint64_t address, void* data, size_t size,
              BypassHint hint = BypassHint::NO_BYPASS);

    bool write(uint64_t address, const void* data, size_t size,
               BypassHint hint = BypassHint::NO_BYPASS);

    void invalidate(uint64_t address);
    void flush();

    // Adaptive bypass control
    void enable_adaptive_bypass(bool enable) { adaptive_bypass_enabled_ = enable; }
    BypassHint get_adaptive_hint(uint64_t address) const;

    // Statistics
    struct SectorCacheStats {
        uint64_t hits;
        uint64_t misses;
        uint64_t sector_hits;       // Hits at sector granularity
        uint64_t sector_misses;     // Misses at sector granularity (partial line fill)
        uint64_t bypass_count;
        uint64_t streaming_count;
        uint64_t evictions;
        double   hit_rate;
        double   bandwidth_saved_pct; // Estimated bandwidth savings from sector fetch
    };

    SectorCacheStats get_statistics() const;
    void reset_statistics();

    // Configuration
    size_t get_total_size() const { return total_size_bytes_; }
    size_t get_line_size() const { return line_size_; }
    size_t get_sector_size() const { return sector_size_; }
    size_t get_associativity() const { return associativity_; }

private:
    // Internal lookup
    SectorCacheLine* find_line(uint64_t address);
    SectorCacheLine* allocate_line(uint64_t address);
    void evict_line(size_t set_index);

    // Address decomposition
    uint64_t get_tag(uint64_t address) const;
    size_t get_set_index(uint64_t address) const;
    size_t get_sector_index(uint64_t address) const;
    size_t get_sector_offset(uint64_t address) const;

    // Streaming detection
    void record_access(uint64_t address);
    bool is_streaming_access(uint64_t address) const;

    size_t total_size_bytes_;
    size_t line_size_;
    size_t sector_size_;
    size_t sectors_per_line_;
    size_t associativity_;
    size_t num_sets_;

    std::vector<std::vector<std::unique_ptr<SectorCacheLine>>> sets_;

    // Adaptive bypass state
    bool adaptive_bypass_enabled_;
    std::vector<uint64_t> recent_addresses_;
    size_t max_recent_addresses_;

    // Statistics
    mutable SectorCacheStats stats_;
    uint64_t access_counter_;
    std::shared_ptr<PerformanceMonitor> perf_monitor_;
};

} // namespace gpu_sim
