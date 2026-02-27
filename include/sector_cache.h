#pragma once

#include <vector>
#include <cstdint>
#include <string>

namespace gpu_sim {

/**
 * Sector Cache with optional bypassing.
 *
 * Background
 * ----------
 * Standard GPU L1 caches fetch a full 128-byte cache line on every miss,
 * even when only 4–8 bytes are needed.  For sparse or strided access
 * patterns this wastes bandwidth and pollutes the cache with unused data.
 *
 * A Sector Cache (introduced in NVIDIA Volta) divides each cache line into
 * fixed-size "sectors" (typically 32 bytes) and fetches only the sector(s)
 * that contain the requested bytes.  Each sector carries its own valid bit,
 * so a tag entry can be partially filled.
 *
 * Bypassing
 * ---------
 * When the access-reuse prediction heuristic determines that a request is
 * unlikely to be reused (e.g., streaming access or warp divergence > 50 %),
 * the sector is fetched from DRAM/L2 directly and never placed in L1.  This
 * prevents cache pollution without evicting useful lines.
 *
 * Design Parameters (matching NVIDIA Volta GV100 L1)
 * ---------------------------------------------------
 *   cache_size   = 128 KB (per SM, configurable)
 *   line_size    = 128 B
 *   sector_size  = 32 B   → 4 sectors per line
 *   associativity = 4-way set-associative
 */

static constexpr size_t SECTOR_SIZE = 32;           // bytes per sector
static constexpr size_t SECTORS_PER_LINE = 4;       // sectors per cache line
static constexpr size_t SECTOR_LINE_SIZE =
    SECTOR_SIZE * SECTORS_PER_LINE;                 // 128 B

/**
 * One sector within a cache line.
 */
struct CacheSector {
    bool          valid = false;
    bool          dirty = false;
    uint8_t       data[SECTOR_SIZE]{};
};

/**
 * A cache line tag entry containing SECTORS_PER_LINE sectors.
 */
struct SectorCacheLine {
    uint64_t      tag        = 0;
    bool          tag_valid  = false;          // at least one sector valid
    uint64_t      last_access_cycle = 0;
    uint32_t      access_count = 0;
    CacheSector   sectors[SECTORS_PER_LINE];
};

/**
 * Bypass heuristic: how aggressively to bypass the sector cache.
 */
enum class BypassPolicy {
    NEVER,          // never bypass (normal caching)
    DIVERGENT,      // bypass when warp active-mask popcount < threshold
    STREAMING,      // bypass detected sequential streaming accesses
    ALWAYS,         // always bypass (route all traffic through L2)
};

/**
 * SectorCache – GPU L1/L2 sector cache with sector-granular allocation and
 *               software-controlled bypassing to reduce thrashing.
 */
class SectorCache {
public:
    /**
     * @param cache_size_bytes  Total cache capacity.
     * @param associativity     Set associativity (ways per set).
     * @param bypass_policy     Bypass heuristic (default: DIVERGENT).
     * @param bypass_threshold  Active-mask popcount ratio below which to
     *                          bypass (0.0–1.0, default 0.5).
     */
    SectorCache(size_t cache_size_bytes,
                size_t associativity = 4,
                BypassPolicy bypass_policy = BypassPolicy::DIVERGENT,
                float bypass_threshold = 0.5f);
    ~SectorCache() = default;

    // ---- Cache operations -------------------------------------------------

    /**
     * Read bytes from the cache.
     * @param address     Byte address.
     * @param data        Output buffer.
     * @param size        Number of bytes to read.
     * @param active_mask Warp active-thread bitmask (used for bypass decision).
     * @param cycle       Current simulator cycle (for LRU tracking).
     * @return true on cache hit (all requested sectors present), false on miss.
     */
    bool read(uint64_t address, void* data, size_t size,
              uint32_t active_mask = 0xFFFFFFFF,
              uint64_t cycle = 0);

    /**
     * Write bytes to the cache (write-back, allocate on miss).
     * @param bypass_hint If true, skip L1 allocation (write directly to L2).
     */
    bool write(uint64_t address, const void* data, size_t size,
               bool bypass_hint = false,
               uint64_t cycle = 0);

    /**
     * Insert a sector that arrived from L2 (fill on miss).
     * Called by the memory system when a sector fetch completes.
     */
    void fill_sector(uint64_t address, const void* sector_data,
                     uint64_t cycle = 0);

    /** Invalidate all sectors covering the given address range. */
    void invalidate(uint64_t address, size_t size = SECTOR_LINE_SIZE);

    /** Flush (write-back + invalidate) the entire cache. */
    void flush();

    // ---- Bypass decision --------------------------------------------------

    /**
     * Determine whether a given access should bypass the cache.
     * @param active_mask Warp active-thread bitmask.
     * @param address     Access address (for streaming detection).
     */
    bool should_bypass(uint32_t active_mask, uint64_t address) const;

    // ---- Statistics -------------------------------------------------------

    struct SectorCacheStats {
        uint64_t sector_hits;           // individual sector hits
        uint64_t sector_misses;         // individual sector misses
        uint64_t line_tag_hits;         // tag present but sector missing
        uint64_t bypasses;              // accesses routed around cache
        uint64_t evictions;             // cache lines evicted (LRU)
        uint64_t dirty_evictions;       // dirty evictions requiring write-back
        double   sector_hit_rate;       // sector_hits / (hits + misses)
        double   bypass_rate;           // bypasses / total_accesses
    };

    SectorCacheStats get_stats() const;
    void             reset_stats();

    /** Human-readable statistics dump. */
    std::string get_stats_summary() const;

    // ---- Configuration accessors ------------------------------------------
    size_t num_sets()        const { return num_sets_; }
    size_t associativity()   const { return associativity_; }
    size_t cache_size_bytes()const { return cache_size_bytes_; }

private:
    // ---- Address decomposition --------------------------------------------
    uint32_t sector_index(uint64_t address) const;
    size_t   set_index(uint64_t address) const;
    uint64_t line_tag(uint64_t address) const;
    uint64_t line_base(uint64_t address) const;

    // ---- LRU set management ----------------------------------------------
    SectorCacheLine* find_line(uint64_t address, size_t set_idx);
    SectorCacheLine* allocate_line(uint64_t line_addr, size_t set_idx,
                                   uint64_t cycle);
    SectorCacheLine* lru_victim(size_t set_idx);

    // ---- State ------------------------------------------------------------
    size_t          cache_size_bytes_;
    size_t          associativity_;
    size_t          num_sets_;
    BypassPolicy    bypass_policy_;
    float           bypass_threshold_;

    // cache_sets_[set_idx][way] → SectorCacheLine
    std::vector<std::vector<SectorCacheLine>> cache_sets_;

    // Streaming detection: last address accessed (per-cache, simplified).
    uint64_t        last_access_address_ = 0;
    uint32_t        streaming_streak_    = 0;
    static constexpr uint32_t STREAMING_THRESHOLD = 8;

    // Statistics
    mutable SectorCacheStats stats_{};
    uint64_t                 total_accesses_ = 0;
};

} // namespace gpu_sim
