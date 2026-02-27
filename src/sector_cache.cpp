#include "sector_cache.h"
#include <algorithm>
#include <cstring>
#include <sstream>
#include <cassert>

namespace gpu_sim {

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

SectorCache::SectorCache(size_t cache_size_bytes,
                         size_t associativity,
                         BypassPolicy bypass_policy,
                         float bypass_threshold)
    : cache_size_bytes_(cache_size_bytes),
      associativity_(associativity),
      bypass_policy_(bypass_policy),
      bypass_threshold_(bypass_threshold)
{
    // Number of sets = total_lines / ways, where line = SECTOR_LINE_SIZE.
    size_t total_lines = cache_size_bytes_ / SECTOR_LINE_SIZE;
    num_sets_ = total_lines / associativity_;
    if (num_sets_ == 0) num_sets_ = 1;

    cache_sets_.assign(num_sets_, std::vector<SectorCacheLine>(associativity_));
    reset_stats();
}

// ---------------------------------------------------------------------------
// Address decomposition
// ---------------------------------------------------------------------------

uint32_t SectorCache::sector_index(uint64_t address) const {
    return static_cast<uint32_t>((address / SECTOR_SIZE) % SECTORS_PER_LINE);
}

size_t SectorCache::set_index(uint64_t address) const {
    return (address / SECTOR_LINE_SIZE) % num_sets_;
}

uint64_t SectorCache::line_tag(uint64_t address) const {
    return address / (SECTOR_LINE_SIZE * num_sets_);
}

uint64_t SectorCache::line_base(uint64_t address) const {
    return address & ~(static_cast<uint64_t>(SECTOR_LINE_SIZE) - 1);
}

// ---------------------------------------------------------------------------
// Set management
// ---------------------------------------------------------------------------

SectorCacheLine* SectorCache::find_line(uint64_t address, size_t set_idx) {
    uint64_t tag = line_tag(address);
    auto& set = cache_sets_[set_idx];
    for (auto& line : set) {
        if (line.tag_valid && line.tag == tag) {
            return &line;
        }
    }
    return nullptr;
}

SectorCacheLine* SectorCache::allocate_line(uint64_t line_addr,
                                             size_t set_idx,
                                             uint64_t cycle) {
    auto& set = cache_sets_[set_idx];

    // Look for an invalid (empty) slot first.
    for (auto& line : set) {
        if (!line.tag_valid) {
            line = SectorCacheLine{};
            line.tag = line_tag(line_addr);
            line.tag_valid = true;
            line.last_access_cycle = cycle;
            return &line;
        }
    }

    // All slots occupied: evict LRU.
    SectorCacheLine* victim = lru_victim(set_idx);
    if (victim) {
        // Count dirty evictions.
        for (auto& s : victim->sectors) {
            if (s.dirty) { stats_.dirty_evictions++; break; }
        }
        stats_.evictions++;
        *victim = SectorCacheLine{};
        victim->tag = line_tag(line_addr);
        victim->tag_valid = true;
        victim->last_access_cycle = cycle;
    }
    return victim;
}

SectorCacheLine* SectorCache::lru_victim(size_t set_idx) {
    auto& set = cache_sets_[set_idx];
    SectorCacheLine* victim = nullptr;
    for (auto& line : set) {
        if (!victim || line.last_access_cycle < victim->last_access_cycle) {
            victim = &line;
        }
    }
    return victim;
}

// ---------------------------------------------------------------------------
// Bypass decision
// ---------------------------------------------------------------------------

bool SectorCache::should_bypass(uint32_t active_mask, uint64_t address) const {
    switch (bypass_policy_) {
        case BypassPolicy::NEVER:
            return false;

        case BypassPolicy::ALWAYS:
            return true;

        case BypassPolicy::DIVERGENT: {
            // Bypass when warp utilisation < bypass_threshold_.
            uint32_t active_threads = static_cast<uint32_t>(__builtin_popcount(active_mask));
            float utilisation = static_cast<float>(active_threads) / 32.0f;
            return utilisation < bypass_threshold_;
        }

        case BypassPolicy::STREAMING: {
            // Bypass when we detect monotonically increasing sequential access.
            (void)active_mask;
            bool sequential = (address == last_access_address_ + SECTOR_SIZE ||
                               address == last_access_address_ + SECTOR_LINE_SIZE);
            return sequential && (streaming_streak_ >= STREAMING_THRESHOLD);
        }
    }
    return false;
}

// ---------------------------------------------------------------------------
// Read
// ---------------------------------------------------------------------------

bool SectorCache::read(uint64_t address, void* data, size_t size,
                       uint32_t active_mask, uint64_t cycle) {
    total_accesses_++;

    // Bypass decision.
    if (should_bypass(active_mask, address)) {
        stats_.bypasses++;
        last_access_address_ = address;
        return false;   // caller must fetch from L2/DRAM
    }

    size_t set_idx = set_index(address);
    SectorCacheLine* line = find_line(address, set_idx);

    uint32_t sec_idx = sector_index(address);
    size_t   offset  = address % SECTOR_SIZE;
    size_t   copy_sz = std::min(size, SECTOR_SIZE - offset);

    if (line && line->sectors[sec_idx].valid) {
        // Sector hit.
        std::memcpy(data, line->sectors[sec_idx].data + offset, copy_sz);
        line->last_access_cycle = cycle;
        line->access_count++;
        stats_.sector_hits++;

        // Update streaming detection.
        if (address == last_access_address_ + SECTOR_SIZE ||
            address == last_access_address_ + SECTOR_LINE_SIZE) {
            streaming_streak_++;
        } else {
            streaming_streak_ = 0;
        }
        last_access_address_ = address;
        return true;
    }

    // Miss.
    if (line) {
        stats_.line_tag_hits++;  // tag present but sector not filled yet
    }
    stats_.sector_misses++;
    last_access_address_ = address;
    return false;
}

// ---------------------------------------------------------------------------
// Write
// ---------------------------------------------------------------------------

bool SectorCache::write(uint64_t address, const void* data, size_t size,
                        bool bypass_hint, uint64_t cycle) {
    total_accesses_++;

    if (bypass_hint || should_bypass(0xFFFFFFFF, address)) {
        stats_.bypasses++;
        return false;   // write goes directly to L2
    }

    size_t set_idx = set_index(address);
    SectorCacheLine* line = find_line(address, set_idx);
    if (!line) {
        line = allocate_line(line_base(address), set_idx, cycle);
    }

    if (line) {
        uint32_t sec_idx = sector_index(address);
        size_t   offset  = address % SECTOR_SIZE;
        size_t   copy_sz = std::min(size, SECTOR_SIZE - offset);
        std::memcpy(line->sectors[sec_idx].data + offset, data, copy_sz);
        line->sectors[sec_idx].valid = true;
        line->sectors[sec_idx].dirty = true;
        line->last_access_cycle = cycle;
        line->access_count++;
        stats_.sector_hits++;
    }
    return true;
}

// ---------------------------------------------------------------------------
// Fill (from L2 / DRAM)
// ---------------------------------------------------------------------------

void SectorCache::fill_sector(uint64_t address, const void* sector_data,
                               uint64_t cycle) {
    size_t set_idx = set_index(address);
    SectorCacheLine* line = find_line(address, set_idx);
    if (!line) {
        line = allocate_line(line_base(address), set_idx, cycle);
    }
    if (line) {
        uint32_t sec_idx = sector_index(address);
        std::memcpy(line->sectors[sec_idx].data, sector_data, SECTOR_SIZE);
        line->sectors[sec_idx].valid = true;
        line->sectors[sec_idx].dirty = false;
        line->last_access_cycle = cycle;
    }
}

// ---------------------------------------------------------------------------
// Invalidate / flush
// ---------------------------------------------------------------------------

void SectorCache::invalidate(uint64_t address, size_t size) {
    for (uint64_t addr = line_base(address);
         addr < address + size;
         addr += SECTOR_LINE_SIZE) {
        size_t set_idx = set_index(addr);
        SectorCacheLine* line = find_line(addr, set_idx);
        if (line) {
            for (auto& s : line->sectors) {
                s.valid = false;
                s.dirty = false;
            }
            line->tag_valid = false;
        }
    }
}

void SectorCache::flush() {
    for (auto& set : cache_sets_) {
        for (auto& line : set) {
            line = SectorCacheLine{};
        }
    }
}

// ---------------------------------------------------------------------------
// Statistics
// ---------------------------------------------------------------------------

SectorCache::SectorCacheStats SectorCache::get_stats() const {
    SectorCacheStats s = stats_;
    uint64_t total = s.sector_hits + s.sector_misses;
    s.sector_hit_rate = total ? static_cast<double>(s.sector_hits) / total : 0.0;
    s.bypass_rate     = total_accesses_
                        ? static_cast<double>(s.bypasses) / total_accesses_
                        : 0.0;
    return s;
}

void SectorCache::reset_stats() {
    stats_ = SectorCacheStats{};
    total_accesses_ = 0;
    streaming_streak_ = 0;
    last_access_address_ = 0;
}

std::string SectorCache::get_stats_summary() const {
    auto s = get_stats();
    std::ostringstream oss;
    oss << "SectorCache stats:\n"
        << "  Sets         : " << num_sets_ << "\n"
        << "  Ways         : " << associativity_ << "\n"
        << "  Sector hits  : " << s.sector_hits << "\n"
        << "  Sector misses: " << s.sector_misses << "\n"
        << "  Tag hits     : " << s.line_tag_hits << " (tag present, sector missing)\n"
        << "  Bypasses     : " << s.bypasses << "\n"
        << "  Evictions    : " << s.evictions << "\n"
        << "  Dirty evicts : " << s.dirty_evictions << "\n"
        << "  Sector hit % : " << (s.sector_hit_rate * 100.0) << "%\n"
        << "  Bypass rate  : " << (s.bypass_rate * 100.0) << "%\n";
    return oss.str();
}

} // namespace gpu_sim
