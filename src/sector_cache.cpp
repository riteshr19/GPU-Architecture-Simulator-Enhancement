#include "sector_cache.h"
#include "performance_monitor.h"
#include <algorithm>
#include <cstring>
#include <numeric>

namespace gpu_sim {

SectorCache::SectorCache(size_t total_size_kb, size_t line_size,
                         size_t sector_size, size_t associativity)
    : total_size_bytes_(total_size_kb * 1024),
      line_size_(line_size),
      sector_size_(sector_size),
      sectors_per_line_(line_size / sector_size),
      associativity_(associativity),
      adaptive_bypass_enabled_(false),
      max_recent_addresses_(64),
      access_counter_(0) {

    num_sets_ = total_size_bytes_ / (line_size_ * associativity_);
    sets_.resize(num_sets_);
    for (auto& set : sets_) {
        set.resize(associativity_); // nullptr initially
    }

    stats_ = SectorCacheStats{};
}

void SectorCache::initialize(std::shared_ptr<PerformanceMonitor> perf_monitor) {
    perf_monitor_ = perf_monitor;
    if (perf_monitor_) {
        perf_monitor_->set_counter("sector_cache_size_kb", total_size_bytes_ / 1024);
        perf_monitor_->set_counter("sector_cache_associativity", associativity_);
    }
}

// ---------------------------------------------------------------------------
// Address decomposition
// ---------------------------------------------------------------------------
uint64_t SectorCache::get_tag(uint64_t address) const {
    return address / (line_size_ * num_sets_);
}

size_t SectorCache::get_set_index(uint64_t address) const {
    return (address / line_size_) % num_sets_;
}

size_t SectorCache::get_sector_index(uint64_t address) const {
    return (address % line_size_) / sector_size_;
}

size_t SectorCache::get_sector_offset(uint64_t address) const {
    return address % sector_size_;
}

// ---------------------------------------------------------------------------
// Line lookup
// ---------------------------------------------------------------------------
SectorCacheLine* SectorCache::find_line(uint64_t address) {
    size_t set_idx = get_set_index(address);
    uint64_t tag = get_tag(address);

    for (auto& line : sets_[set_idx]) {
        if (line && line->tag == tag && line->any_sector_valid()) {
            return line.get();
        }
    }
    return nullptr;
}

SectorCacheLine* SectorCache::allocate_line(uint64_t address) {
    size_t set_idx = get_set_index(address);
    uint64_t tag = get_tag(address);

    // Find an empty way
    for (auto& line : sets_[set_idx]) {
        if (!line || !line->any_sector_valid()) {
            line = std::make_unique<SectorCacheLine>(tag, sectors_per_line_, sector_size_);
            line->last_access_time = access_counter_;
            return line.get();
        }
    }

    // Evict LRU line
    evict_line(set_idx);

    // Now find the freed slot
    for (auto& line : sets_[set_idx]) {
        if (!line || !line->any_sector_valid()) {
            line = std::make_unique<SectorCacheLine>(tag, sectors_per_line_, sector_size_);
            line->last_access_time = access_counter_;
            return line.get();
        }
    }

    return nullptr; // Should not reach here
}

void SectorCache::evict_line(size_t set_index) {
    auto& set = sets_[set_index];
    auto lru_it = std::min_element(set.begin(), set.end(),
        [](const auto& a, const auto& b) {
            if (!a || !a->any_sector_valid()) return true;
            if (!b || !b->any_sector_valid()) return false;
            return a->last_access_time < b->last_access_time;
        });

    if (lru_it != set.end()) {
        lru_it->reset();
        stats_.evictions++;
    }
}

// ---------------------------------------------------------------------------
// Read with bypass support
// ---------------------------------------------------------------------------
bool SectorCache::read(uint64_t address, void* data, size_t size,
                       BypassHint hint) {
    access_counter_++;
    record_access(address);

    // Apply adaptive bypass if enabled
    if (adaptive_bypass_enabled_ && hint == BypassHint::NO_BYPASS) {
        hint = get_adaptive_hint(address);
    }

    // Handle bypass
    if (hint == BypassHint::BYPASS_L1 || hint == BypassHint::BYPASS_ALL_CACHE) {
        stats_.bypass_count++;
        stats_.misses++;
        return false; // Caller should go to next level
    }

    if (hint == BypassHint::STREAMING) {
        stats_.streaming_count++;
    }

    // Try to find the cache line
    SectorCacheLine* line = find_line(address);
    size_t sector_idx = get_sector_index(address);

    if (line) {
        line->last_access_time = access_counter_;
        line->access_count++;

        // Check if the specific sector is valid
        if (sector_idx < line->sectors.size() && line->sectors[sector_idx].valid) {
            // Sector hit
            stats_.hits++;
            stats_.sector_hits++;
            size_t offset = get_sector_offset(address);
            size_t copy_size = std::min(size, sector_size_ - offset);
            std::memcpy(data, line->sectors[sector_idx].data.data() + offset, copy_size);

            if (perf_monitor_) {
                perf_monitor_->record_cache_access("sector_cache", true);
            }
            return true;
        } else {
            // Line present but sector invalid — partial miss
            stats_.sector_misses++;
            stats_.misses++;
            if (perf_monitor_) {
                perf_monitor_->record_cache_access("sector_cache", false);
            }
            return false;
        }
    }

    // Full miss
    stats_.misses++;
    stats_.sector_misses++;
    if (perf_monitor_) {
        perf_monitor_->record_cache_access("sector_cache", false);
    }
    return false;
}

// ---------------------------------------------------------------------------
// Write with bypass support
// ---------------------------------------------------------------------------
bool SectorCache::write(uint64_t address, const void* data, size_t size,
                        BypassHint hint) {
    access_counter_++;

    if (hint == BypassHint::BYPASS_ALL_CACHE) {
        stats_.bypass_count++;
        return false;
    }

    SectorCacheLine* line = find_line(address);
    if (!line) {
        if (hint == BypassHint::STREAMING) {
            stats_.streaming_count++;
            return false; // Streaming: write-evict, don't allocate
        }
        line = allocate_line(address);
        if (!line) return false;
    }

    size_t sector_idx = get_sector_index(address);
    if (sector_idx < line->sectors.size()) {
        auto& sector = line->sectors[sector_idx];
        size_t offset = get_sector_offset(address);
        size_t copy_size = std::min(size, sector_size_ - offset);
        std::memcpy(sector.data.data() + offset, data, copy_size);
        sector.valid = true;
        sector.dirty = true;
        line->last_access_time = access_counter_;
        line->access_count++;
    }

    return true;
}

void SectorCache::invalidate(uint64_t address) {
    SectorCacheLine* line = find_line(address);
    if (line) {
        size_t sector_idx = get_sector_index(address);
        if (sector_idx < line->sectors.size()) {
            line->sectors[sector_idx].valid = false;
            line->sectors[sector_idx].dirty = false;
        }
    }
}

void SectorCache::flush() {
    for (auto& set : sets_) {
        for (auto& line : set) {
            line.reset();
        }
    }
}

// ---------------------------------------------------------------------------
// Adaptive bypass detection
// ---------------------------------------------------------------------------
void SectorCache::record_access(uint64_t address) {
    if (recent_addresses_.size() >= max_recent_addresses_) {
        recent_addresses_.erase(recent_addresses_.begin());
    }
    recent_addresses_.push_back(address);
}

bool SectorCache::is_streaming_access(uint64_t /*address*/) const {
    if (recent_addresses_.size() < 4) return false;

    // Check if recent accesses form a sequential (streaming) pattern
    size_t sequential_count = 0;
    for (size_t i = recent_addresses_.size() - 1; i > 0 && i > recent_addresses_.size() - 5; --i) {
        int64_t diff = static_cast<int64_t>(recent_addresses_[i]) -
                       static_cast<int64_t>(recent_addresses_[i - 1]);
        if (diff > 0 && static_cast<uint64_t>(diff) <= line_size_ * 2) {
            sequential_count++;
        }
    }
    return sequential_count >= 3;
}

BypassHint SectorCache::get_adaptive_hint(uint64_t address) const {
    if (is_streaming_access(address)) {
        return BypassHint::STREAMING;
    }
    return BypassHint::NO_BYPASS;
}

// ---------------------------------------------------------------------------
// Statistics
// ---------------------------------------------------------------------------
SectorCache::SectorCacheStats SectorCache::get_statistics() const {
    SectorCacheStats s = stats_;
    uint64_t total = s.hits + s.misses;
    s.hit_rate = (total > 0) ? static_cast<double>(s.hits) / total : 0.0;

    // Estimate bandwidth savings: sector fetch vs. full-line fetch
    uint64_t sector_fetches = s.sector_misses; // Only fetched needed sectors
    uint64_t full_line_bytes = sector_fetches * line_size_;
    uint64_t sector_bytes = sector_fetches * sector_size_;
    if (full_line_bytes > 0) {
        s.bandwidth_saved_pct = 1.0 - static_cast<double>(sector_bytes) /
                                       static_cast<double>(full_line_bytes);
    }
    return s;
}

void SectorCache::reset_statistics() {
    stats_ = SectorCacheStats{};
}

} // namespace gpu_sim
