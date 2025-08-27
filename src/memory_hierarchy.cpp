#include "memory_hierarchy.h"
#include <algorithm>
#include <cstring>
#include <iostream>

namespace gpu_sim {

// GPUCache implementation
GPUCache::GPUCache(size_t cache_size, size_t line_size, size_t associativity)
    : cache_size_(cache_size), line_size_(line_size), associativity_(associativity),
      hit_count_(0), miss_count_(0), access_count_(0) {
    
    num_sets_ = cache_size_ / (line_size_ * associativity_);
    cache_sets_.resize(num_sets_);
    
    for (auto& set : cache_sets_) {
        set.resize(associativity_);
    }
}

size_t GPUCache::get_set_index(uint64_t address) const {
    return (address / line_size_) % num_sets_;
}

uint64_t GPUCache::get_tag(uint64_t address) const {
    return address / (line_size_ * num_sets_);
}

CacheLine* GPUCache::find_line(uint64_t address) {
    size_t set_index = get_set_index(address);
    uint64_t tag = get_tag(address);
    
    auto& set = cache_sets_[set_index];
    for (auto& line : set) {
        if (line && line->valid && line->address == (address & ~(line_size_ - 1))) {
            line->access_time = access_count_;
            return line.get();
        }
    }
    return nullptr;
}

CacheLine* GPUCache::allocate_line(uint64_t address) {
    size_t set_index = get_set_index(address);
    auto& set = cache_sets_[set_index];
    
    // Find empty slot first
    for (auto& line : set) {
        if (!line || !line->valid) {
            line = std::make_unique<CacheLine>(address & ~(line_size_ - 1), line_size_);
            line->valid = true;
            line->access_time = access_count_;
            return line.get();
        }
    }
    
    // No empty slot, evict LRU
    auto lru_it = std::min_element(set.begin(), set.end(),
        [](const auto& a, const auto& b) {
            if (!a || !a->valid) return true;
            if (!b || !b->valid) return false;
            return a->access_time < b->access_time;
        });
    
    *lru_it = std::make_unique<CacheLine>(address & ~(line_size_ - 1), line_size_);
    (*lru_it)->valid = true;
    (*lru_it)->access_time = access_count_;
    return lru_it->get();
}

bool GPUCache::read(uint64_t address, void* data, size_t size) {
    access_count_++;
    
    CacheLine* line = find_line(address);
    if (line) {
        // Cache hit
        hit_count_++;
        size_t offset = address % line_size_;
        size_t copy_size = std::min(size, line_size_ - offset);
        std::memcpy(data, line->data.data() + offset, copy_size);
        return true;
    }
    
    // Cache miss
    miss_count_++;
    return false;
}

bool GPUCache::write(uint64_t address, const void* data, size_t size) {
    access_count_++;
    
    CacheLine* line = find_line(address);
    if (!line) {
        // Allocate new line on write miss
        line = allocate_line(address);
        miss_count_++;
    } else {
        hit_count_++;
    }
    
    size_t offset = address % line_size_;
    size_t copy_size = std::min(size, line_size_ - offset);
    std::memcpy(line->data.data() + offset, data, copy_size);
    line->dirty = true;
    
    return true;
}

void GPUCache::invalidate(uint64_t address) {
    CacheLine* line = find_line(address);
    if (line) {
        line->valid = false;
        line->dirty = false;
    }
}

void GPUCache::flush() {
    for (auto& set : cache_sets_) {
        for (auto& line : set) {
            if (line) {
                line->valid = false;
                line->dirty = false;
            }
        }
    }
}

double GPUCache::get_hit_rate() const {
    if (access_count_ == 0) return 0.0;
    return static_cast<double>(hit_count_) / access_count_;
}

// MemoryHierarchy implementation
MemoryHierarchy::MemoryHierarchy()
    : next_allocation_address_(0x10000000) { // Start allocations at 256MB
    
    // Initialize L1 cache: 32KB, 64-byte lines, 4-way associative
    l1_cache_ = std::make_unique<GPUCache>(32 * 1024, 64, 4);
    
    // Initialize L2 cache: 512KB, 128-byte lines, 8-way associative
    l2_cache_ = std::make_unique<GPUCache>(512 * 1024, 128, 8);
    
    // Initialize VRAM: 4GB simulation
    vram_.resize(4ULL * 1024 * 1024 * 1024);
}

bool MemoryHierarchy::read(uint64_t address, void* data, size_t size) {
    // Try L1 cache first
    if (l1_cache_->read(address, data, size)) {
        return true;
    }
    
    // Try L2 cache
    if (l2_cache_->read(address, data, size)) {
        // Load into L1
        l1_cache_->write(address, data, size);
        return true;
    }
    
    // Read from VRAM
    if (address + size <= vram_.size()) {
        std::memcpy(data, vram_.data() + address, size);
        
        // Load into caches
        l2_cache_->write(address, data, size);
        l1_cache_->write(address, data, size);
        return true;
    }
    
    return false;
}

bool MemoryHierarchy::write(uint64_t address, const void* data, size_t size) {
    // Write through to all levels
    l1_cache_->write(address, data, size);
    l2_cache_->write(address, data, size);
    
    // Write to VRAM
    if (address + size <= vram_.size()) {
        std::memcpy(vram_.data() + address, data, size);
        return true;
    }
    
    return false;
}

uint64_t MemoryHierarchy::allocate(size_t size) {
    uint64_t address = next_allocation_address_;
    
    // Align to 16-byte boundary
    size = (size + 15) & ~15ULL;
    
    if (address + size > vram_.size()) {
        return 0; // Out of memory
    }
    
    allocations_[address] = size;
    next_allocation_address_ += size;
    
    return address;
}

void MemoryHierarchy::deallocate(uint64_t address) {
    auto it = allocations_.find(address);
    if (it != allocations_.end()) {
        // Invalidate cache lines for this allocation
        size_t size = it->second;
        for (uint64_t addr = address; addr < address + size; addr += 64) {
            l1_cache_->invalidate(addr);
            l2_cache_->invalidate(addr);
        }
        allocations_.erase(it);
    }
}

void MemoryHierarchy::flush_all_caches() {
    l1_cache_->flush();
    l2_cache_->flush();
}

MemoryHierarchy::MemoryStats MemoryHierarchy::get_statistics() const {
    MemoryStats stats;
    stats.l1_hits = l1_cache_->get_hit_count();
    stats.l1_misses = l1_cache_->get_miss_count();
    stats.l2_hits = l2_cache_->get_hit_count();
    stats.l2_misses = l2_cache_->get_miss_count();
    stats.vram_accesses = stats.l2_misses; // VRAM accessed on L2 miss
    
    // Calculate average access latency
    uint64_t total_accesses = stats.l1_hits + stats.l1_misses;
    if (total_accesses > 0) {
        double l1_contribution = (static_cast<double>(stats.l1_hits) / total_accesses) * L1_LATENCY;
        double l2_contribution = (static_cast<double>(stats.l2_hits) / total_accesses) * L2_LATENCY;
        double vram_contribution = (static_cast<double>(stats.vram_accesses) / total_accesses) * VRAM_LATENCY;
        stats.avg_access_latency = l1_contribution + l2_contribution + vram_contribution;
    } else {
        stats.avg_access_latency = 0.0;
    }
    
    return stats;
}

} // namespace gpu_sim