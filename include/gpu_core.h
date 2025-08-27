#pragma once

#include <vector>
#include <memory>
#include <cstdint>

namespace gpu_sim {

// Forward declarations
class MemoryHierarchy;
class PerformanceMonitor;

/**
 * Represents a single GPU shader core that can execute shader programs
 */
class ShaderCore {
public:
    explicit ShaderCore(uint32_t core_id);
    ~ShaderCore() = default;

    // Core operations
    void execute_instruction(const std::vector<uint32_t>& instruction);
    bool is_busy() const { return busy_; }
    uint32_t get_core_id() const { return core_id_; }
    
    // Performance metrics
    uint64_t get_instruction_count() const { return instruction_count_; }
    uint64_t get_cycle_count() const { return cycle_count_; }

private:
    uint32_t core_id_;
    bool busy_;
    uint64_t instruction_count_;
    uint64_t cycle_count_;
    std::vector<float> registers_;
};

/**
 * Main GPU core that manages multiple shader cores and coordinates execution
 */
class GPUCore {
public:
    explicit GPUCore(uint32_t num_shader_cores = 32);
    ~GPUCore() = default;

    // Core management
    void initialize(std::shared_ptr<MemoryHierarchy> memory, 
                   std::shared_ptr<PerformanceMonitor> perf_monitor);
    
    // Execution control
    void dispatch_compute(const std::vector<uint32_t>& program, 
                         uint32_t num_threads);
    void wait_for_completion();
    
    // Status and metrics
    bool is_idle() const;
    uint32_t get_active_cores() const;
    
    // Getters
    const std::vector<std::unique_ptr<ShaderCore>>& get_shader_cores() const {
        return shader_cores_;
    }

private:
    std::vector<std::unique_ptr<ShaderCore>> shader_cores_;
    std::shared_ptr<MemoryHierarchy> memory_;
    std::shared_ptr<PerformanceMonitor> perf_monitor_;
    uint32_t num_cores_;
    bool initialized_;
};

} // namespace gpu_sim