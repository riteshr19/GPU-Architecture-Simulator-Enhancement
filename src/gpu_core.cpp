#include "gpu_core.h"
#include "memory_hierarchy.h"
#include "performance_monitor.h"
#include <algorithm>
#include <thread>
#include <iostream>

namespace gpu_sim {

// ShaderCore implementation
ShaderCore::ShaderCore(uint32_t core_id)
    : core_id_(core_id), busy_(false), instruction_count_(0), 
      cycle_count_(0), registers_(32, 0.0f) {
}

void ShaderCore::execute_instruction(const std::vector<uint32_t>& instruction) {
    if (instruction.empty()) return;
    
    busy_ = true;
    
    // Simulate instruction execution
    uint32_t opcode = instruction[0];
    
    switch (opcode) {
        case 0x01: // ADD
            if (instruction.size() >= 4) {
                uint32_t dst = instruction[1];
                uint32_t src1 = instruction[2];
                uint32_t src2 = instruction[3];
                if (dst < registers_.size() && src1 < registers_.size() && src2 < registers_.size()) {
                    registers_[dst] = registers_[src1] + registers_[src2];
                }
            }
            break;
            
        case 0x02: // MUL
            if (instruction.size() >= 4) {
                uint32_t dst = instruction[1];
                uint32_t src1 = instruction[2];
                uint32_t src2 = instruction[3];
                if (dst < registers_.size() && src1 < registers_.size() && src2 < registers_.size()) {
                    registers_[dst] = registers_[src1] * registers_[src2];
                }
            }
            break;
            
        case 0x03: // LOAD
            // Simulate memory load (would interact with memory hierarchy)
            cycle_count_ += 10; // Memory access penalty
            break;
            
        case 0x04: // STORE
            // Simulate memory store
            cycle_count_ += 5;
            break;
            
        default:
            // Unknown instruction
            break;
    }
    
    instruction_count_++;
    cycle_count_++;
    busy_ = false;
}

// GPUCore implementation
GPUCore::GPUCore(uint32_t num_shader_cores)
    : num_cores_(num_shader_cores), initialized_(false) {
    
    shader_cores_.reserve(num_cores_);
    for (uint32_t i = 0; i < num_cores_; ++i) {
        shader_cores_.emplace_back(std::make_unique<ShaderCore>(i));
    }
}

void GPUCore::initialize(std::shared_ptr<MemoryHierarchy> memory,
                        std::shared_ptr<PerformanceMonitor> perf_monitor) {
    memory_ = memory;
    perf_monitor_ = perf_monitor;
    initialized_ = true;
    
    if (perf_monitor_) {
        perf_monitor_->set_counter("gpu_cores_total", num_cores_);
    }
}

void GPUCore::dispatch_compute(const std::vector<uint32_t>& program, uint32_t num_threads) {
    if (!initialized_) {
        std::cerr << "GPU core not initialized!" << std::endl;
        return;
    }
    
    if (perf_monitor_) {
        perf_monitor_->start_timer("compute_dispatch");
        perf_monitor_->increment_counter("dispatched_threads", num_threads);
    }
    
    // Distribute threads across available cores
    uint32_t threads_per_core = (num_threads + num_cores_ - 1) / num_cores_;
    
    for (uint32_t core_idx = 0; core_idx < num_cores_; ++core_idx) {
        uint32_t start_thread = core_idx * threads_per_core;
        uint32_t end_thread = std::min(start_thread + threads_per_core, num_threads);
        
        if (start_thread >= num_threads) break;
        
        auto& core = shader_cores_[core_idx];
        
        // Execute program on this core for assigned threads
        for (uint32_t thread = start_thread; thread < end_thread; ++thread) {
            // Simple instruction execution simulation
            for (size_t i = 0; i < program.size(); i += 4) {
                std::vector<uint32_t> instruction;
                for (size_t j = 0; j < 4 && (i + j) < program.size(); ++j) {
                    instruction.push_back(program[i + j]);
                }
                core->execute_instruction(instruction);
            }
        }
    }
    
    if (perf_monitor_) {
        perf_monitor_->end_timer("compute_dispatch");
    }
}

void GPUCore::wait_for_completion() {
    // In a real implementation, this would wait for all cores to finish
    // For simulation, we assume synchronous execution
    
    if (perf_monitor_) {
        perf_monitor_->increment_counter("wait_for_completion_calls");
    }
    
    // Simulate some wait time
    std::this_thread::sleep_for(std::chrono::microseconds(100));
}

bool GPUCore::is_idle() const {
    return std::all_of(shader_cores_.begin(), shader_cores_.end(),
                      [](const auto& core) { return !core->is_busy(); });
}

uint32_t GPUCore::get_active_cores() const {
    return static_cast<uint32_t>(
        std::count_if(shader_cores_.begin(), shader_cores_.end(),
                     [](const auto& core) { return core->is_busy(); })
    );
}

} // namespace gpu_sim