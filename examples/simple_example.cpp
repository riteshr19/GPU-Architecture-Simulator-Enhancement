#include "gpu_core.h"
#include "memory_hierarchy.h"
#include "graphics_pipeline.h"
#include "texture_cache.h"
#include "performance_monitor.h"
#include <iostream>
#include <vector>

using namespace gpu_sim;

int main() {
    std::cout << "Simple GPU Simulator Example" << std::endl;
    std::cout << "============================" << std::endl;
    
    // Initialize core components
    auto perf_monitor = std::make_shared<PerformanceMonitor>();
    auto memory = std::make_shared<MemoryHierarchy>();
    auto gpu_core = std::make_shared<GPUCore>(16);  // 16 shader cores
    auto texture_cache = std::make_shared<TextureCache>(64);  // 64MB cache
    auto pipeline = std::make_shared<GraphicsPipeline>();
    
    // Initialize system
    gpu_core->initialize(memory, perf_monitor);
    texture_cache->initialize(memory, perf_monitor);
    pipeline->initialize(gpu_core, memory, texture_cache, perf_monitor);
    
    // Enable advanced texture cache features
    texture_cache->enable_smart_prefetching(true);
    texture_cache->enable_adaptive_caching(true);
    
    std::cout << "✓ GPU simulator initialized" << std::endl;
    std::cout << "✓ Advanced texture cache enabled" << std::endl;
    
    // Create simple triangle
    std::vector<Vertex> triangle = {
        {{-0.5f, -0.5f, 0.0f, 1.0f}, {1.0f, 0.0f, 0.0f, 1.0f}, {0.0f, 0.0f}, {0.0f, 0.0f, 1.0f}},
        {{ 0.5f, -0.5f, 0.0f, 1.0f}, {0.0f, 1.0f, 0.0f, 1.0f}, {1.0f, 0.0f}, {0.0f, 0.0f, 1.0f}},
        {{ 0.0f,  0.5f, 0.0f, 1.0f}, {0.0f, 0.0f, 1.0f, 1.0f}, {0.5f, 1.0f}, {0.0f, 0.0f, 1.0f}}
    };
    
    // Render multiple frames to demonstrate cache performance
    std::cout << "\nRendering frames to demonstrate texture cache performance..." << std::endl;
    
    for (int frame = 0; frame < 10; ++frame) {
        pipeline->begin_frame();
        pipeline->draw_triangles(triangle);
        pipeline->end_frame();
        pipeline->present();
        
        if (frame % 3 == 0) {
            std::cout << "  Frame " << frame + 1 << " rendered" << std::endl;
        }
    }
    
    // Display performance metrics
    auto pipeline_stats = pipeline->get_statistics();
    auto cache_metrics = texture_cache->get_metrics();
    auto memory_stats = memory->get_statistics();
    
    std::cout << "\nPerformance Results:" << std::endl;
    std::cout << "===================" << std::endl;
    std::cout << "Graphics Pipeline:" << std::endl;
    std::cout << "  Vertices processed: " << pipeline_stats.vertices_processed << std::endl;
    std::cout << "  Triangles drawn: " << pipeline_stats.triangles_drawn << std::endl;
    std::cout << "  Texture samples: " << pipeline_stats.texture_samples << std::endl;
    std::cout << "  Average frame time: " << pipeline_stats.frame_time_ms << " ms" << std::endl;
    
    std::cout << "\nAdvanced Texture Cache (NEW FEATURE):" << std::endl;
    std::cout << "  Cache hit rate: " << (cache_metrics.hit_rate * 100.0) << "%" << std::endl;
    std::cout << "  Prefetch efficiency: " << (cache_metrics.prefetch_efficiency * 100.0) << "%" << std::endl;
    std::cout << "  Cache utilization: " << cache_metrics.cache_utilization_percent << "%" << std::endl;
    std::cout << "  Data transferred: " << (cache_metrics.bytes_transferred / 1024) << " KB" << std::endl;
    
    std::cout << "\nMemory Hierarchy:" << std::endl;
    std::cout << "  L1 Cache hit rate: " << 
        (static_cast<double>(memory_stats.l1_hits) / (memory_stats.l1_hits + memory_stats.l1_misses) * 100.0) << "%" << std::endl;
    std::cout << "  L2 Cache hit rate: " << 
        (static_cast<double>(memory_stats.l2_hits) / (memory_stats.l2_hits + memory_stats.l2_misses) * 100.0) << "%" << std::endl;
    std::cout << "  Average access latency: " << memory_stats.avg_access_latency << " cycles" << std::endl;
    
    std::cout << "\n🎉 Example completed successfully!" << std::endl;
    std::cout << "The GPU simulator demonstrates:" << std::endl;
    std::cout << "✓ Complete graphics pipeline execution" << std::endl;
    std::cout << "✓ Advanced texture cache with smart prefetching" << std::endl;
    std::cout << "✓ Multi-level memory hierarchy simulation" << std::endl;
    std::cout << "✓ Comprehensive performance monitoring" << std::endl;
    
    return 0;
}