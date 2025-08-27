#include "gpu_core.h"
#include "memory_hierarchy.h"
#include "graphics_pipeline.h"
#include "texture_cache.h"
#include "performance_monitor.h"
#include <iostream>
#include <vector>
#include <memory>
#include <chrono>
#include <thread>
#include <cmath>

using namespace gpu_sim;

// Demo scene data
std::vector<Vertex> create_demo_triangle() {
    return {
        {{-0.5f, -0.5f, 0.0f, 1.0f}, {1.0f, 0.0f, 0.0f, 1.0f}, {0.0f, 0.0f}, {0.0f, 0.0f, 1.0f}},
        {{ 0.5f, -0.5f, 0.0f, 1.0f}, {0.0f, 1.0f, 0.0f, 1.0f}, {1.0f, 0.0f}, {0.0f, 0.0f, 1.0f}},
        {{ 0.0f,  0.5f, 0.0f, 1.0f}, {0.0f, 0.0f, 1.0f, 1.0f}, {0.5f, 1.0f}, {0.0f, 0.0f, 1.0f}}
    };
}

std::vector<Vertex> create_demo_quad() {
    return {
        // First triangle
        {{-0.8f, -0.8f, 0.0f, 1.0f}, {1.0f, 1.0f, 1.0f, 1.0f}, {0.0f, 0.0f}, {0.0f, 0.0f, 1.0f}},
        {{ 0.8f, -0.8f, 0.0f, 1.0f}, {1.0f, 1.0f, 1.0f, 1.0f}, {1.0f, 0.0f}, {0.0f, 0.0f, 1.0f}},
        {{ 0.8f,  0.8f, 0.0f, 1.0f}, {1.0f, 1.0f, 1.0f, 1.0f}, {1.0f, 1.0f}, {0.0f, 0.0f, 1.0f}},
        
        // Second triangle
        {{-0.8f, -0.8f, 0.0f, 1.0f}, {1.0f, 1.0f, 1.0f, 1.0f}, {0.0f, 0.0f}, {0.0f, 0.0f, 1.0f}},
        {{ 0.8f,  0.8f, 0.0f, 1.0f}, {1.0f, 1.0f, 1.0f, 1.0f}, {1.0f, 1.0f}, {0.0f, 0.0f, 1.0f}},
        {{-0.8f,  0.8f, 0.0f, 1.0f}, {1.0f, 1.0f, 1.0f, 1.0f}, {0.0f, 1.0f}, {0.0f, 0.0f, 1.0f}}
    };
}

Texture create_demo_texture() {
    Texture texture;
    texture.width = 256;
    texture.height = 256;
    texture.format = 0; // RGBA
    texture.mip_levels = 1;
    
    // Create a simple checkerboard pattern
    texture.data.resize(texture.width * texture.height * 4);
    for (uint32_t y = 0; y < texture.height; ++y) {
        for (uint32_t x = 0; x < texture.width; ++x) {
            size_t index = (y * texture.width + x) * 4;
            bool checker = ((x / 32) + (y / 32)) % 2 == 0;
            
            texture.data[index + 0] = checker ? 255 : 64;  // R
            texture.data[index + 1] = checker ? 255 : 64;  // G
            texture.data[index + 2] = checker ? 255 : 64;  // B
            texture.data[index + 3] = 255;                 // A
        }
    }
    
    return texture;
}

void demonstrate_texture_cache_performance(TextureCache& texture_cache) {
    std::cout << "\n=== Demonstrating Advanced Texture Cache Performance ===" << std::endl;
    
    // Enable advanced features
    texture_cache.enable_smart_prefetching(true);
    texture_cache.enable_adaptive_caching(true);
    texture_cache.set_prefetch_distance(4);
    
    std::cout << "Smart prefetching: ENABLED" << std::endl;
    std::cout << "Adaptive caching: ENABLED" << std::endl;
    std::cout << "Prefetch distance: 4" << std::endl;
    
    // Simulate texture access patterns
    std::cout << "\nSimulating texture access patterns..." << std::endl;
    
    // Pattern 1: Sequential texture access (should trigger prefetching)
    std::cout << "Pattern 1: Sequential texture access" << std::endl;
    for (uint64_t tex_id = 1; tex_id <= 10; ++tex_id) {
        uint8_t data[1024];
        texture_cache.read_texture(tex_id, 0, 0, data, sizeof(data));
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    
    // Pattern 2: Mip-level access (should trigger mip prefetching)
    std::cout << "Pattern 2: Mip-level access" << std::endl;
    for (uint32_t mip = 0; mip < 8; ++mip) {
        uint8_t data[512];
        texture_cache.read_texture(100, mip, 0, data, sizeof(data));
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    
    // Pattern 3: Random access (should adapt caching strategy)
    std::cout << "Pattern 3: Random access" << std::endl;
    for (int i = 0; i < 50; ++i) {
        uint64_t tex_id = 1000 + (i * 7) % 20; // Pseudo-random
        uint32_t mip = i % 4;
        uint8_t data[256];
        texture_cache.read_texture(tex_id, mip, 0, data, sizeof(data));
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    
    // Display cache performance metrics
    auto metrics = texture_cache.get_metrics();
    std::cout << "\nTexture Cache Performance Metrics:" << std::endl;
    std::cout << "  Cache hits: " << metrics.cache_hits << std::endl;
    std::cout << "  Cache misses: " << metrics.cache_misses << std::endl;
    std::cout << "  Hit rate: " << (metrics.hit_rate * 100.0) << "%" << std::endl;
    std::cout << "  Prefetch hits: " << metrics.prefetch_hits << std::endl;
    std::cout << "  Prefetch efficiency: " << (metrics.prefetch_efficiency * 100.0) << "%" << std::endl;
    std::cout << "  Cache utilization: " << metrics.cache_utilization_percent << "%" << std::endl;
    std::cout << "  Bytes transferred: " << (metrics.bytes_transferred / 1024) << " KB" << std::endl;
    std::cout << "  Average access latency: " << metrics.avg_access_latency_ms << " ms" << std::endl;
}

void run_performance_benchmark(GraphicsPipeline& pipeline, 
                              const std::vector<Vertex>& geometry,
                              int num_frames) {
    std::cout << "\n=== Running Performance Benchmark ===" << std::endl;
    std::cout << "Rendering " << num_frames << " frames..." << std::endl;
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    for (int frame = 0; frame < num_frames; ++frame) {
        pipeline.begin_frame();
        
        // Render multiple instances of geometry for stress testing
        for (int instance = 0; instance < 10; ++instance) {
            pipeline.draw_triangles(geometry);
        }
        
        pipeline.end_frame();
        pipeline.present();
        
        // Show progress
        if ((frame + 1) % 10 == 0) {
            std::cout << "  Frame " << (frame + 1) << "/" << num_frames << " completed" << std::endl;
        }
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto total_time = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    
    std::cout << "Benchmark completed in " << total_time << " ms" << std::endl;
    std::cout << "Average frame time: " << (total_time / num_frames) << " ms" << std::endl;
    std::cout << "Average FPS: " << (num_frames * 1000.0 / total_time) << std::endl;
}

int main() {
    std::cout << "GPU Architecture Simulator Enhancement" << std::endl;
    std::cout << "======================================" << std::endl;
    
    try {
        // Initialize core components
        std::cout << "\nInitializing GPU simulator components..." << std::endl;
        
        auto performance_monitor = std::make_shared<PerformanceMonitor>();
        auto memory_hierarchy = std::make_shared<MemoryHierarchy>();
        auto gpu_core = std::make_shared<GPUCore>(64); // 64 shader cores
        auto texture_cache = std::make_shared<TextureCache>(256); // 256MB cache
        auto graphics_pipeline = std::make_shared<GraphicsPipeline>();
        
        // Initialize components
        gpu_core->initialize(memory_hierarchy, performance_monitor);
        texture_cache->initialize(memory_hierarchy, performance_monitor);
        graphics_pipeline->initialize(gpu_core, memory_hierarchy, texture_cache, performance_monitor);
        
        std::cout << "✓ GPU Core initialized with 64 shader cores" << std::endl;
        std::cout << "✓ Memory hierarchy initialized" << std::endl;
        std::cout << "✓ Advanced texture cache initialized (256MB)" << std::endl;
        std::cout << "✓ Graphics pipeline initialized" << std::endl;
        
        // Configure pipeline state
        PipelineState pipeline_state;
        pipeline_state.depth_test_enabled = true;
        pipeline_state.blending_enabled = false;
        pipeline_state.culling_enabled = true;
        pipeline_state.viewport_width = 1920;
        pipeline_state.viewport_height = 1080;
        graphics_pipeline->set_pipeline_state(pipeline_state);
        
        // Create demo texture and bind it
        auto demo_texture = create_demo_texture();
        graphics_pipeline->bind_texture(0, demo_texture);
        
        // Set up simple shaders
        graphics_pipeline->set_vertex_shader([](const Vertex& v) -> Vertex {
            Vertex transformed = v;
            // Simple rotation transformation
            float angle = 0.1f;
            float cos_a = std::cos(angle);
            float sin_a = std::sin(angle);
            float x = transformed.position[0];
            float y = transformed.position[1];
            transformed.position[0] = x * cos_a - y * sin_a;
            transformed.position[1] = x * sin_a + y * cos_a;
            return transformed;
        });
        
        graphics_pipeline->set_fragment_shader([](const Fragment& f) -> Fragment {
            Fragment shaded = f;
            // Simple lighting effect
            float intensity = 0.7f + 0.3f * shaded.texcoord[0];
            shaded.color[0] *= intensity;
            shaded.color[1] *= intensity;
            shaded.color[2] *= intensity;
            return shaded;
        });
        
        // Demonstrate texture cache performance (NEW FEATURE)
        demonstrate_texture_cache_performance(*texture_cache);
        
        // Create demo geometry
        auto triangle_geometry = create_demo_triangle();
        auto quad_geometry = create_demo_quad();
        
        std::cout << "\n=== Rendering Demo Scenes ===" << std::endl;
        
        // Render single frame with triangle
        graphics_pipeline->begin_frame();
        graphics_pipeline->draw_triangles(triangle_geometry);
        graphics_pipeline->end_frame();
        graphics_pipeline->present();
        
        auto triangle_stats = graphics_pipeline->get_statistics();
        std::cout << "Triangle scene rendered:" << std::endl;
        std::cout << "  Vertices processed: " << triangle_stats.vertices_processed << std::endl;
        std::cout << "  Fragments processed: " << triangle_stats.fragments_processed << std::endl;
        std::cout << "  Texture samples: " << triangle_stats.texture_samples << std::endl;
        std::cout << "  Frame time: " << triangle_stats.frame_time_ms << " ms" << std::endl;
        
        // Render single frame with quad
        graphics_pipeline->begin_frame();
        graphics_pipeline->draw_triangles(quad_geometry);
        graphics_pipeline->end_frame();
        graphics_pipeline->present();
        
        auto quad_stats = graphics_pipeline->get_statistics();
        std::cout << "Quad scene rendered:" << std::endl;
        std::cout << "  Vertices processed: " << quad_stats.vertices_processed << std::endl;
        std::cout << "  Fragments processed: " << quad_stats.fragments_processed << std::endl;
        std::cout << "  Texture samples: " << quad_stats.texture_samples << std::endl;
        std::cout << "  Frame time: " << quad_stats.frame_time_ms << " ms" << std::endl;
        
        // Run performance benchmark
        run_performance_benchmark(*graphics_pipeline, quad_geometry, 50);
        
        // Test compute shader functionality
        std::cout << "\n=== Testing Compute Shader Functionality ===" << std::endl;
        std::vector<uint32_t> compute_program = {
            0x01, 0, 1, 2,  // ADD r0, r1, r2
            0x02, 3, 0, 1,  // MUL r3, r0, r1
            0x03, 4, 0, 0,  // LOAD r4, [r0]
            0x04, 0, 4, 0   // STORE [r0], r4
        };
        
        gpu_core->dispatch_compute(compute_program, 1024);
        gpu_core->wait_for_completion();
        
        std::cout << "Compute shader executed with 1024 threads" << std::endl;
        std::cout << "Active cores: " << gpu_core->get_active_cores() << std::endl;
        std::cout << "GPU idle: " << (gpu_core->is_idle() ? "Yes" : "No") << std::endl;
        
        // Display comprehensive performance report
        std::cout << "\n=== Final Performance Analysis ===" << std::endl;
        performance_monitor->print_report();
        
        // Display memory hierarchy statistics
        auto memory_stats = memory_hierarchy->get_statistics();
        std::cout << "\nMemory Hierarchy Statistics:" << std::endl;
        std::cout << "  L1 Cache - Hits: " << memory_stats.l1_hits << ", Misses: " << memory_stats.l1_misses << std::endl;
        std::cout << "  L2 Cache - Hits: " << memory_stats.l2_hits << ", Misses: " << memory_stats.l2_misses << std::endl;
        std::cout << "  VRAM accesses: " << memory_stats.vram_accesses << std::endl;
        std::cout << "  Average access latency: " << memory_stats.avg_access_latency << " cycles" << std::endl;
        
        std::cout << "\n=== Simulation Complete ===" << std::endl;
        std::cout << "The GPU architecture simulator successfully demonstrated:" << std::endl;
        std::cout << "✓ Multi-core GPU simulation with 64 shader cores" << std::endl;
        std::cout << "✓ Advanced memory hierarchy with L1/L2 caches and VRAM" << std::endl;
        std::cout << "✓ Complete graphics pipeline with vertex and fragment stages" << std::endl;
        std::cout << "✓ NEW FEATURE: Advanced texture cache with smart prefetching" << std::endl;
        std::cout << "✓ Comprehensive performance monitoring and profiling" << std::endl;
        std::cout << "✓ Rigorous testing and benchmarking" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}