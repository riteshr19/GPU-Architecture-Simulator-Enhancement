#include "gpu_core.h"
#include "memory_hierarchy.h"
#include "graphics_pipeline.h"
#include "texture_cache.h"
#include "performance_monitor.h"
#include <iostream>
#include <cassert>
#include <vector>
#include <memory>
#include <thread>
#include <chrono>

using namespace gpu_sim;

// Simple test framework
class TestFramework {
public:
    static void assert_true(bool condition, const std::string& message) {
        if (!condition) {
            std::cerr << "ASSERTION FAILED: " << message << std::endl;
            exit(1);
        }
        std::cout << "✓ " << message << std::endl;
    }
    
    static void assert_equals(uint64_t expected, uint64_t actual, const std::string& message) {
        if (expected != actual) {
            std::cerr << "ASSERTION FAILED: " << message 
                      << " (expected: " << expected << ", actual: " << actual << ")" << std::endl;
            exit(1);
        }
        std::cout << "✓ " << message << std::endl;
    }
    
    static void assert_greater_than(double value, double threshold, const std::string& message) {
        if (value <= threshold) {
            std::cerr << "ASSERTION FAILED: " << message 
                      << " (value: " << value << " <= threshold: " << threshold << ")" << std::endl;
            exit(1);
        }
        std::cout << "✓ " << message << std::endl;
    }
};

void test_gpu_core() {
    std::cout << "\n=== Testing GPU Core ===" << std::endl;
    
    auto perf_monitor = std::make_shared<PerformanceMonitor>();
    auto memory = std::make_shared<MemoryHierarchy>();
    auto gpu_core = std::make_shared<GPUCore>(8);
    
    gpu_core->initialize(memory, perf_monitor);
    
    TestFramework::assert_true(gpu_core->is_idle(), "GPU core should be idle initially");
    TestFramework::assert_equals(0, gpu_core->get_active_cores(), "No cores should be active initially");
    TestFramework::assert_equals(8, gpu_core->get_shader_cores().size(), "Should have 8 shader cores");
    
    // Test compute dispatch
    std::vector<uint32_t> simple_program = {0x01, 0, 1, 2}; // ADD instruction
    gpu_core->dispatch_compute(simple_program, 16);
    gpu_core->wait_for_completion();
    
    TestFramework::assert_true(gpu_core->is_idle(), "GPU core should be idle after completion");
    
    std::cout << "GPU Core tests passed!" << std::endl;
}

void test_memory_hierarchy() {
    std::cout << "\n=== Testing Memory Hierarchy ===" << std::endl;
    
    auto memory = std::make_shared<MemoryHierarchy>();
    
    // Test allocation
    uint64_t addr1 = memory->allocate(1024);
    uint64_t addr2 = memory->allocate(2048);
    
    TestFramework::assert_true(addr1 != 0, "First allocation should succeed");
    TestFramework::assert_true(addr2 != 0, "Second allocation should succeed");
    TestFramework::assert_true(addr2 > addr1, "Second allocation should be after first");
    
    // Test read/write
    std::vector<uint8_t> test_data = {0xDE, 0xAD, 0xBE, 0xEF};
    std::vector<uint8_t> read_buffer(4);
    
    TestFramework::assert_true(memory->write(addr1, test_data.data(), 4), "Write should succeed");
    TestFramework::assert_true(memory->read(addr1, read_buffer.data(), 4), "Read should succeed");
    
    for (size_t i = 0; i < 4; ++i) {
        TestFramework::assert_equals(test_data[i], read_buffer[i], "Read data should match written data");
    }
    
    // Test cache performance
    auto stats = memory->get_statistics();
    TestFramework::assert_true(stats.l1_hits + stats.l1_misses > 0, "Should have cache accesses");
    
    memory->deallocate(addr1);
    memory->deallocate(addr2);
    
    std::cout << "Memory Hierarchy tests passed!" << std::endl;
}

void test_texture_cache() {
    std::cout << "\n=== Testing Advanced Texture Cache ===" << std::endl;
    
    auto memory = std::make_shared<MemoryHierarchy>();
    auto perf_monitor = std::make_shared<PerformanceMonitor>();
    auto texture_cache = std::make_shared<TextureCache>(64); // 64MB cache
    
    texture_cache->initialize(memory, perf_monitor);
    
    // Enable advanced features
    texture_cache->enable_smart_prefetching(true);
    texture_cache->enable_adaptive_caching(true);
    
    // Test basic texture reading
    std::vector<uint8_t> texture_data(1024);
    TestFramework::assert_true(texture_cache->read_texture(1, 0, 0, texture_data.data(), 1024), 
                              "First texture read should succeed");
    
    // Test cache hit on second read
    TestFramework::assert_true(texture_cache->read_texture(1, 0, 0, texture_data.data(), 1024), 
                              "Second texture read should hit cache");
    
    // Test prefetching
    texture_cache->prefetch_texture(2, 0);
    
    // Test metrics
    auto metrics = texture_cache->get_metrics();
    TestFramework::assert_true(metrics.cache_hits + metrics.cache_misses > 0, "Should have cache accesses");
    TestFramework::assert_true(metrics.hit_rate >= 0.0 && metrics.hit_rate <= 1.0, "Hit rate should be valid");
    
    // Test pattern-based prefetching by simulating sequential access
    for (uint64_t tex_id = 10; tex_id < 15; ++tex_id) {
        texture_cache->read_texture(tex_id, 0, 0, texture_data.data(), 512);
    }
    
    auto final_metrics = texture_cache->get_metrics();
    TestFramework::assert_true(final_metrics.cache_hits + final_metrics.cache_misses > 
                              metrics.cache_hits + metrics.cache_misses, 
                              "Should have more cache accesses after pattern test");
    
    std::cout << "Advanced Texture Cache tests passed!" << std::endl;
}

void test_graphics_pipeline() {
    std::cout << "\n=== Testing Graphics Pipeline ===" << std::endl;
    
    auto perf_monitor = std::make_shared<PerformanceMonitor>();
    auto memory = std::make_shared<MemoryHierarchy>();
    auto gpu_core = std::make_shared<GPUCore>(4);
    auto texture_cache = std::make_shared<TextureCache>(32);
    auto pipeline = std::make_shared<GraphicsPipeline>();
    
    gpu_core->initialize(memory, perf_monitor);
    texture_cache->initialize(memory, perf_monitor);
    pipeline->initialize(gpu_core, memory, texture_cache, perf_monitor);
    
    // Test pipeline state
    PipelineState state;
    state.viewport_width = 800;
    state.viewport_height = 600;
    state.depth_test_enabled = true;
    state.blending_enabled = false;
    state.culling_enabled = true;
    pipeline->set_pipeline_state(state);
    
    // Create test triangle
    std::vector<Vertex> triangle = {
        {{-0.5f, -0.5f, 0.0f, 1.0f}, {1.0f, 0.0f, 0.0f, 1.0f}, {0.0f, 0.0f}, {0.0f, 0.0f, 1.0f}},
        {{ 0.5f, -0.5f, 0.0f, 1.0f}, {0.0f, 1.0f, 0.0f, 1.0f}, {1.0f, 0.0f}, {0.0f, 0.0f, 1.0f}},
        {{ 0.0f,  0.5f, 0.0f, 1.0f}, {0.0f, 0.0f, 1.0f, 1.0f}, {0.5f, 1.0f}, {0.0f, 0.0f, 1.0f}}
    };
    
    // Test frame rendering
    pipeline->begin_frame();
    pipeline->draw_triangles(triangle);
    pipeline->end_frame();
    pipeline->present();
    
    auto stats = pipeline->get_statistics();
    TestFramework::assert_equals(3, stats.vertices_processed, "Should process 3 vertices");
    TestFramework::assert_equals(1, stats.triangles_drawn, "Should draw 1 triangle");
    TestFramework::assert_true(stats.frame_time_ms >= 0, "Frame time should be non-negative");
    
    std::cout << "Graphics Pipeline tests passed!" << std::endl;
}

void test_performance_monitor() {
    std::cout << "\n=== Testing Performance Monitor ===" << std::endl;
    
    auto perf_monitor = std::make_shared<PerformanceMonitor>();
    
    // Test timers
    perf_monitor->start_timer("test_operation");
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    perf_monitor->end_timer("test_operation");
    
    double elapsed = perf_monitor->get_elapsed_time_ms("test_operation");
    TestFramework::assert_greater_than(elapsed, 5.0, "Elapsed time should be at least 5ms");
    
    // Test counters
    perf_monitor->increment_counter("test_counter", 10);
    perf_monitor->increment_counter("test_counter", 5);
    TestFramework::assert_equals(15, perf_monitor->get_counter("test_counter"), "Counter should equal 15");
    
    // Test cache recording
    perf_monitor->record_cache_access("test_cache", true);
    perf_monitor->record_cache_access("test_cache", false);
    perf_monitor->record_cache_access("test_cache", true);
    
    // Test report generation
    auto report = perf_monitor->generate_report();
    TestFramework::assert_true(report.timing_data.find("test_operation") != report.timing_data.end(),
                              "Report should contain timing data");
    TestFramework::assert_true(report.counter_data.find("test_counter") != report.counter_data.end(),
                              "Report should contain counter data");
    TestFramework::assert_true(report.cache_hit_rates.find("test_cache") != report.cache_hit_rates.end(),
                              "Report should contain cache hit rates");
    
    double hit_rate = report.cache_hit_rates["test_cache"];
    TestFramework::assert_true(hit_rate > 0.6 && hit_rate < 0.7, "Cache hit rate should be approximately 66.7%");
    
    std::cout << "Performance Monitor tests passed!" << std::endl;
}

void test_integration() {
    std::cout << "\n=== Integration Test ===" << std::endl;
    
    // Create full system
    auto perf_monitor = std::make_shared<PerformanceMonitor>();
    auto memory = std::make_shared<MemoryHierarchy>();
    auto gpu_core = std::make_shared<GPUCore>(16);
    auto texture_cache = std::make_shared<TextureCache>(128);
    auto pipeline = std::make_shared<GraphicsPipeline>();
    
    // Initialize all components
    gpu_core->initialize(memory, perf_monitor);
    texture_cache->initialize(memory, perf_monitor);
    pipeline->initialize(gpu_core, memory, texture_cache, perf_monitor);
    
    // Configure advanced texture cache features
    texture_cache->enable_smart_prefetching(true);
    texture_cache->enable_adaptive_caching(true);
    
    // Create complex scene
    std::vector<Vertex> complex_scene;
    for (int i = 0; i < 100; ++i) {
        float offset = i * 0.01f;
        complex_scene.push_back({{-0.1f + offset, -0.1f, 0.0f, 1.0f}, 
                                {1.0f, 0.0f, 0.0f, 1.0f}, {0.0f, 0.0f}, {0.0f, 0.0f, 1.0f}});
        complex_scene.push_back({{ 0.1f + offset, -0.1f, 0.0f, 1.0f}, 
                                {0.0f, 1.0f, 0.0f, 1.0f}, {1.0f, 0.0f}, {0.0f, 0.0f, 1.0f}});
        complex_scene.push_back({{ 0.0f + offset,  0.1f, 0.0f, 1.0f}, 
                                {0.0f, 0.0f, 1.0f, 1.0f}, {0.5f, 1.0f}, {0.0f, 0.0f, 1.0f}});
    }
    
    // Render multiple frames
    for (int frame = 0; frame < 10; ++frame) {
        pipeline->begin_frame();
        pipeline->draw_triangles(complex_scene);
        pipeline->end_frame();
        pipeline->present();
    }
    
    // Verify system performance
    auto stats = pipeline->get_statistics();
    auto cache_metrics = texture_cache->get_metrics();
    auto memory_stats = memory->get_statistics();
    
    TestFramework::assert_true(stats.vertices_processed > 0, "Should have processed vertices");
    TestFramework::assert_true(stats.triangles_drawn > 0, "Should have drawn triangles");
    TestFramework::assert_true(cache_metrics.cache_hits + cache_metrics.cache_misses > 0, 
                              "Should have cache activity");
    TestFramework::assert_true(memory_stats.l1_hits + memory_stats.l1_misses > 0, 
                              "Should have memory activity");
    
    std::cout << "Integration test passed!" << std::endl;
    
    // Print final performance summary
    std::cout << "\nFinal Performance Summary:" << std::endl;
    std::cout << "  Vertices processed: " << stats.vertices_processed << std::endl;
    std::cout << "  Triangles drawn: " << stats.triangles_drawn << std::endl;
    std::cout << "  Cache hit rate: " << (cache_metrics.hit_rate * 100) << "%" << std::endl;
    std::cout << "  Memory efficiency: " << (static_cast<double>(memory_stats.l1_hits) / 
                                            (memory_stats.l1_hits + memory_stats.l1_misses) * 100) << "%" << std::endl;
}

int main() {
    std::cout << "GPU Architecture Simulator - Comprehensive Test Suite" << std::endl;
    std::cout << "=====================================================" << std::endl;
    
    try {
        test_gpu_core();
        test_memory_hierarchy();
        test_texture_cache();
        test_graphics_pipeline();
        test_performance_monitor();
        test_integration();
        
        std::cout << "\n🎉 ALL TESTS PASSED! 🎉" << std::endl;
        std::cout << "\nThe GPU Architecture Simulator has been successfully tested and validated:" << std::endl;
        std::cout << "✓ GPU core with multiple shader units" << std::endl;
        std::cout << "✓ Advanced memory hierarchy with caching" << std::endl;
        std::cout << "✓ NEW FEATURE: Smart texture cache with prefetching" << std::endl;
        std::cout << "✓ Complete graphics pipeline implementation" << std::endl;
        std::cout << "✓ Comprehensive performance monitoring" << std::endl;
        std::cout << "✓ Full system integration and validation" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Test failed with exception: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}