#pragma once

#include <vector>
#include <memory>
#include <functional>
#include <cstdint>

namespace gpu_sim {

// Forward declarations
class GPUCore;
class MemoryHierarchy;
class TextureCache;
class PerformanceMonitor;

/**
 * Vertex data structure
 */
struct Vertex {
    float position[4];  // x, y, z, w
    float color[4];     // r, g, b, a
    float texcoord[2];  // u, v
    float normal[3];    // nx, ny, nz
};

/**
 * Fragment data structure
 */
struct Fragment {
    float position[4];
    float color[4];
    float texcoord[2];
    float depth;
    bool valid;
};

/**
 * Texture data structure
 */
struct Texture {
    uint32_t width, height;
    uint32_t format;
    std::vector<uint8_t> data;
    uint32_t mip_levels;
};

/**
 * Pipeline state configuration
 */
struct PipelineState {
    bool depth_test_enabled;
    bool blending_enabled;
    bool culling_enabled;
    uint32_t viewport_width;
    uint32_t viewport_height;
};

/**
 * Graphics pipeline implementation with stages
 */
class GraphicsPipeline {
public:
    GraphicsPipeline();
    ~GraphicsPipeline() = default;

    // Initialization
    void initialize(std::shared_ptr<GPUCore> gpu_core,
                   std::shared_ptr<MemoryHierarchy> memory,
                   std::shared_ptr<TextureCache> texture_cache,
                   std::shared_ptr<PerformanceMonitor> perf_monitor);

    // Pipeline configuration
    void set_pipeline_state(const PipelineState& state);
    void bind_texture(uint32_t unit, const Texture& texture);

    // Shader program management
    void set_vertex_shader(std::function<Vertex(const Vertex&)> shader);
    void set_fragment_shader(std::function<Fragment(const Fragment&)> shader);

    // Rendering operations
    void draw_triangles(const std::vector<Vertex>& vertices);
    void draw_indexed(const std::vector<Vertex>& vertices, 
                     const std::vector<uint32_t>& indices);

    // Frame management
    void begin_frame();
    void end_frame();
    void present();

    // Performance metrics
    struct PipelineStats {
        uint64_t vertices_processed;
        uint64_t fragments_processed;
        uint64_t triangles_drawn;
        uint64_t texture_samples;
        double frame_time_ms;
    };

    PipelineStats get_statistics() const;

private:
    // Pipeline stages
    std::vector<Vertex> vertex_stage(const std::vector<Vertex>& input_vertices);
    std::vector<Fragment> rasterization_stage(const std::vector<Vertex>& vertices);
    std::vector<Fragment> fragment_stage(const std::vector<Fragment>& fragments);
    void output_merger_stage(const std::vector<Fragment>& fragments);

    // Helper functions
    bool is_triangle_culled(const Vertex& v0, const Vertex& v1, const Vertex& v2);
    Fragment interpolate_fragment(const Vertex& v0, const Vertex& v1, const Vertex& v2,
                                 float u, float v, float w);

    // Components
    std::shared_ptr<GPUCore> gpu_core_;
    std::shared_ptr<MemoryHierarchy> memory_;
    std::shared_ptr<TextureCache> texture_cache_;
    std::shared_ptr<PerformanceMonitor> perf_monitor_;

    // State
    PipelineState pipeline_state_;
    std::vector<Texture> bound_textures_;
    
    // Shaders
    std::function<Vertex(const Vertex&)> vertex_shader_;
    std::function<Fragment(const Fragment&)> fragment_shader_;

    // Frame buffer
    std::vector<uint32_t> color_buffer_;
    std::vector<float> depth_buffer_;
    
    // Statistics
    mutable PipelineStats stats_;
    uint64_t frame_start_time_;
};

} // namespace gpu_sim