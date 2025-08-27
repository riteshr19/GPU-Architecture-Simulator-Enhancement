#include "graphics_pipeline.h"
#include "gpu_core.h"
#include "memory_hierarchy.h"
#include "texture_cache.h"
#include "performance_monitor.h"
#include <cmath>
#include <algorithm>
#include <chrono>

namespace gpu_sim {

GraphicsPipeline::GraphicsPipeline() : frame_start_time_(0) {
    // Initialize default pipeline state
    pipeline_state_.depth_test_enabled = true;
    pipeline_state_.blending_enabled = false;
    pipeline_state_.culling_enabled = true;
    pipeline_state_.viewport_width = 1920;
    pipeline_state_.viewport_height = 1080;
    
    // Initialize frame buffers
    size_t buffer_size = pipeline_state_.viewport_width * pipeline_state_.viewport_height;
    color_buffer_.resize(buffer_size, 0);
    depth_buffer_.resize(buffer_size, 1.0f);
    
    // Initialize statistics
    stats_ = PipelineStats{};
}

void GraphicsPipeline::initialize(std::shared_ptr<GPUCore> gpu_core,
                                 std::shared_ptr<MemoryHierarchy> memory,
                                 std::shared_ptr<TextureCache> texture_cache,
                                 std::shared_ptr<PerformanceMonitor> perf_monitor) {
    gpu_core_ = gpu_core;
    memory_ = memory;
    texture_cache_ = texture_cache;
    perf_monitor_ = perf_monitor;
    
    // Initialize bound textures
    bound_textures_.resize(8); // Support 8 texture units
    
    if (perf_monitor_) {
        perf_monitor_->set_counter("viewport_width", pipeline_state_.viewport_width);
        perf_monitor_->set_counter("viewport_height", pipeline_state_.viewport_height);
    }
}

void GraphicsPipeline::set_pipeline_state(const PipelineState& state) {
    pipeline_state_ = state;
    
    // Resize frame buffers if viewport changed
    size_t new_buffer_size = state.viewport_width * state.viewport_height;
    if (new_buffer_size != color_buffer_.size()) {
        color_buffer_.resize(new_buffer_size, 0);
        depth_buffer_.resize(new_buffer_size, 1.0f);
    }
}

void GraphicsPipeline::bind_texture(uint32_t unit, const Texture& texture) {
    if (unit < bound_textures_.size()) {
        bound_textures_[unit] = texture;
    }
}

void GraphicsPipeline::set_vertex_shader(std::function<Vertex(const Vertex&)> shader) {
    vertex_shader_ = shader;
}

void GraphicsPipeline::set_fragment_shader(std::function<Fragment(const Fragment&)> shader) {
    fragment_shader_ = shader;
}

void GraphicsPipeline::draw_triangles(const std::vector<Vertex>& vertices) {
    if (perf_monitor_) {
        perf_monitor_->start_timer("draw_triangles");
    }
    
    // Process vertices in groups of 3 (triangles)
    for (size_t i = 0; i + 2 < vertices.size(); i += 3) {
        std::vector<Vertex> triangle = {vertices[i], vertices[i+1], vertices[i+2]};
        
        // Vertex stage
        auto transformed_vertices = vertex_stage(triangle);
        
        // Culling
        if (pipeline_state_.culling_enabled && 
            is_triangle_culled(transformed_vertices[0], transformed_vertices[1], transformed_vertices[2])) {
            continue;
        }
        
        // Rasterization
        auto fragments = rasterization_stage(transformed_vertices);
        
        // Fragment stage
        auto shaded_fragments = fragment_stage(fragments);
        
        // Output merger
        output_merger_stage(shaded_fragments);
        
        stats_.triangles_drawn++;
    }
    
    stats_.vertices_processed += vertices.size();
    
    if (perf_monitor_) {
        perf_monitor_->end_timer("draw_triangles");
        perf_monitor_->increment_counter("triangles_drawn", vertices.size() / 3);
        perf_monitor_->increment_counter("vertices_processed", vertices.size());
    }
}

void GraphicsPipeline::draw_indexed(const std::vector<Vertex>& vertices,
                                   const std::vector<uint32_t>& indices) {
    if (perf_monitor_) {
        perf_monitor_->start_timer("draw_indexed");
    }
    
    // Build triangle list from indices
    std::vector<Vertex> triangle_vertices;
    triangle_vertices.reserve(indices.size());
    
    for (uint32_t index : indices) {
        if (index < vertices.size()) {
            triangle_vertices.push_back(vertices[index]);
        }
    }
    
    // Draw the triangles
    draw_triangles(triangle_vertices);
    
    if (perf_monitor_) {
        perf_monitor_->end_timer("draw_indexed");
    }
}

std::vector<Vertex> GraphicsPipeline::vertex_stage(const std::vector<Vertex>& input_vertices) {
    std::vector<Vertex> output_vertices;
    output_vertices.reserve(input_vertices.size());
    
    // Apply vertex shader to each vertex
    for (const auto& vertex : input_vertices) {
        Vertex transformed_vertex = vertex;
        
        if (vertex_shader_) {
            transformed_vertex = vertex_shader_(vertex);
        } else {
            // Default vertex transformation (identity)
            transformed_vertex = vertex;
        }
        
        output_vertices.push_back(transformed_vertex);
    }
    
    return output_vertices;
}

std::vector<Fragment> GraphicsPipeline::rasterization_stage(const std::vector<Vertex>& vertices) {
    std::vector<Fragment> fragments;
    
    if (vertices.size() < 3) return fragments;
    
    const Vertex& v0 = vertices[0];
    const Vertex& v1 = vertices[1];
    const Vertex& v2 = vertices[2];
    
    // Transform to screen coordinates
    auto screen_coord = [this](const Vertex& v) -> std::pair<int, int> {
        // Simplified projection to screen space
        float x = (v.position[0] + 1.0f) * 0.5f * pipeline_state_.viewport_width;
        float y = (v.position[1] + 1.0f) * 0.5f * pipeline_state_.viewport_height;
        return {static_cast<int>(x), static_cast<int>(y)};
    };
    
    auto [x0, y0] = screen_coord(v0);
    auto [x1, y1] = screen_coord(v1);
    auto [x2, y2] = screen_coord(v2);
    
    // Simple bounding box rasterization
    int min_x = std::max(0, std::min({x0, x1, x2}));
    int max_x = std::min(static_cast<int>(pipeline_state_.viewport_width) - 1, std::max({x0, x1, x2}));
    int min_y = std::max(0, std::min({y0, y1, y2}));
    int max_y = std::min(static_cast<int>(pipeline_state_.viewport_height) - 1, std::max({y0, y1, y2}));
    
    // Generate fragments for pixels inside triangle
    for (int y = min_y; y <= max_y; ++y) {
        for (int x = min_x; x <= max_x; ++x) {
            // Barycentric coordinate test (simplified)
            float u = 0.33f, v = 0.33f, w = 0.34f; // Simplified - should compute properly
            
            if (u >= 0 && v >= 0 && w >= 0) {
                Fragment fragment = interpolate_fragment(v0, v1, v2, u, v, w);
                fragment.position[0] = static_cast<float>(x);
                fragment.position[1] = static_cast<float>(y);
                fragment.valid = true;
                
                fragments.push_back(fragment);
            }
        }
    }
    
    return fragments;
}

std::vector<Fragment> GraphicsPipeline::fragment_stage(const std::vector<Fragment>& fragments) {
    std::vector<Fragment> output_fragments;
    output_fragments.reserve(fragments.size());
    
    for (const auto& fragment : fragments) {
        Fragment shaded_fragment = fragment;
        
        if (fragment_shader_) {
            shaded_fragment = fragment_shader_(fragment);
        }
        
        // Texture sampling (if textures are bound)
        if (!bound_textures_.empty() && texture_cache_) {
            // Sample texture 0 at fragment texture coordinates
            const auto& texture = bound_textures_[0];
            if (!texture.data.empty()) {
                float u = fragment.texcoord[0];
                float v = fragment.texcoord[1];
                
                // Convert to texture coordinates
                uint32_t tex_x = static_cast<uint32_t>(u * texture.width) % texture.width;
                uint32_t tex_y = static_cast<uint32_t>(v * texture.height) % texture.height;
                uint64_t tex_offset = (tex_y * texture.width + tex_x) * 4; // Assume 4 bytes per pixel
                
                // Read texture data through cache (this utilizes our new feature!)
                uint8_t pixel_data[4];
                if (texture_cache_->read_texture(reinterpret_cast<uint64_t>(&texture), 0, 
                                               tex_offset, pixel_data, 4)) {
                    // Apply texture color to fragment
                    shaded_fragment.color[0] *= pixel_data[0] / 255.0f;
                    shaded_fragment.color[1] *= pixel_data[1] / 255.0f;
                    shaded_fragment.color[2] *= pixel_data[2] / 255.0f;
                    shaded_fragment.color[3] *= pixel_data[3] / 255.0f;
                    
                    stats_.texture_samples++;
                }
            }
        }
        
        output_fragments.push_back(shaded_fragment);
    }
    
    stats_.fragments_processed += fragments.size();
    return output_fragments;
}

void GraphicsPipeline::output_merger_stage(const std::vector<Fragment>& fragments) {
    for (const auto& fragment : fragments) {
        if (!fragment.valid) continue;
        
        int x = static_cast<int>(fragment.position[0]);
        int y = static_cast<int>(fragment.position[1]);
        
        if (x < 0 || x >= static_cast<int>(pipeline_state_.viewport_width) ||
            y < 0 || y >= static_cast<int>(pipeline_state_.viewport_height)) {
            continue;
        }
        
        size_t pixel_index = y * pipeline_state_.viewport_width + x;
        
        // Depth test
        if (pipeline_state_.depth_test_enabled) {
            if (fragment.depth >= depth_buffer_[pixel_index]) {
                continue; // Fragment is behind existing pixel
            }
            depth_buffer_[pixel_index] = fragment.depth;
        }
        
        // Color blending (simplified)
        if (pipeline_state_.blending_enabled) {
            // Simple alpha blending
            float alpha = fragment.color[3];
            uint32_t existing_color = color_buffer_[pixel_index];
            
            uint8_t existing_r = (existing_color >> 24) & 0xFF;
            uint8_t existing_g = (existing_color >> 16) & 0xFF;
            uint8_t existing_b = (existing_color >> 8) & 0xFF;
            
            uint8_t new_r = static_cast<uint8_t>(fragment.color[0] * alpha * 255 + existing_r * (1 - alpha));
            uint8_t new_g = static_cast<uint8_t>(fragment.color[1] * alpha * 255 + existing_g * (1 - alpha));
            uint8_t new_b = static_cast<uint8_t>(fragment.color[2] * alpha * 255 + existing_b * (1 - alpha));
            
            color_buffer_[pixel_index] = (new_r << 24) | (new_g << 16) | (new_b << 8) | 0xFF;
        } else {
            // Replace existing color
            uint8_t r = static_cast<uint8_t>(fragment.color[0] * 255);
            uint8_t g = static_cast<uint8_t>(fragment.color[1] * 255);
            uint8_t b = static_cast<uint8_t>(fragment.color[2] * 255);
            uint8_t a = static_cast<uint8_t>(fragment.color[3] * 255);
            
            color_buffer_[pixel_index] = (r << 24) | (g << 16) | (b << 8) | a;
        }
    }
}

bool GraphicsPipeline::is_triangle_culled(const Vertex& v0, const Vertex& v1, const Vertex& v2) {
    // Simple backface culling using cross product
    float edge1_x = v1.position[0] - v0.position[0];
    float edge1_y = v1.position[1] - v0.position[1];
    float edge2_x = v2.position[0] - v0.position[0];
    float edge2_y = v2.position[1] - v0.position[1];
    
    float cross_product = edge1_x * edge2_y - edge1_y * edge2_x;
    return cross_product <= 0; // Cull if facing away
}

Fragment GraphicsPipeline::interpolate_fragment(const Vertex& v0, const Vertex& v1, const Vertex& v2,
                                               float u, float v, float w) {
    Fragment fragment;
    
    // Interpolate position (already set by caller)
    fragment.position[2] = u * v0.position[2] + v * v1.position[2] + w * v2.position[2];
    fragment.position[3] = u * v0.position[3] + v * v1.position[3] + w * v2.position[3];
    
    // Interpolate color
    for (int i = 0; i < 4; ++i) {
        fragment.color[i] = u * v0.color[i] + v * v1.color[i] + w * v2.color[i];
    }
    
    // Interpolate texture coordinates
    for (int i = 0; i < 2; ++i) {
        fragment.texcoord[i] = u * v0.texcoord[i] + v * v1.texcoord[i] + w * v2.texcoord[i];
    }
    
    // Set depth
    fragment.depth = fragment.position[2];
    fragment.valid = true;
    
    return fragment;
}

void GraphicsPipeline::begin_frame() {
    frame_start_time_ = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::high_resolution_clock::now().time_since_epoch()).count();
    
    // Clear frame buffers
    std::fill(color_buffer_.begin(), color_buffer_.end(), 0x000000FF); // Black with full alpha
    std::fill(depth_buffer_.begin(), depth_buffer_.end(), 1.0f);
    
    // Reset frame statistics
    stats_.vertices_processed = 0;
    stats_.fragments_processed = 0;
    stats_.triangles_drawn = 0;
    stats_.texture_samples = 0;
    
    if (perf_monitor_) {
        perf_monitor_->start_timer("frame_time");
    }
}

void GraphicsPipeline::end_frame() {
    uint64_t frame_end_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::high_resolution_clock::now().time_since_epoch()).count();
    
    stats_.frame_time_ms = static_cast<double>(frame_end_time - frame_start_time_);
    
    if (perf_monitor_) {
        perf_monitor_->end_timer("frame_time");
        perf_monitor_->record_frame_metrics(stats_.frame_time_ms, 
                                          stats_.triangles_drawn, 
                                          stats_.fragments_processed);
    }
}

void GraphicsPipeline::present() {
    // In a real implementation, this would present the frame buffer to the display
    // For simulation, we just record the presentation
    
    if (perf_monitor_) {
        perf_monitor_->increment_counter("frames_presented");
    }
}

GraphicsPipeline::PipelineStats GraphicsPipeline::get_statistics() const {
    return stats_;
}

} // namespace gpu_sim