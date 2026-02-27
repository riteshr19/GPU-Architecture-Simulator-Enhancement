#pragma once

#include <vector>
#include <memory>
#include <cstdint>
#include <array>

namespace gpu_sim {

class PerformanceMonitor;

/**
 * Supported precision modes for Tensor Core MMA operations
 */
enum class PrecisionMode {
    FP16_FP16,    // FP16 inputs, FP16 accumulate
    FP16_FP32,    // FP16 inputs, FP32 accumulate (mixed-precision)
    BF16_FP32,    // BF16 inputs, FP32 accumulate
    TF32_FP32,    // TF32 inputs, FP32 accumulate
    INT8_INT32,   // INT8 inputs, INT32 accumulate
    INT4_INT32    // INT4 inputs, INT32 accumulate
};

/**
 * Matrix fragment descriptor — describes a tile held in registers
 */
struct MatrixFragment {
    uint32_t rows;
    uint32_t cols;
    PrecisionMode precision;
    std::vector<float> data;   // Stored as float for simulation; real HW uses packed types

    MatrixFragment(uint32_t r, uint32_t c, PrecisionMode p)
        : rows(r), cols(c), precision(p), data(r * c, 0.0f) {}

    float& at(uint32_t row, uint32_t col) { return data[row * cols + col]; }
    const float& at(uint32_t row, uint32_t col) const { return data[row * cols + col]; }
};

/**
 * Tensor Core Unit — simulates mixed-precision Matrix-Multiply-Accumulate
 *
 * Models NVIDIA-style Tensor Core operations:
 *   D = A * B + C
 * where A, B are input fragments and C/D are accumulator fragments.
 *
 * The default tile size is 16x16x16 (M=16, N=16, K=16) following
 * the Volta/Turing WMMA API convention.
 *
 * Pipeline latency is modeled per MMA instruction to enable accurate
 * throughput estimation when integrated with the warp scheduler.
 */
class TensorCore {
public:
    explicit TensorCore(uint32_t unit_id = 0,
                        uint32_t tile_m = 16,
                        uint32_t tile_n = 16,
                        uint32_t tile_k = 16);
    ~TensorCore() = default;

    // Initialization
    void initialize(std::shared_ptr<PerformanceMonitor> perf_monitor);

    // Core MMA operation: D = A * B + C
    void mma(const MatrixFragment& A,
             const MatrixFragment& B,
             const MatrixFragment& C,
             MatrixFragment& D);

    // Load / Store fragment helpers (simulated register file interaction)
    MatrixFragment load_matrix_a(const std::vector<float>& data,
                                 uint32_t rows, uint32_t cols,
                                 PrecisionMode precision);

    MatrixFragment load_matrix_b(const std::vector<float>& data,
                                 uint32_t rows, uint32_t cols,
                                 PrecisionMode precision);

    MatrixFragment create_accumulator(uint32_t rows, uint32_t cols,
                                      PrecisionMode precision,
                                      float init_value = 0.0f);

    // Configuration
    void set_precision_mode(PrecisionMode mode) { current_precision_ = mode; }
    PrecisionMode get_precision_mode() const { return current_precision_; }

    // Performance metrics
    struct TensorCoreStats {
        uint64_t mma_operations;
        uint64_t total_flops;           // Floating-point operations executed
        uint64_t total_cycles;
        double   throughput_tflops;     // Estimated TFLOPS
        uint64_t precision_conversions; // FP16↔FP32 conversions
    };

    TensorCoreStats get_statistics() const;
    void reset_statistics();

    // Tile size accessors
    uint32_t get_tile_m() const { return tile_m_; }
    uint32_t get_tile_n() const { return tile_n_; }
    uint32_t get_tile_k() const { return tile_k_; }

private:
    // Precision-aware multiply (simulates quantization effects)
    float precision_mul(float a, float b, PrecisionMode mode) const;
    float precision_add(float a, float b, PrecisionMode mode) const;

    uint32_t unit_id_;
    uint32_t tile_m_, tile_n_, tile_k_;
    PrecisionMode current_precision_;

    // Latency model (cycles per MMA instruction)
    static constexpr uint32_t MMA_LATENCY_FP16  = 8;
    static constexpr uint32_t MMA_LATENCY_BF16  = 8;
    static constexpr uint32_t MMA_LATENCY_TF32  = 16;
    static constexpr uint32_t MMA_LATENCY_INT8  = 4;
    static constexpr uint32_t MMA_LATENCY_INT4  = 4;

    mutable TensorCoreStats stats_;
    std::shared_ptr<PerformanceMonitor> perf_monitor_;
};

} // namespace gpu_sim
