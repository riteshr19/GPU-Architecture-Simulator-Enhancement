#include "tensor_core.h"
#include "performance_monitor.h"
#include <cmath>
#include <cstring>
#include <algorithm>
#include <limits>

namespace gpu_sim {

TensorCore::TensorCore(uint32_t unit_id, uint32_t tile_m, uint32_t tile_n, uint32_t tile_k)
    : unit_id_(unit_id), tile_m_(tile_m), tile_n_(tile_n), tile_k_(tile_k),
      current_precision_(PrecisionMode::FP16_FP32) {
    stats_ = TensorCoreStats{};
}

void TensorCore::initialize(std::shared_ptr<PerformanceMonitor> perf_monitor) {
    perf_monitor_ = perf_monitor;
    if (perf_monitor_) {
        perf_monitor_->set_counter("tensor_core_tile_m", tile_m_);
        perf_monitor_->set_counter("tensor_core_tile_n", tile_n_);
        perf_monitor_->set_counter("tensor_core_tile_k", tile_k_);
    }
}

// ---------------------------------------------------------------------------
// D = A * B + C   (Matrix-Multiply-Accumulate)
// ---------------------------------------------------------------------------
void TensorCore::mma(const MatrixFragment& A,
                     const MatrixFragment& B,
                     const MatrixFragment& C,
                     MatrixFragment& D) {
    // Validate dimensions: A is MxK, B is KxN, C and D are MxN
    if (A.cols != B.rows || A.rows != C.rows || B.cols != C.cols) {
        return; // Dimension mismatch
    }

    uint32_t M = A.rows;
    uint32_t N = B.cols;
    uint32_t K = A.cols;

    // Ensure D has proper dimensions
    D = MatrixFragment(M, N, C.precision);

    // Perform MMA: D[m][n] = sum_k(A[m][k] * B[k][n]) + C[m][n]
    for (uint32_t m = 0; m < M; ++m) {
        for (uint32_t n = 0; n < N; ++n) {
            float acc = C.at(m, n);   // Start from accumulator C

            for (uint32_t k = 0; k < K; ++k) {
                float a_val = precision_mul(A.at(m, k), B.at(k, n), A.precision);
                acc = precision_add(acc, a_val, C.precision);
            }

            D.at(m, n) = acc;
        }
    }

    // Update statistics
    uint64_t flops = 2ULL * M * N * K;  // Each multiply-add = 2 FLOPs
    stats_.mma_operations++;
    stats_.total_flops += flops;

    // Determine latency based on precision
    uint32_t latency = MMA_LATENCY_FP16;
    switch (A.precision) {
        case PrecisionMode::FP16_FP16:
        case PrecisionMode::FP16_FP32:
            latency = MMA_LATENCY_FP16;
            stats_.precision_conversions += (A.precision == PrecisionMode::FP16_FP32) ? M * K : 0;
            break;
        case PrecisionMode::BF16_FP32:
            latency = MMA_LATENCY_BF16;
            stats_.precision_conversions += M * K;
            break;
        case PrecisionMode::TF32_FP32:
            latency = MMA_LATENCY_TF32;
            break;
        case PrecisionMode::INT8_INT32:
            latency = MMA_LATENCY_INT8;
            break;
        case PrecisionMode::INT4_INT32:
            latency = MMA_LATENCY_INT4;
            break;
    }

    stats_.total_cycles += latency;

    if (perf_monitor_) {
        perf_monitor_->increment_counter("tensor_core_mma_ops");
        perf_monitor_->increment_counter("tensor_core_flops", flops);
    }
}

MatrixFragment TensorCore::load_matrix_a(const std::vector<float>& data,
                                          uint32_t rows, uint32_t cols,
                                          PrecisionMode precision) {
    MatrixFragment frag(rows, cols, precision);
    size_t copy_size = std::min(data.size(), static_cast<size_t>(rows * cols));
    std::copy_n(data.begin(), copy_size, frag.data.begin());
    return frag;
}

MatrixFragment TensorCore::load_matrix_b(const std::vector<float>& data,
                                          uint32_t rows, uint32_t cols,
                                          PrecisionMode precision) {
    MatrixFragment frag(rows, cols, precision);
    size_t copy_size = std::min(data.size(), static_cast<size_t>(rows * cols));
    std::copy_n(data.begin(), copy_size, frag.data.begin());
    return frag;
}

MatrixFragment TensorCore::create_accumulator(uint32_t rows, uint32_t cols,
                                               PrecisionMode precision,
                                               float init_value) {
    MatrixFragment frag(rows, cols, precision);
    std::fill(frag.data.begin(), frag.data.end(), init_value);
    return frag;
}

// ---------------------------------------------------------------------------
// Precision simulation helpers
// ---------------------------------------------------------------------------
float TensorCore::precision_mul(float a, float b, PrecisionMode mode) const {
    switch (mode) {
        case PrecisionMode::FP16_FP16:
        case PrecisionMode::FP16_FP32: {
            // Simulate FP16 range clamp: ±65504
            auto clamp_fp16 = [](float v) -> float {
                return std::max(-65504.0f, std::min(65504.0f, v));
            };
            return clamp_fp16(a) * clamp_fp16(b);
        }
        case PrecisionMode::BF16_FP32: {
            // BF16: same exponent range as FP32 but only 8 mantissa bits
            // Simulate by truncating mantissa (round to nearest)
            auto to_bf16 = [](float v) -> float {
                uint32_t bits;
                std::memcpy(&bits, &v, sizeof(bits));
                bits &= 0xFFFF0000u;  // Zero low 16 bits of mantissa
                float result;
                std::memcpy(&result, &bits, sizeof(result));
                return result;
            };
            return to_bf16(a) * to_bf16(b);
        }
        case PrecisionMode::TF32_FP32: {
            // TF32: 19-bit format (1 sign + 8 exponent + 10 mantissa)
            auto to_tf32 = [](float v) -> float {
                uint32_t bits;
                std::memcpy(&bits, &v, sizeof(bits));
                bits &= 0xFFFFE000u;  // Zero low 13 bits of mantissa
                float result;
                std::memcpy(&result, &bits, sizeof(result));
                return result;
            };
            return to_tf32(a) * to_tf32(b);
        }
        case PrecisionMode::INT8_INT32: {
            // Clamp to INT8 range [-128, 127]
            int8_t ai = static_cast<int8_t>(std::max(-128.0f, std::min(127.0f, a)));
            int8_t bi = static_cast<int8_t>(std::max(-128.0f, std::min(127.0f, b)));
            return static_cast<float>(static_cast<int32_t>(ai) * static_cast<int32_t>(bi));
        }
        case PrecisionMode::INT4_INT32: {
            // Clamp to INT4 range [-8, 7]
            int8_t ai = static_cast<int8_t>(std::max(-8.0f, std::min(7.0f, a)));
            int8_t bi = static_cast<int8_t>(std::max(-8.0f, std::min(7.0f, b)));
            return static_cast<float>(static_cast<int32_t>(ai) * static_cast<int32_t>(bi));
        }
    }
    return a * b;
}

float TensorCore::precision_add(float a, float b, PrecisionMode mode) const {
    switch (mode) {
        case PrecisionMode::FP16_FP16: {
            float sum = a + b;
            return std::max(-65504.0f, std::min(65504.0f, sum));
        }
        default:
            // FP32 or INT32 accumulation — no precision loss
            return a + b;
    }
}

TensorCore::TensorCoreStats TensorCore::get_statistics() const {
    TensorCoreStats s = stats_;
    if (s.total_cycles > 0) {
        // Throughput in TFLOPS (assuming 1 GHz clock for simulation)
        s.throughput_tflops = static_cast<double>(s.total_flops) /
                              (static_cast<double>(s.total_cycles) * 1e3);
    }
    return s;
}

void TensorCore::reset_statistics() {
    stats_ = TensorCoreStats{};
}

} // namespace gpu_sim
