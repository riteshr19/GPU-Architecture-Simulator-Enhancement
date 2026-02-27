#include "tensor_core.h"
#include <cstring>
#include <cmath>
#include <sstream>
#include <stdexcept>

namespace gpu_sim {

// ---------------------------------------------------------------------------
// FP16 ↔ FP32 conversion helpers
// ---------------------------------------------------------------------------

fp16_t float_to_fp16(float v) {
    // IEEE 754 FP16: 1 sign, 5 exponent, 10 mantissa
    uint32_t bits;
    std::memcpy(&bits, &v, sizeof(bits));

    uint32_t sign     = (bits >> 31) & 0x1;
    int32_t  exponent = static_cast<int32_t>((bits >> 23) & 0xFF) - 127 + 15;
    uint32_t mantissa = (bits >> 13) & 0x3FF;

    if (exponent <= 0) {
        // Underflow to zero (denormals not modelled)
        return static_cast<fp16_t>(sign << 15);
    }
    if (exponent >= 31) {
        // Overflow to infinity
        return static_cast<fp16_t>((sign << 15) | (0x1F << 10));
    }
    return static_cast<fp16_t>((sign << 15) | (static_cast<uint32_t>(exponent) << 10) | mantissa);
}

float fp16_to_float(fp16_t h) {
    uint32_t sign     = (h >> 15) & 0x1;
    uint32_t exponent = (h >> 10) & 0x1F;
    uint32_t mantissa = h & 0x3FF;

    uint32_t bits;
    if (exponent == 0) {
        // Zero / denormal → zero
        bits = sign << 31;
    } else if (exponent == 0x1F) {
        // Inf / NaN
        bits = (sign << 31) | (0xFF << 23) | (mantissa << 13);
    } else {
        bits = (sign << 31)
             | ((exponent - 15 + 127) << 23)
             | (mantissa << 13);
    }

    float result;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

// ---------------------------------------------------------------------------
// TensorCore
// ---------------------------------------------------------------------------

TensorCore::TensorCore(uint32_t unit_id)
    : unit_id_(unit_id),
      mma_cycles_(0),
      mma_count_(0),
      flops_(0),
      busy_until_cycle_(0)
{}

// ---- Fragment load/store --------------------------------------------------

void TensorCore::load_matrix_a(FragmentA& frag, const fp16_t* src,
                                FragmentLayout layout) {
    frag.layout = layout;
    std::memcpy(frag.data.data(), src, WMMA_M * WMMA_K * sizeof(fp16_t));
}

void TensorCore::load_matrix_b(FragmentB& frag, const fp16_t* src,
                                FragmentLayout layout) {
    frag.layout = layout;
    std::memcpy(frag.data.data(), src, WMMA_K * WMMA_N * sizeof(fp16_t));
}

void TensorCore::fill_zero(FragmentC& frag) {
    frag.data.fill(0.0f);
}

void TensorCore::load_accumulator(FragmentC& frag, const float* src) {
    std::memcpy(frag.data.data(), src, WMMA_M * WMMA_N * sizeof(float));
}

void TensorCore::store_accumulator(const FragmentC& frag, float* dst) {
    std::memcpy(dst, frag.data.data(), WMMA_M * WMMA_N * sizeof(float));
}

// ---- Core MMA operation --------------------------------------------------

/*
 * D[m][n] = sum_{k} A[m][k] * B[k][n]  +  C[m][n]
 *
 * A is stored row-major (WMMA_M × WMMA_K),
 * B is stored col-major (WMMA_K × WMMA_N) by default (matching NVIDIA layout).
 * C and D are row-major (WMMA_M × WMMA_N).
 *
 * This software emulation promotes FP16 inputs to FP32 for each multiply,
 * accumulates in FP32, and adds the FP32 accumulator — exactly as real
 * Tensor Cores behave (mixed-precision MMA).
 */
void TensorCore::mma_sync(FragmentC& d,
                          const FragmentA& a,
                          const FragmentB& b,
                          const FragmentC& c,
                          uint64_t cycle) {
    // Stall check: unit is occupied until busy_until_cycle_.
    if (cycle < busy_until_cycle_) {
        // In a real pipeline this would cause a pipeline stall; here we
        // simply advance our model's notion of when we start.
        cycle = busy_until_cycle_;
    }

    // Compute D = A × B + C
    for (uint32_t m = 0; m < WMMA_M; ++m) {
        for (uint32_t n = 0; n < WMMA_N; ++n) {
            float acc = c.data[m * WMMA_N + n];
            for (uint32_t k = 0; k < WMMA_K; ++k) {
                // A is row-major: element (m, k) → index m*K + k
                float a_val = fp16_to_float(a.data[m * WMMA_K + k]);

                // B layout handling:
                //   ROW_MAJOR: element (k, n) → index k*N + n
                //   COL_MAJOR: element (k, n) → index n*K + k
                float b_val;
                if (b.layout == FragmentLayout::COL_MAJOR) {
                    b_val = fp16_to_float(b.data[n * WMMA_K + k]);
                } else {
                    b_val = fp16_to_float(b.data[k * WMMA_N + n]);
                }

                acc += a_val * b_val;
            }
            d.data[m * WMMA_N + n] = acc;
        }
    }

    // Update timing model.
    busy_until_cycle_ = cycle + MMA_LATENCY_CYCLES;
    mma_cycles_ += MMA_LATENCY_CYCLES;
    mma_count_++;

    // Floating-point operations: 2 * M * N * K (multiply + add per element).
    flops_ += 2ULL * WMMA_M * WMMA_N * WMMA_K;
}

// ---- Diagnostics ---------------------------------------------------------

std::string TensorCore::get_stats_summary() const {
    std::ostringstream oss;
    oss << "TensorCore[" << unit_id_ << "] stats:\n"
        << "  MMA count  : " << mma_count_ << "\n"
        << "  MMA cycles : " << mma_cycles_ << "\n"
        << "  FLOPs      : " << flops_ << "\n"
        << "  Busy until : cycle " << busy_until_cycle_ << "\n";
    return oss.str();
}

} // namespace gpu_sim
