#pragma once

#include <array>
#include <cstdint>
#include <string>

namespace gpu_sim {

/**
 * Half-precision float (FP16) represented as a 16-bit unsigned integer.
 * A minimal software implementation is provided so the simulator can run
 * without hardware FP16 support.  Real GPU toolchains use __half.
 */
using fp16_t = uint16_t;

/**
 * Convert a 32-bit float to FP16 (round-to-nearest, no denormal support).
 */
fp16_t float_to_fp16(float v);

/**
 * Convert FP16 back to a 32-bit float.
 */
float fp16_to_float(fp16_t h);

// ---------------------------------------------------------------------------
// WMMA tile dimensions matching NVIDIA Volta / Turing hardware:
//   m=16, n=16, k=16  (FP16 inputs, FP32 accumulation)
// ---------------------------------------------------------------------------
static constexpr uint32_t WMMA_M = 16;
static constexpr uint32_t WMMA_N = 16;
static constexpr uint32_t WMMA_K = 16;

/**
 * Fragment types mirror the CUDA WMMA API.
 */
enum class FragmentUse { MATRIX_A, MATRIX_B, ACCUMULATOR };
enum class FragmentLayout { ROW_MAJOR, COL_MAJOR };

/**
 * A 16×16 fragment of matrix A (FP16, stored in FP16).
 */
struct FragmentA {
    std::array<fp16_t, WMMA_M * WMMA_K> data{};
    FragmentLayout layout = FragmentLayout::ROW_MAJOR;
};

/**
 * A 16×16 fragment of matrix B (FP16, stored in FP16).
 */
struct FragmentB {
    std::array<fp16_t, WMMA_K * WMMA_N> data{};
    FragmentLayout layout = FragmentLayout::COL_MAJOR;
};

/**
 * A 16×16 accumulator fragment (FP32).
 */
struct FragmentC {
    std::array<float, WMMA_M * WMMA_N> data{};
};

/**
 * TensorCore – models a single Tensor Core execution unit inside a warp.
 *
 * It exposes the three-step WMMA workflow:
 *   1. load_matrix_a / load_matrix_b   – load FP16 tiles from memory.
 *   2. load_accumulator / fill_zero    – set up the FP32 accumulator.
 *   3. mma_sync                        – perform D = A × B + C (FP16→FP32).
 *   4. store_accumulator               – write result back to memory.
 *
 * Latency modelling:
 *   - Each mma_sync call advances the internal cycle counter by
 *     MMA_LATENCY_CYCLES (16 cycles, matching Volta latency).
 *   - Stalls are recorded when the unit is unavailable.
 */
class TensorCore {
public:
    /** Construct a TensorCore; optionally set a unique unit ID. */
    explicit TensorCore(uint32_t unit_id = 0);
    ~TensorCore() = default;

    // ---- Fragment load/store ----------------------------------------------

    /**
     * Load a 16×16 FP16 tile into fragment A from a row-major host buffer.
     * @param src    Pointer to WMMA_M * WMMA_K fp16_t values.
     * @param layout ROW_MAJOR or COL_MAJOR.
     */
    void load_matrix_a(FragmentA& frag, const fp16_t* src,
                       FragmentLayout layout = FragmentLayout::ROW_MAJOR);

    /**
     * Load a 16×16 FP16 tile into fragment B.
     * @param src    Pointer to WMMA_K * WMMA_N fp16_t values.
     * @param layout ROW_MAJOR or COL_MAJOR.
     */
    void load_matrix_b(FragmentB& frag, const fp16_t* src,
                       FragmentLayout layout = FragmentLayout::COL_MAJOR);

    /** Zero-initialise the accumulator fragment. */
    void fill_zero(FragmentC& frag);

    /**
     * Load a 16×16 FP32 tile into the accumulator fragment.
     * @param src Pointer to WMMA_M * WMMA_N float values (row-major).
     */
    void load_accumulator(FragmentC& frag, const float* src);

    /**
     * Store the accumulator fragment to a row-major host buffer.
     * @param dst Pointer to WMMA_M * WMMA_N float values.
     */
    void store_accumulator(const FragmentC& frag, float* dst);

    // ---- Core MMA operation -----------------------------------------------

    /**
     * Perform: D = A × B + C
     *
     * A is (M×K) FP16, B is (K×N) FP16, C and D are (M×N) FP32.
     * Models the synchronous warp-level MMA instruction.
     *
     * @param d        Output accumulator.
     * @param a        Input matrix A fragment.
     * @param b        Input matrix B fragment.
     * @param c        Input accumulator (added to A×B).
     * @param cycle    Current simulator cycle (for latency tracking).
     */
    void mma_sync(FragmentC& d,
                  const FragmentA& a,
                  const FragmentB& b,
                  const FragmentC& c,
                  uint64_t cycle = 0);

    // ---- Diagnostics ------------------------------------------------------

    /** Cycles consumed by MMA operations. */
    uint64_t mma_cycles()  const { return mma_cycles_; }

    /** Total number of MMA operations completed. */
    uint64_t mma_count()   const { return mma_count_; }

    /** Floating-point operations performed (2 * M * N * K per MMA). */
    uint64_t flops()       const { return flops_; }

    /** Unit is busy until this cycle. */
    uint64_t busy_until()  const { return busy_until_cycle_; }

    /** True if the unit is available at the given cycle. */
    bool is_available(uint64_t cycle) const { return cycle >= busy_until_cycle_; }

    /** Human-readable statistics string. */
    std::string get_stats_summary() const;

    // ---- Latency constant (public for testing) ----------------------------
    static constexpr uint32_t MMA_LATENCY_CYCLES = 16;

private:
    uint32_t unit_id_;
    uint64_t mma_cycles_;
    uint64_t mma_count_;
    uint64_t flops_;
    uint64_t busy_until_cycle_;
};

} // namespace gpu_sim
