// candle_mmq_dense.cu
// extern "C" __global__ обёртки над llama.cpp mul_mat_q (Tensor-Core MMA) для
// плотного prefill. Вызываются из candle-core fast_mmq.rs через cudarc.
// Включён в PTX-сборку (имя не совпадает с exclude "mmq_*.cu").

#include "mmq_common.cuh"
#include "mmq_gguf.cuh"
#include "mmq_quantize.cu"

// ---------------------------------------------------------------------------
// Quantize q8_1 (MMQ layout) обёртки
// ---------------------------------------------------------------------------
extern "C" __global__ void candle_mmq_quant_d4(
        const float * __restrict__ x, const int32_t * __restrict__ ids, void * __restrict__ vy,
        const int64_t ne00, const int64_t s01, const int64_t s02, const int64_t s03,
        const int64_t ne0, const int ne1, const int ne2) {
    quantize_mmq_q8_1_impl<MMQ_Q8_1_DS_LAYOUT_D4>(x, ids, vy, ne00, s01, s02, s03, ne0, ne1, ne2);
}
extern "C" __global__ void candle_mmq_quant_ds4(
        const float * __restrict__ x, const int32_t * __restrict__ ids, void * __restrict__ vy,
        const int64_t ne00, const int64_t s01, const int64_t s02, const int64_t s03,
        const int64_t ne0, const int ne1, const int ne2) {
    quantize_mmq_q8_1_impl<MMQ_Q8_1_DS_LAYOUT_DS4>(x, ids, vy, ne00, s01, s02, s03, ne0, ne1, ne2);
}
extern "C" __global__ void candle_mmq_quant_d2s6(
        const float * __restrict__ x, const int32_t * __restrict__ ids, void * __restrict__ vy,
        const int64_t ne00, const int64_t s01, const int64_t s02, const int64_t s03,
        const int64_t ne0, const int ne1, const int ne2) {
    quantize_mmq_q8_1_impl<MMQ_Q8_1_DS_LAYOUT_D2S6>(x, ids, vy, ne00, s01, s02, s03, ne0, ne1, ne2);
}

// ---------------------------------------------------------------------------
// mul_mat_q обёртки. Сигнатура = mul_mat_q_impl (23 аргумента).
// Для плотного matmul: ids_dst=null, expert_bounds=null, tmp_fixup=null,
// channel_ratio=1, nchannels_y=1, sample_ratio=1, nsamples_y=1, все stride_*=0.
// ---------------------------------------------------------------------------
#define MMQ_ARGS \
    const char * __restrict__ x, const int * __restrict__ y, const int32_t * __restrict__ ids_dst, \
    const int32_t * __restrict__ expert_bounds, float * __restrict__ dst, float * __restrict__ tmp_fixup, \
    const int ncols_x, const int nrows_x, const int ncols_dst, const int stride_row_x, const int ncols_y, const int stride_col_dst, \
    const int channel_ratio, const int nchannels_y, const int stride_channel_x, const int stride_channel_y, const int stride_channel_dst, \
    const int sample_ratio, const int nsamples_y, const int stride_sample_x, const int stride_sample_y, const int stride_sample_dst, \
    const int ncols_max

#define MMQ_PASS x, y, ids_dst, expert_bounds, dst, tmp_fixup, \
    ncols_x, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, \
    channel_ratio, nchannels_y, stride_channel_x, stride_channel_y, stride_channel_dst, \
    sample_ratio, nsamples_y, stride_sample_x, stride_sample_y, stride_sample_dst, ncols_max

// need_check=false (nrows_x кратен mmq_y — гарантируем паддингом весов)
#define DEFINE_MMQ(ggml_type_const, tag, MMQX) \
    extern "C" __global__ void __launch_bounds__(WARP_SIZE * (256 / WARP_SIZE)) \
    candle_mmq_##tag##_x##MMQX(MMQ_ARGS) { \
        mul_mat_q_impl<ggml_type_const, MMQX, false>(MMQ_PASS); \
    }

// K-quants, mmq_x ∈ {32, 64, 128}
DEFINE_MMQ(GGML_TYPE_Q2_K, q2_k, 32)
DEFINE_MMQ(GGML_TYPE_Q2_K, q2_k, 64)
DEFINE_MMQ(GGML_TYPE_Q2_K, q2_k, 128)
DEFINE_MMQ(GGML_TYPE_Q3_K, q3_k, 32)
DEFINE_MMQ(GGML_TYPE_Q3_K, q3_k, 64)
DEFINE_MMQ(GGML_TYPE_Q3_K, q3_k, 128)
DEFINE_MMQ(GGML_TYPE_Q4_K, q4_k, 32)
DEFINE_MMQ(GGML_TYPE_Q4_K, q4_k, 64)
DEFINE_MMQ(GGML_TYPE_Q4_K, q4_k, 128)
DEFINE_MMQ(GGML_TYPE_Q5_K, q5_k, 32)
DEFINE_MMQ(GGML_TYPE_Q5_K, q5_k, 64)
DEFINE_MMQ(GGML_TYPE_Q5_K, q5_k, 128)
DEFINE_MMQ(GGML_TYPE_Q6_K, q6_k, 32)
DEFINE_MMQ(GGML_TYPE_Q6_K, q6_k, 64)
DEFINE_MMQ(GGML_TYPE_Q6_K, q6_k, 128)
// базовые типы тоже (q4_0/q4_1 встречаются в некоторых моделях)
DEFINE_MMQ(GGML_TYPE_Q4_0, q4_0, 32)
DEFINE_MMQ(GGML_TYPE_Q4_0, q4_0, 64)
DEFINE_MMQ(GGML_TYPE_Q4_0, q4_0, 128)
DEFINE_MMQ(GGML_TYPE_Q8_0, q8_0, 32)
DEFINE_MMQ(GGML_TYPE_Q8_0, q8_0, 64)
DEFINE_MMQ(GGML_TYPE_Q8_0, q8_0, 128)
