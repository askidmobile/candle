// candle_mmq_dense.cu
// extern "C" __global__ обёртки над llama.cpp mul_mat_q (Tensor-Core MMA) для
// плотного prefill. Вызываются из candle-core fast_mmq.rs через cudarc.
// Включён в PTX-сборку (имя не совпадает с exclude "mmq_*.cu").

#include "mmq_common.cuh"
#include "mmq_gguf.cuh"

// ---------------------------------------------------------------------------
// Quantize q8_1 (MMQ layout) — __device__ шаблон (копия из mmq_quantize.cu,
// чтобы не тянуть его хост-лаунчеры с <<<>>>).
// ---------------------------------------------------------------------------
template <mmq_q8_1_ds_layout ds_layout>
static __device__ __forceinline__ void quantize_mmq_q8_1_impl(
        const float * __restrict__ x, const int32_t * __restrict__ ids, void * __restrict__ vy,
        const int64_t ne00, const int64_t s01, const int64_t s02, const int64_t s03,
        const int64_t ne0, const int ne1, const int ne2) {

    constexpr int vals_per_scale = ds_layout == MMQ_Q8_1_DS_LAYOUT_D2S6 ? 64 : 32;
    constexpr int vals_per_sum   = ds_layout == MMQ_Q8_1_DS_LAYOUT_D2S6 ? 16 : 32;

    const int64_t i0 = ((int64_t)blockDim.x*blockIdx.y + threadIdx.x)*4;

    if (i0 >= ne0) {
        return;
    }

    const int64_t i1 = blockIdx.x;
    const int64_t i2 = blockIdx.z % ne2;
    const int64_t i3 = blockIdx.z / ne2;

    const int64_t i00 = i0;
    const int64_t i01 = ids ? ids[i1] : i1;
    const int64_t i02 = i2;
    const int64_t i03 = i3;

    const float4 * x4 = (const float4 *) x;

    block_q8_1_mmq * y = (block_q8_1_mmq *) vy;

    const int64_t ib0 = blockIdx.z*((int64_t)gridDim.x*gridDim.y*blockDim.x/QK8_1);
    const int64_t ib  = ib0 + (i0 / (4*QK8_1))*ne1 + blockIdx.x;
    const int64_t iqs = i0 % (4*QK8_1);

    const float4 xi = i0 < ne00 ? x4[(i03*s03 + i02*s02 + i01*s01 + i00)/4] : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float amax = fabsf(xi.x);
    amax = fmaxf(amax, fabsf(xi.y));
    amax = fmaxf(amax, fabsf(xi.z));
    amax = fmaxf(amax, fabsf(xi.w));

#pragma unroll
    for (int offset = vals_per_scale/8; offset > 0; offset >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFF, amax, offset, WARP_SIZE));
    }

    float sum;
    if (ds_layout != MMQ_Q8_1_DS_LAYOUT_D4) {
        sum = xi.x + xi.y + xi.z + xi.w;

#pragma unroll
        for (int offset = vals_per_sum/8; offset > 0; offset >>= 1) {
            sum += __shfl_xor_sync(0xFFFFFFFF, sum, offset, WARP_SIZE);
        }
    }

    const float d_inv = 127.0f / amax;
    char4 q;
    q.x = roundf(xi.x*d_inv);
    q.y = roundf(xi.y*d_inv);
    q.z = roundf(xi.z*d_inv);
    q.w = roundf(xi.w*d_inv);

    char4 * yqs4 = (char4 *) y[ib].qs;
    yqs4[iqs/4] = q;

    if (ds_layout == MMQ_Q8_1_DS_LAYOUT_D2S6) {
        if (iqs % 16 != 0 || iqs >= 96) {
            return;
        }

        y[ib].d2s6[2 + iqs/16] = sum;

        if (iqs % 64 != 0) {
            return;
        }

        const float d = 1.0f / d_inv;

        y[ib].d2s6[iqs/64] = d;

        return;
    }

    if (iqs % 32 != 0) {
        return;
    }

    const float d = 1.0f / d_inv;

    if (ds_layout == MMQ_Q8_1_DS_LAYOUT_DS4) {
        y[ib].ds4[iqs/32] = make_half2(d, sum);
    } else {
        y[ib].d4[iqs/32]  = d;
    }
}

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
