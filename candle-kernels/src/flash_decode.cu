// Split-K flash-decoding для decode (seqlen_q=1, длинный KV).
// FA2 при q=1 держит на GPU только n_head блоков (24 для 27B) — latency-bound
// на длинном KV. Здесь: KV режется на S чанков, каждый блок считает частичный
// online-softmax, второй kernel объединяет. Grid A = (n_head, S).
//
// Layout: q [n_head, hd] F16; k/v [kv_len, n_kv, hd] F16 (head-last, как наш
// Q8 batched cache после dequant). GQA: kv_head = h / (n_head / n_kv).

#include <cuda_fp16.h>
#include "cuda_utils.cuh"

struct FlashDecodeParams {
    unsigned int n_head;    // 24
    unsigned int n_kv;      // 4
    unsigned int hd;        // 256
    unsigned int kv_len;
    unsigned int splits;    // S
    float scale;            // 1/sqrt(hd)
};

__device__ __forceinline__ float warp_sum(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffff, v, o);
    return v;
}

// Kernel A: частичный attention по чанку KV.
// grid=(n_head, S), block=128 (4 warps).
// partials layout: [n_head, S, 2 + hd] f32 → [m, l, acc[hd]]
extern "C" __global__ void flash_decode_partial(
    const __half* __restrict__ q,      // [n_head * hd]
    const __half* __restrict__ k,      // [kv_len * n_kv * hd]
    const __half* __restrict__ v,      // [kv_len * n_kv * hd]
    float* __restrict__ partials,      // [n_head * S * (2 + hd)]
    const FlashDecodeParams params
) {
    const unsigned int h = blockIdx.x;
    const unsigned int s = blockIdx.y;
    const unsigned int hd = params.hd;
    const unsigned int n_kv = params.n_kv;
    const unsigned int kvh = h / (params.n_head / n_kv);
    const unsigned int lane = threadIdx.x % 32;
    const unsigned int warp = threadIdx.x / 32;
    const unsigned int nwarp = blockDim.x / 32;

    const unsigned int chunk = (params.kv_len + params.splits - 1) / params.splits;
    const unsigned int p_start = s * chunk;
    const unsigned int p_end = min(p_start + chunk, params.kv_len);

    // q головы в shared (f32).
    __shared__ float q_sh[256];
    __shared__ float red[4];
    __shared__ float m_shared, l_shared;
    for (unsigned int d = threadIdx.x; d < hd; d += blockDim.x) {
        q_sh[d] = __half2float(q[h * hd + d]);
    }
    if (threadIdx.x == 0) { m_shared = -INFINITY; l_shared = 0.0f; }
    __syncthreads();

    // acc: thread владеет двумя выходными dim (128 threads × 2 = 256).
    float acc0 = 0.0f, acc1 = 0.0f;
    const unsigned int d0 = threadIdx.x;          // 0..127
    const unsigned int d1 = threadIdx.x + 128;    // 128..255

    for (unsigned int p = p_start + warp; p < p_end; p += nwarp) {
        const __half* kp = k + ((size_t)p * n_kv + kvh) * hd;
        // dot(q, k[p]): 8 элементов на lane + warp reduce.
        float part = 0.0f;
        for (unsigned int d = lane * 8; d < lane * 8 + 8 && d < hd; d++) {
            part += q_sh[d] * __half2float(kp[d]);
        }
        part = warp_sum(part);
        // broadcast dot всему блоку.
        if (lane == 0) red[warp] = part;
        __syncthreads();

        const float dot = red[warp] * params.scale;
        // online softmax update (все потоки считают одно и то же — детерминизм).
        const float m_old = m_shared;
        const float m_new = fmaxf(m_old, dot);
        const float corr = __expf(m_old - m_new);
        const float p_exp = __expf(dot - m_new);
        const float l_new = l_shared * corr + p_exp;
        m_shared = m_new;
        l_shared = l_new;

        const __half* vp = v + ((size_t)p * n_kv + kvh) * hd;
        if (d0 < hd) acc0 = acc0 * corr + p_exp * __half2float(vp[d0]);
        if (d1 < hd) acc1 = acc1 * corr + p_exp * __half2float(vp[d1]);
        __syncthreads(); // red[] reuse guard
    }

    // Запись partial.
    float* out = partials + ((size_t)h * params.splits + s) * (2 + hd);
    if (threadIdx.x == 0) {
        out[0] = m_shared;
        out[1] = l_shared;
    }
    if (d0 < hd) out[2 + d0] = acc0;
    if (d1 < hd) out[2 + d1] = acc1;
}

// Kernel B: combine partials → final. grid=(n_head), block=128.
extern "C" __global__ void flash_decode_combine(
    const float* __restrict__ partials, // [n_head * S * (2 + hd)]
    __half* __restrict__ out,           // [n_head * hd]
    const FlashDecodeParams params
) {
    const unsigned int h = blockIdx.x;
    const unsigned int hd = params.hd;
    const unsigned int S = params.splits;
    const size_t stride = 2 + hd;

    __shared__ float m_g, l_g;
    if (threadIdx.x == 0) {
        float m = -INFINITY;
        for (unsigned int s = 0; s < S; s++) {
            m = fmaxf(m, partials[((size_t)h * S + s) * stride]);
        }
        float l = 0.0f;
        for (unsigned int s = 0; s < S; s++) {
            const float* p = partials + ((size_t)h * S + s) * stride;
            if (p[1] > 0.0f) l += p[1] * __expf(p[0] - m);
        }
        m_g = m;
        l_g = l;
    }
    __syncthreads();

    const float inv_l = (l_g > 0.0f) ? 1.0f / l_g : 0.0f;
    for (unsigned int d = threadIdx.x; d < hd; d += blockDim.x) {
        float acc = 0.0f;
        for (unsigned int s = 0; s < S; s++) {
            const float* p = partials + ((size_t)h * S + s) * stride;
            if (p[1] > 0.0f) {
                acc += p[2 + d] * __expf(p[0] - m_g);
            }
        }
        out[h * hd + d] = __float2half(acc * inv_l);
    }
}
