// delta_rule.cu — CUDA compute kernels для DeltaNet delta_rule (per-token decode).
//
// Прямой порт Metal-ядер из Yttri (modules/ai/local_llm/metal/delta_rule.metal).
// Математика и layout идентичны Metal-референсу — это portable-эквивалент
// fused GPU-пути для Windows/Linux (NVIDIA), устраняющий GPU↔CPU sync на
// рекуррентном шаге DeltaNet.
//
// Qwen3.5-4B DeltaNet: 24 рекуррентных слоя, каждый содержит:
// 1. QMatMul проекции (Candle CUDA, уже на GPU)
// 2. Prep: conv1d + sigmoid + softplus + L2 norm + head expansion
// 3. Delta rule: decay + sk + delta + rank-1 update + output
// 4. Norm + gate: Group RMS norm + SiLU(z) gating
//
// Размерности (Qwen3.5-4B):
//   n_k_heads = 16, n_v_heads = 32
//   head_k_dim = 128, head_v_dim = 128
//   key_dim = 2048, value_dim = 4096
//   channels = 8192 (key_dim*2 + value_dim)
//   conv_kernel = 4
//
// ВАЖНО (precision): per-token decode kernels работают в F32. Long-prompt
// тесты на Metal показали: F16 в decode накапливает cumulative drift, модель
// сразу выдаёт EOS на первом decode-токене после большого prefill. F16 живёт
// только в prefill/batch (отдельный fused-kernel).

#include "cuda_utils.cuh"
#include <stdint.h>

// ═══════════════════════════════════════════════════════════════
// Параметры (layout ДОЛЖЕН совпадать со struct DeltaParams в Rust:
//   #[repr(C)] 11 полей u32/f32, 44 байта). Передаётся по значению.
// ═══════════════════════════════════════════════════════════════

struct DeltaParams {
    unsigned int n_k_heads;   // 16
    unsigned int n_v_heads;   // 32
    unsigned int head_k_dim;  // 128
    unsigned int head_v_dim;  // 128
    unsigned int key_dim;     // n_k_heads * head_k_dim = 2048
    unsigned int value_dim;   // n_v_heads * head_v_dim = 4096
    unsigned int channels;    // key_dim * 2 + value_dim = 8192
    unsigned int conv_kernel; // 4
    float q_scale;            // 1 / sqrt(head_k_dim)
    float rms_norm_eps;       // 1e-6
    unsigned int heads_per_kv;// n_v_heads / n_k_heads = 2
};

// ═══════════════════════════════════════════════════════════════
// Вспомогательные функции (F32, single-precision math)
// ═══════════════════════════════════════════════════════════════

__device__ __forceinline__ float silu_f(float x) {
    return x / (1.0f + __expf(-x));
}

__device__ __forceinline__ float sigmoid_f(float x) {
    return 1.0f / (1.0f + __expf(-x));
}

__device__ __forceinline__ float softplus_f(float x) {
    // log(1 + exp(x)), численно стабильная версия (совпадает с Metal).
    if (x > 20.0f) return x;
    if (x < -20.0f) return 0.0f;
    return logf(1.0f + __expf(x));
}

// ═══════════════════════════════════════════════════════════════
// Kernel 1: delta_conv1d_prep
//
// Conv1d step + SiLU + sigmoid(beta) + softplus(alpha)*A
//
// Launch: grid=(ceil(channels/256),1,1), block=(256,1,1)
// Глобальный tid; guard `tid < channels` (как Metal dispatch_threads).
//
// conv_state layout = [(conv_k-1), channels] row-major
// conv_weights layout = [channels, conv_k] row-major
// ═══════════════════════════════════════════════════════════════
extern "C" __global__ void delta_conv1d_prep(
    const float* __restrict__ qkv_raw,      // [channels] — from QMatMul
    const float* __restrict__ beta_raw,     // [n_v_heads]
    const float* __restrict__ alpha_raw,    // [n_v_heads]
    const float* __restrict__ conv_weights, // [channels * conv_k]
    const float* __restrict__ dt_bias,      // [n_v_heads]
    const float* __restrict__ ssm_a,        // [n_v_heads]
    float* __restrict__ conv_state,         // [(conv_k-1) * channels] — persistent
    float* __restrict__ qkv_conv_out,       // [channels] — output
    float* __restrict__ beta_out,           // [n_v_heads]
    float* __restrict__ gate_out,           // [n_v_heads]
    const DeltaParams params
) {
    const unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int channels = params.channels;
    const unsigned int conv_k = params.conv_kernel;

    // ── Часть A: Conv1d step (по одному потоку на канал) ──
    if (tid < channels) {
        float sum = 0.0f;
        unsigned int weight_base = tid * conv_k;

        // Свёртка с предыдущими входами из буфера
        for (unsigned int i = 0; i < conv_k - 1; i++) {
            sum += conv_state[i * channels + tid] * conv_weights[weight_base + i];
        }
        // Текущий вход
        sum += qkv_raw[tid] * conv_weights[weight_base + conv_k - 1];

        // SiLU активация
        qkv_conv_out[tid] = silu_f(sum);

        // Обновление conv_state: сдвиг влево + запись нового входа
        if (conv_k > 2) {
            for (unsigned int i = 0; i < conv_k - 2; i++) {
                conv_state[i * channels + tid] = conv_state[(i + 1) * channels + tid];
            }
        }
        conv_state[(conv_k - 2) * channels + tid] = qkv_raw[tid];
    }

    // ── Часть B: sigmoid(beta) и softplus(alpha)*A (первые n_v_heads потоков) ──
    if (tid < params.n_v_heads) {
        beta_out[tid] = sigmoid_f(beta_raw[tid]);
        float alpha_biased = alpha_raw[tid] + dt_bias[tid];
        gate_out[tid] = softplus_f(alpha_biased) * ssm_a[tid];
    }
}

// ═══════════════════════════════════════════════════════════════
// Kernel 2: delta_l2_norm_expand
//
// L2 норм Q и K per k_head, расширение голов 16→32, Q scaling, copy V.
//
// Launch: grid=(n_v_heads,1,1), block=(head_k_dim,1,1)=(128)
//   blockIdx.x = head_v (0..31), threadIdx.x = dim (0..127)
// Shared memory tree-reduction (sum of squares) внутри блока.
// ═══════════════════════════════════════════════════════════════
extern "C" __global__ void delta_l2_norm_expand(
    const float* __restrict__ qkv_conv, // [channels] — conv1d output
    float* __restrict__ q_out,          // [n_v_heads * head_k_dim]
    float* __restrict__ k_out,          // [n_v_heads * head_k_dim]
    float* __restrict__ v_out,          // [n_v_heads * head_v_dim]
    const DeltaParams params
) {
    const unsigned int head_v = blockIdx.x;   // 0..31
    const unsigned int dim = threadIdx.x;     // 0..127
    const unsigned int hkd = params.head_k_dim; // 128
    const unsigned int hvd = params.head_v_dim; // 128
    const unsigned int key_dim = params.key_dim;

    // GQA: v-head → k-head через modulo (совпадает с Metal/production)
    const unsigned int head_k = head_v % params.n_k_heads;

    __shared__ float sq_sum_q[128];
    __shared__ float sq_sum_k[128];

    // Q начинается с 0, K — после Q (key_dim) в qkv_conv
    unsigned int q_idx = head_k * hkd + dim;
    unsigned int k_idx = key_dim + head_k * hkd + dim;

    float q_val = qkv_conv[q_idx];
    float k_val = qkv_conv[k_idx];

    sq_sum_q[dim] = q_val * q_val;
    sq_sum_k[dim] = k_val * k_val;

    __syncthreads();

    // Tree reduction (128 → 64 → ... → 1)
    for (unsigned int stride = hkd / 2; stride > 0; stride >>= 1) {
        if (dim < stride) {
            sq_sum_q[dim] += sq_sum_q[dim + stride];
            sq_sum_k[dim] += sq_sum_k[dim + stride];
        }
        __syncthreads();
    }

    float q_inv_norm = rsqrtf(sq_sum_q[0] + 1e-6f);
    float k_inv_norm = rsqrtf(sq_sum_k[0] + 1e-6f);

    unsigned int out_idx = head_v * hkd + dim;
    q_out[out_idx] = q_val * q_inv_norm * params.q_scale;
    k_out[out_idx] = k_val * k_inv_norm;

    // Copy V (без нормализации). V — после Q и K в qkv_conv.
    unsigned int v_src_idx = key_dim * 2 + head_v * hvd + dim;
    unsigned int v_dst_idx = head_v * hvd + dim;
    v_out[v_dst_idx] = qkv_conv[v_src_idx];
}

// ═══════════════════════════════════════════════════════════════
// Kernel 3: delta_rule_kernel
//
// Рекуррентный шаг DeltaNet. Каждый поток владеет одним столбцом state.
//
// Launch: grid=(n_v_heads,1,1), block=(head_v_dim,1,1)=(128)
//   blockIdx.x = head (0..31), threadIdx.x = col (0..127)
//
// state layout: [n_v_heads * hd * hd], per head row-major [hd × hd]
//   state[head][row][col] = ssm_state[head*hd*hd + row*hd + col]
//
// 1. Decay:  state[row][col] *= exp(gate[head])
// 2. sk[col] = sum_row(state[row][col] * k[row])     (S^T @ k)
// 3. d[col]  = (v[col] - sk[col]) * beta[head]
// 4. state[row][col] += k[row] * d[col]               (rank-1 update)
// 5. out[col] = sum_row(state[row][col] * q[row])     (S^T @ q)
// ═══════════════════════════════════════════════════════════════
extern "C" __global__ void delta_rule_kernel(
    const float* __restrict__ q,    // [n_v_heads * head_k_dim]
    const float* __restrict__ k,    // [n_v_heads * head_k_dim]
    const float* __restrict__ v,    // [n_v_heads * head_v_dim]
    const float* __restrict__ beta, // [n_v_heads]
    const float* __restrict__ gate, // [n_v_heads]
    float* __restrict__ ssm_state,  // [n_v_heads * hd * hd] — persistent
    float* __restrict__ output,     // [n_v_heads * head_v_dim]
    const DeltaParams params
) {
    const unsigned int head = blockIdx.x;   // 0..31
    const unsigned int col = threadIdx.x;   // 0..127
    const unsigned int hd = params.head_v_dim; // 128

    const unsigned int state_base = head * hd * hd;
    const unsigned int vec_base = head * hd;

    __shared__ float shared_sk[128];
    __shared__ float shared_d[128];

    // ── 1. Decay: каждый поток множит свой столбец (все строки) на exp(gate) ──
    float gate_exp = __expf(gate[head]);
    for (unsigned int row = 0; row < hd; row++) {
        ssm_state[state_base + row * hd + col] *= gate_exp;
    }
    // Барьер не нужен: каждый поток читает/пишет только свой столбец.

    // ── 2. sk[col] = sum_row(state[row][col] * k[row]) ──
    float sk_val = 0.0f;
    for (unsigned int row = 0; row < hd; row++) {
        sk_val += ssm_state[state_base + row * hd + col] * k[vec_base + row];
    }
    shared_sk[col] = sk_val;

    __syncthreads();

    // ── 3. d[col] = (v[col] - sk[col]) * beta[head] ──
    float beta_h = beta[head];
    float d_val = (v[vec_base + col] - shared_sk[col]) * beta_h;
    shared_d[col] = d_val;

    __syncthreads();

    // ── 4. Rank-1 update: state[row][col] += k[row] * d[col] ──
    float d_col = shared_d[col];
    for (unsigned int row = 0; row < hd; row++) {
        ssm_state[state_base + row * hd + col] += k[vec_base + row] * d_col;
    }

    // ── 5. out[col] = sum_row(state[row][col] * q[row]) ──
    float out_val = 0.0f;
    for (unsigned int row = 0; row < hd; row++) {
        out_val += ssm_state[state_base + row * hd + col] * q[vec_base + row];
    }
    output[vec_base + col] = out_val;
}

// ═══════════════════════════════════════════════════════════════
// Kernel 4: delta_norm_gate_kernel
//
// Group RMS Norm per head + SiLU(z) gating.
//
// Launch: grid=(n_v_heads,1,1), block=(head_v_dim,1,1)=(128)
//   blockIdx.x = head (0..31), threadIdx.x = dim (0..127)
//
// 1. sq_mean = mean(out[head]^2) — shared reduction
// 2. inv_rms = 1 / sqrt(sq_mean + eps)
// 3. gated[i] = out[i] * inv_rms * norm_weight[i % hvd] * silu(z[i])
// ═══════════════════════════════════════════════════════════════
extern "C" __global__ void delta_norm_gate_kernel(
    const float* __restrict__ raw_output,  // [n_v_heads * head_v_dim]
    const float* __restrict__ z,           // [value_dim]
    const float* __restrict__ norm_weight, // [head_v_dim] — shared across heads
    float* __restrict__ gated_output,      // [value_dim]
    const DeltaParams params
) {
    const unsigned int head = blockIdx.x;  // 0..31
    const unsigned int dim = threadIdx.x;  // 0..127
    const unsigned int hvd = params.head_v_dim;
    const float eps = params.rms_norm_eps;

    const unsigned int idx = head * hvd + dim;

    __shared__ float sq_vals[128];

    float val = raw_output[idx];
    sq_vals[dim] = val * val;

    __syncthreads();

    for (unsigned int stride = hvd / 2; stride > 0; stride >>= 1) {
        if (dim < stride) {
            sq_vals[dim] += sq_vals[dim + stride];
        }
        __syncthreads();
    }

    float inv_rms = rsqrtf(sq_vals[0] / (float)hvd + eps);

    float normed = val * inv_rms * norm_weight[dim];
    float z_val = z[idx];
    gated_output[idx] = normed * silu_f(z_val);
}

// ═══════════════════════════════════════════════════════════════
// PREFILL (fused, single-slot): вся последовательность за 4 launch'а
// вместо 4 × T (token-by-token). Рекуррентность сохраняется циклом
// внутри kernel 3 (state в global, горячо в L2).
// ═══════════════════════════════════════════════════════════════

// P1: causal depthwise conv1d + SiLU по всей последовательности + beta/gate prep.
// grid = (ceil(channels/256), T), block = (256).
// Хвост до t<conv_k-1 читается из persistent conv_state.
extern "C" __global__ void delta_conv1d_prefill(
    const float* __restrict__ qkv_raw,      // [T * channels]
    const float* __restrict__ beta_raw,     // [T * n_v_heads]
    const float* __restrict__ alpha_raw,    // [T * n_v_heads]
    const float* __restrict__ conv_weights, // [channels * conv_k]
    const float* __restrict__ dt_bias,      // [n_v_heads]
    const float* __restrict__ ssm_a,        // [n_v_heads]
    const float* __restrict__ conv_state,   // [(conv_k-1) * channels] persistent
    float* __restrict__ qkv_conv_out,       // [T * channels]
    float* __restrict__ beta_out,           // [T * n_v_heads]
    float* __restrict__ gate_out,           // [T * n_v_heads]
    const DeltaParams params,
    const unsigned int T
) {
    const unsigned int t = blockIdx.y;
    const unsigned int ch = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int channels = params.channels;
    const unsigned int conv_k = params.conv_kernel;
    const unsigned int n_v = params.n_v_heads;
    if (t >= T) return;

    if (ch < channels) {
        float sum = 0.0f;
        for (unsigned int j = 0; j < conv_k; j++) {
            const int src = (int)t - (int)(conv_k - 1) + (int)j;
            float x;
            if (src >= 0) {
                x = qkv_raw[(unsigned int)src * channels + ch];
            } else {
                // хвост persistent conv_state: индексы [0, conv_k-1)
                x = conv_state[(unsigned int)((int)(conv_k - 1) + src) * channels + ch];
            }
            sum += x * conv_weights[ch * conv_k + j];
        }
        qkv_conv_out[t * channels + ch] = silu_f(sum);
    }
    if (ch < n_v) {
        const unsigned int idx = t * n_v + ch;
        beta_out[idx] = sigmoid_f(beta_raw[idx]);
        gate_out[idx] = softplus_f(alpha_raw[idx] + dt_bias[ch]) * ssm_a[ch];
    }
}

// P2: L2 norm Q/K + expand + scale по всей последовательности.
// grid = (n_v_heads, T), block = (head_k_dim).
extern "C" __global__ void delta_l2_norm_prefill(
    const float* __restrict__ qkv_conv, // [T * channels]
    float* __restrict__ q_out,          // [T * n_v_heads * head_k_dim]
    float* __restrict__ k_out,          // [T * n_v_heads * head_k_dim]
    float* __restrict__ v_out,          // [T * n_v_heads * head_v_dim]
    const DeltaParams params,
    const unsigned int T
) {
    const unsigned int head_v = blockIdx.x;
    const unsigned int t = blockIdx.y;
    const unsigned int dim = threadIdx.x;
    if (t >= T) return;
    const unsigned int hkd = params.head_k_dim;
    const unsigned int hvd = params.head_v_dim;
    const unsigned int key_dim = params.key_dim;
    const unsigned int n_v = params.n_v_heads;
    const unsigned int channels = params.channels;
    const unsigned int head_k = head_v % params.n_k_heads;

    const unsigned int base = t * channels;
    __shared__ float sq_q[128];
    __shared__ float sq_k[128];

    const float q_val = qkv_conv[base + head_k * hkd + dim];
    const float k_val = qkv_conv[base + key_dim + head_k * hkd + dim];
    sq_q[dim] = q_val * q_val;
    sq_k[dim] = k_val * k_val;
    __syncthreads();
    for (unsigned int stride = hkd / 2; stride > 0; stride >>= 1) {
        if (dim < stride) {
            sq_q[dim] += sq_q[dim + stride];
            sq_k[dim] += sq_k[dim + stride];
        }
        __syncthreads();
    }
    const float q_inv = rsqrtf(sq_q[0] + 1e-6f);
    const float k_inv = rsqrtf(sq_k[0] + 1e-6f);

    q_out[(t * n_v + head_v) * hkd + dim] = q_val * q_inv * params.q_scale;
    k_out[(t * n_v + head_v) * hkd + dim] = k_val * k_inv;
    v_out[(t * n_v + head_v) * hvd + dim] = qkv_conv[base + key_dim * 2 + head_v * hvd + dim];
}

// P3: рекуррентный delta rule по всей последовательности — цикл внутри kernel.
// STATE В РЕГИСТРАХ (схема llama.cpp gated_delta_net.cu): warp владеет
// колонкой state, каждый lane держит 4 строки (hd=128/32) в регистрах на
// весь цикл — нулевой global state traffic между токенами. Глобальный state
// читается один раз в начале и пишется один раз в конце.
// grid = (n_v_heads, hd/4), block = (32, 4): col = blockIdx.y*4 + threadIdx.y.
extern "C" __global__ void delta_rule_prefill(
    const float* __restrict__ q,     // [T * n_v * hkd]
    const float* __restrict__ k,     // [T * n_v * hkd]
    const float* __restrict__ v,     // [T * n_v * hvd]
    const float* __restrict__ beta,  // [T * n_v]
    const float* __restrict__ gate,  // [T * n_v]
    float* __restrict__ ssm_state,   // [n_v * hd * hd] persistent
    float* __restrict__ output,      // [T * n_v * hvd]
    const DeltaParams params,
    const unsigned int T
) {
    const unsigned int head = blockIdx.x;
    const unsigned int col = blockIdx.y * blockDim.y + threadIdx.y;
    const unsigned int lane = threadIdx.x;
    const unsigned int hd = params.head_v_dim;   // 128
    const unsigned int n_v = params.n_v_heads;
    const unsigned int hkd = params.head_k_dim;
    constexpr unsigned int ROWS = 4;             // hd / warp_size = 128/32

    const unsigned int state_base = head * hd * hd;

    // Начальная загрузка state-шарда в регистры (col-я колонка, 4 строки).
    float s[ROWS];
    #pragma unroll
    for (unsigned int r = 0; r < ROWS; r++) {
        const unsigned int row = r * 32 + lane;
        s[r] = ssm_state[state_base + row * hd + col];
    }

    for (unsigned int t = 0; t < T; t++) {
        const unsigned int kv_base = (t * n_v + head) * hkd;
        const unsigned int out_base = (t * n_v + head) * hd;
        const float g = __expf(gate[t * n_v + head]);
        const float beta_h = beta[t * n_v + head];

        float k_reg[ROWS], q_reg[ROWS];
        #pragma unroll
        for (unsigned int r = 0; r < ROWS; r++) {
            const unsigned int row = r * 32 + lane;
            k_reg[r] = k[kv_base + row];
            q_reg[r] = q[kv_base + row];
        }

        // kv_col = (S^T k)[col] = Σ_row S[row][col]·k[row]
        float kv_part = 0.0f;
        #pragma unroll
        for (unsigned int r = 0; r < ROWS; r++) kv_part += s[r] * k_reg[r];
        float kv_col = kv_part;
        for (int o = 16; o > 0; o >>= 1) kv_col += __shfl_down_sync(0xffffffff, kv_col, o);
        // broadcast в warp
        kv_col = __shfl_sync(0xffffffff, kv_col, 0);

        // delta = (v[col] - g·kv_col)·beta
        const float v_col = v[out_base + col];
        const float delta_col = (v_col - g * kv_col) * beta_h;

        // S = g·S + k·delta^T; attn = (S^T q)[col]
        float attn_part = 0.0f;
        #pragma unroll
        for (unsigned int r = 0; r < ROWS; r++) {
            s[r] = g * s[r] + k_reg[r] * delta_col;
            attn_part += s[r] * q_reg[r];
        }
        float attn_col = attn_part;
        for (int o = 16; o > 0; o >>= 1) attn_col += __shfl_down_sync(0xffffffff, attn_col, o);
        if (lane == 0) output[out_base + col] = attn_col;
    }

    // Финальная запись state.
    #pragma unroll
    for (unsigned int r = 0; r < ROWS; r++) {
        const unsigned int row = r * 32 + lane;
        ssm_state[state_base + row * hd + col] = s[r];
    }
}

// P4: group RMS norm + SiLU(z) gate по всей последовательности.
// grid = (n_v_heads, T), block = (head_v_dim).
extern "C" __global__ void delta_norm_gate_prefill(
    const float* __restrict__ raw_output,  // [T * n_v * hvd]
    const float* __restrict__ z,           // [T * value_dim]
    const float* __restrict__ norm_weight, // [hvd]
    float* __restrict__ gated_output,      // [T * value_dim]
    const DeltaParams params,
    const unsigned int T
) {
    const unsigned int head = blockIdx.x;
    const unsigned int t = blockIdx.y;
    const unsigned int dim = threadIdx.x;
    if (t >= T) return;
    const unsigned int hvd = params.head_v_dim;
    const unsigned int n_v = params.n_v_heads;
    const float eps = params.rms_norm_eps;

    const unsigned int idx = (t * n_v + head) * hvd + dim;
    const unsigned int z_idx = t * params.value_dim + head * hvd + dim;

    __shared__ float sq_vals[128];
    const float val = raw_output[idx];
    sq_vals[dim] = val * val;
    __syncthreads();
    for (unsigned int stride = hvd / 2; stride > 0; stride >>= 1) {
        if (dim < stride) sq_vals[dim] += sq_vals[dim + stride];
        __syncthreads();
    }
    const float inv_rms = rsqrtf(sq_vals[0] / (float)hvd + eps);
    gated_output[z_idx] = val * inv_rms * norm_weight[dim] * silu_f(z[z_idx]);
}
