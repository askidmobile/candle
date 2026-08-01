// delta_rule_batched.cu — CUDA compute kernels для batched DeltaNet decode.
//
// Phase 2: true batched decode с осью slot B. Прямой порт Metal-ядер из
// delta_rule_batched.metal. Каждый (head, slot) независим (decode не имеет
// sequential token loop, в отличие от fused prefill). Math per (head, slot)
// идентична per-head math из delta_rule.cu — добавлена только slot-индексация
// в state/temp буферах и grid.
//
// Размерности (Qwen3.5-4B):
//   n_k_heads = 16, n_v_heads = 32
//   head_k_dim = 128, head_v_dim = 128
//   key_dim = 2048, value_dim = 4096
//   channels = 8192, conv_kernel = 4
//   B = batch_size (slot count), 1..MAX_BATCH
//
// Layout буферов (leading B):
//   ssm_state   [B * n_v_heads * hd * hd]
//   conv_state  [B * (conv_k-1) * channels]
//   qkv_conv    [B * channels]
//   q/k/v       [B * n_v_heads * head_{k,v}_dim]
//   beta/gate   [B * n_v_heads]
//   delta_out   [B * n_v_heads * head_v_dim]
//   gated_out   [B * value_dim]
// Веса (conv_weights, dt_bias, ssm_a, norm_weight) — shared across slots, без B.
// Входы batched QMatMul: qkv_raw [B*channels], z_raw [B*value_dim],
//   beta_raw [B*n_v_heads], alpha_raw [B*n_v_heads] (row-major, leading B).
//
// Precision: F32 (decode precision critical — см. delta_rule.cu).

#include "cuda_utils.cuh"
#include <stdint.h>

// ═══════════════════════════════════════════════════════════════
// Параметры (layout ДОЛЖЕН совпадать со struct DeltaParams в Rust:
//   #[repr(C)] 12 полей u32/f32 = 48 байтов). Передаётся по значению.
// ═══════════════════════════════════════════════════════════════

struct DeltaParams {
    unsigned int n_k_heads;     // 16
    unsigned int n_v_heads;     // 32
    unsigned int head_k_dim;    // 128
    unsigned int head_v_dim;    // 128
    unsigned int key_dim;       // n_k_heads * head_k_dim = 2048
    unsigned int value_dim;     // n_v_heads * head_v_dim = 4096
    unsigned int channels;      // key_dim * 2 + value_dim = 8192
    unsigned int conv_kernel;   // 4
    float q_scale;              // 1 / sqrt(head_k_dim)
    float rms_norm_eps;         // 1e-6
    unsigned int heads_per_kv;  // n_v_heads / n_k_heads = 2 (GQA)
    unsigned int batch_size;    // B — число одновременных слотов
};

// ═══════════════════════════════════════════════════════════════
// Вспомогательные функции (копия из delta_rule.cu — identical math)
// ═══════════════════════════════════════════════════════════════

__device__ __forceinline__ float silu_f(float x) {
    return x / (1.0f + __expf(-x));
}

__device__ __forceinline__ float sigmoid_f(float x) {
    return 1.0f / (1.0f + __expf(-x));
}

__device__ __forceinline__ float softplus_f(float x) {
    // log(1 + exp(x)), численно стабильная версия
    if (x > 20.0f) return x;
    if (x < -20.0f) return 0.0f;
    return logf(1.0f + __expf(x));
}

// ═══════════════════════════════════════════════════════════════
// Kernel 1: delta_conv1d_prep_batched
//
// Conv1d step + SiLU + sigmoid(beta) + softplus(alpha)*A, per slot.
//
// Launch: grid=(ceil(channels/256), B, 1), block=(256,1,1)
//   blockIdx.x = channel block, blockIdx.y = slot
//   Глобальный (ch, slot): ch = blockIdx.x*blockDim.x + threadIdx.x, slot = blockIdx.y
//
// conv_state layout = [B, (conv_k-1), channels] row-major
// conv_weights layout = [channels, conv_k] row-major (shared, без B)
// ═══════════════════════════════════════════════════════════════

extern "C" __global__ void delta_conv1d_prep_batched(
    const float* __restrict__ qkv_raw,      // [B * channels] — from batched QMatMul
    const float* __restrict__ beta_raw,     // [B * n_v_heads]
    const float* __restrict__ alpha_raw,    // [B * n_v_heads]
    const float* __restrict__ conv_weights, // [channels * conv_k] — shared
    const float* __restrict__ dt_bias,      // [n_v_heads] — shared
    const float* __restrict__ ssm_a,        // [n_v_heads] — shared
    float* __restrict__ conv_state,         // [B * (conv_k-1) * channels] — persistent
    float* __restrict__ qkv_conv_out,       // [B * channels] — output
    float* __restrict__ beta_out,           // [B * n_v_heads]
    float* __restrict__ gate_out,           // [B * n_v_heads]
    const DeltaParams params
) {
    const unsigned int ch = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int slot = blockIdx.y;
    const unsigned int channels = params.channels;
    const unsigned int conv_k = params.conv_kernel;
    const unsigned int n_v = params.n_v_heads;

    if (slot >= params.batch_size) return;

    // ── Часть A: Conv1d step (per (ch, slot)) ──
    if (ch < channels) {
        const unsigned int slot_conv_base = slot * (conv_k - 1) * channels;
        const unsigned int slot_qkv_base = slot * channels;

        float sum = 0.0f;
        const unsigned int weight_base = ch * conv_k;

        // Свёртка с предыдущими входами из буфера (этого slot'а)
        for (unsigned int i = 0; i < conv_k - 1; i++) {
            sum += conv_state[slot_conv_base + i * channels + ch] * conv_weights[weight_base + i];
        }
        // Текущий вход
        sum += qkv_raw[slot_qkv_base + ch] * conv_weights[weight_base + conv_k - 1];

        // SiLU активация
        qkv_conv_out[slot_qkv_base + ch] = silu_f(sum);

        // Обновление conv_state для этого slot'а: сдвиг влево + запись нового входа
        if (conv_k > 2) {
            for (unsigned int i = 0; i < conv_k - 2; i++) {
                conv_state[slot_conv_base + i * channels + ch] =
                    conv_state[slot_conv_base + (i + 1) * channels + ch];
            }
        }
        conv_state[slot_conv_base + (conv_k - 2) * channels + ch] = qkv_raw[slot_qkv_base + ch];
    }

    // ── Часть B: sigmoid(beta) и softplus(alpha)*A (per (slot, head)) ──
    // Первые n_v_heads потоков в каждом блоке делают beta/gate для этого slot'а.
    // blockDim.x = 256 >= n_v (32 в Qwen3.5-4B), поэтому в каждом блоке потоки
    // с threadIdx.x (== ch в части A) < n_v покрывают beta/gate для своего slot.
    if (ch < n_v) {
        const unsigned int slot_head = slot * n_v + ch;
        beta_out[slot_head] = sigmoid_f(beta_raw[slot_head]);
        float alpha_biased = alpha_raw[slot_head] + dt_bias[ch];
        gate_out[slot_head] = softplus_f(alpha_biased) * ssm_a[ch];
    }
}

// ═══════════════════════════════════════════════════════════════
// Kernel 2: delta_l2_norm_expand_batched
//
// L2 норм Q/K per k_head, расширение голов 16→32, Q scaling, copy V — per (head, slot).
//
// Launch: grid=(n_v_heads, B, 1), block=(head_k_dim, 1, 1)
//   blockIdx.x = head_v (0..31), blockIdx.y = slot (0..B-1)
//   threadIdx.x = dim (0..127)
// Shared memory tree-reduction внутри блока (= один (v_head, slot)).
// ═══════════════════════════════════════════════════════════════

extern "C" __global__ void delta_l2_norm_expand_batched(
    const float* __restrict__ qkv_conv, // [B * channels] — conv1d output
    float* __restrict__ q_out,          // [B * n_v_heads * head_k_dim]
    float* __restrict__ k_out,          // [B * n_v_heads * head_k_dim]
    float* __restrict__ v_out,          // [B * n_v_heads * head_v_dim]
    const DeltaParams params
) {
    const unsigned int head_v = blockIdx.x;   // 0..31
    const unsigned int slot = blockIdx.y;      // 0..B-1
    const unsigned int dim = threadIdx.x;      // 0..127
    const unsigned int hkd = params.head_k_dim; // 128
    const unsigned int hvd = params.head_v_dim; // 128
    const unsigned int key_dim = params.key_dim;
    const unsigned int n_v = params.n_v_heads;

    // slot offsets (leading B)
    const unsigned int slot_qkv = slot * params.channels;
    const unsigned int slot_q = slot * n_v * hkd;
    const unsigned int slot_k = slot * n_v * hkd;
    const unsigned int slot_v = slot * n_v * hvd;

    // GQA: v-head → k-head (same as original)
    const unsigned int head_k = head_v % params.n_k_heads;

    __shared__ float sq_sum_q[128];
    __shared__ float sq_sum_k[128];

    // Q начинается с 0, K — после Q (key_dim) в qkv_conv (per slot)
    unsigned int q_idx = slot_qkv + head_k * hkd + dim;
    unsigned int k_idx = slot_qkv + key_dim + head_k * hkd + dim;

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

    // Нормализация + Q scaling + запись (per v-head, per slot)
    unsigned int out_idx = slot_q + head_v * hkd + dim;
    q_out[out_idx] = q_val * q_inv_norm * params.q_scale;
    k_out[slot_k + head_v * hkd + dim] = k_val * k_inv_norm;

    // Copy V (без нормализации) — per v-head, per slot
    unsigned int v_src_idx = slot_qkv + key_dim * 2 + head_v * hvd + dim;
    unsigned int v_dst_idx = slot_v + head_v * hvd + dim;
    v_out[v_dst_idx] = qkv_conv[v_src_idx];
}

// ═══════════════════════════════════════════════════════════════
// Kernel 3: delta_rule_kernel_batched
//
// Рекуррентный шаг DeltaNet, per (head, slot).
//
// Launch: grid=(n_v_heads, B, 1), block=(head_v_dim, 1, 1)
//   blockIdx.x = head (0..31), blockIdx.y = slot (0..B-1)
//   threadIdx.x = col (0..127)
//
// state layout: [B * n_v_heads * hd * hd], per (slot, head) = [hd × hd] row-major
//   state[slot][head][row][col] = ssm_state[slot*n_v*hd*hd + head*hd*hd + row*hd + col]
//
// 1. Decay:  state[row][col] *= exp(gate[slot][head])
// 2. sk[col] = sum_row(state[row][col] * k[slot][head][row])  (S^T @ k)
// 3. d[col]  = (v[slot][head][col] - sk[col]) * beta[slot][head])
// 4. state[row][col] += k[slot][head][row] * d[col]            (rank-1 update)
// 5. out[col] = sum_row(state[row][col] * q[slot][head][row]) (S^T @ q)
//
// Shared memory reduction внутри блока (= один (head, slot)), изоляция автоматична.
// ═══════════════════════════════════════════════════════════════

extern "C" __global__ void delta_rule_kernel_batched(
    const float* __restrict__ q,    // [B * n_v_heads * head_k_dim]
    const float* __restrict__ k,    // [B * n_v_heads * head_k_dim]
    const float* __restrict__ v,    // [B * n_v_heads * head_v_dim]
    const float* __restrict__ beta, // [B * n_v_heads]
    const float* __restrict__ gate, // [B * n_v_heads]
    float* __restrict__ ssm_state,  // [B * n_v_heads * hd * hd] — persistent
    float* __restrict__ output,     // [B * n_v_heads * head_v_dim]
    const DeltaParams params
) {
    const unsigned int head = blockIdx.x;   // 0..31
    const unsigned int slot = blockIdx.y;   // 0..B-1
    const unsigned int col = threadIdx.x;   // 0..127
    const unsigned int hd = params.head_v_dim; // 128
    const unsigned int n_v = params.n_v_heads;

    // Базовые смещения (leading B)
    const unsigned int slot_head = slot * n_v + head;
    const unsigned int state_base = slot * n_v * hd * hd + head * hd * hd;
    const unsigned int vec_base = slot * n_v * hd + head * hd;

    __shared__ float shared_sk[128];
    __shared__ float shared_d[128];

    // ── 1. Decay: state[row][col] *= exp(gate[slot][head]) ──
    float gate_exp = __expf(gate[slot_head]);
    for (unsigned int row = 0; row < hd; row++) {
        ssm_state[state_base + row * hd + col] *= gate_exp;
    }
    // Барьер не нужен: каждый поток читает/пишет только свой столбец.

    // ── 2. sk[col] = sum_row(state[row][col] * k[slot][head][row]) ──
    float sk_val = 0.0f;
    for (unsigned int row = 0; row < hd; row++) {
        sk_val += ssm_state[state_base + row * hd + col] * k[vec_base + row];
    }
    shared_sk[col] = sk_val;

    __syncthreads();

    // ── 3. d[col] = (v[slot][head][col] - sk[col]) * beta[slot][head] ──
    float beta_h = beta[slot_head];
    float d_val = (v[vec_base + col] - shared_sk[col]) * beta_h;
    shared_d[col] = d_val;

    __syncthreads();

    // ── 4. Rank-1 update: state[row][col] += k[slot][head][row] * d[col] ──
    float d_col = shared_d[col];
    for (unsigned int row = 0; row < hd; row++) {
        ssm_state[state_base + row * hd + col] += k[vec_base + row] * d_col;
    }

    // ── 5. out[slot][head][col] = sum_row(state[row][col] * q[slot][head][row]) ──
    float out_val = 0.0f;
    for (unsigned int row = 0; row < hd; row++) {
        out_val += ssm_state[state_base + row * hd + col] * q[vec_base + row];
    }
    output[vec_base + col] = out_val;
}

// ═══════════════════════════════════════════════════════════════
// Kernel 4: delta_norm_gate_kernel_batched
//
// Group RMS Norm per (head, slot) + SiLU(z) gating, per slot.
//
// Launch: grid=(n_v_heads, B, 1), block=(head_v_dim, 1, 1)
//   blockIdx.x = head (0..31), blockIdx.y = slot (0..B-1)
//   threadIdx.x = dim (0..127)
//
// 1. sq_mean = mean(out[slot][head]^2) — shared reduction
// 2. inv_rms = 1 / sqrt(sq_mean + eps)
// 3. gated[slot][head*hvd+dim] = out[..] * inv_rms * norm_weight[dim] * silu(z[slot*value_dim + head*hvd + dim])
// ═══════════════════════════════════════════════════════════════

extern "C" __global__ void delta_norm_gate_kernel_batched(
    const float* __restrict__ raw_output,  // [B * n_v_heads * head_v_dim]
    const float* __restrict__ z,           // [B * value_dim]
    const float* __restrict__ norm_weight, // [head_v_dim] — shared across slots/heads
    float* __restrict__ gated_output,      // [B * value_dim]
    const DeltaParams params
) {
    const unsigned int head = blockIdx.x;  // 0..31
    const unsigned int slot = blockIdx.y;  // 0..B-1
    const unsigned int dim = threadIdx.x;  // 0..127
    const unsigned int hvd = params.head_v_dim;
    const unsigned int n_v = params.n_v_heads;
    const float eps = params.rms_norm_eps;

    const unsigned int idx = slot * n_v * hvd + head * hvd + dim;     // raw_output per (slot, head)
    const unsigned int z_idx = slot * params.value_dim + head * hvd + dim;
    const unsigned int out_idx = slot * params.value_dim + head * hvd + dim;

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
    float z_val = z[z_idx];
    gated_output[out_idx] = normed * silu_f(z_val);
}