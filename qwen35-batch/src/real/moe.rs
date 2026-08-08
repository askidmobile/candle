//! Qwen3.6 35B-A3B MoE block — Phase 2 reference path.
//!
//! Router (top-k), packed routed experts, sigmoid-gated shared expert.
//! Reference backend dequantizes selected experts — diagnostics/parity only.
//! PTX backend is Phase 3.

use candle_core::quantized::{QMatMul, QTensor};
use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::{ops::softmax_last_dim, Linear};
use std::sync::Arc;

// ─── Forward mode ─────────────────────────────────────────────────────────────

/// Distinguishes chunked prefill from batched decode.
///
/// Mathematical semantics are identical; only workspace/scratch selection
/// differs in the PTX backend (Phase 3). Reference backend ignores it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ForwardMode {
    Prefill,
    DecodeBatch,
}

// ─── Backend selection ────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MoeBackend {
    /// CPU/Metal reference: dequantizes selected experts. Diagnostics/parity only.
    Reference,
    /// CUDA PTX sparse GEMM (Phase 3).
    Ptx,
}

// ─── Config ───────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct Qwen35MoeConfig {
    pub hidden_size: usize,
    pub n_experts: usize,
    pub n_experts_per_tok: usize,
    pub routed_intermediate: usize,
    pub shared_intermediate: usize,
    pub norm_topk_prob: bool,
}

// ─── Route plan ───────────────────────────────────────────────────────────────

/// Result of router top-k selection.
#[derive(Debug, Clone)]
pub struct RoutePlan {
    /// Selected expert indices per token: `[n_tokens][k]`.
    pub experts: Vec<Vec<usize>>,
    /// Normalized routing weights per token: `[n_tokens][k]`.
    pub weights: Vec<Vec<f32>>,
    /// Router logits `[n_tokens, n_experts]` F32.
    pub logits: Tensor,
    /// Softmax probabilities `[n_tokens, n_experts]` F32.
    pub probs: Tensor,
}

// ─── Router ───────────────────────────────────────────────────────────────────

/// F32-dequantized router linear with llama.cpp-compatible top-k.
pub struct MoeRouter {
    linear: Linear,
    n_experts: usize,
    n_experts_per_tok: usize,
    norm_topk_prob: bool,
}

impl MoeRouter {
    pub fn new(
        linear: Linear,
        n_experts: usize,
        n_experts_per_tok: usize,
        norm_topk_prob: bool,
    ) -> Self {
        Self {
            linear,
            n_experts,
            n_experts_per_tok,
            norm_topk_prob,
        }
    }

    /// Compute router logits, softmax, stable top-k, and weight normalization.
    ///
    /// Matches llama.cpp qwen3moe router transform:
    /// ```text
    /// logits   = mm(ffn_gate_inp, x)     → [n_tokens, n_experts]
    /// probs    = softmax(logits)
    /// selected = top_k(probs, k)         — stable, tie-break by lower expert index
    /// weights  = gather(probs, selected)
    /// if norm_topk_prob: weights /= sum(weights)
    /// ```
    ///
    /// `arg_sort_last_dim` is **unstable** and would break parity — we use
    /// manual iterative argmax on CPU (probs are small: n_tokens × n_experts).
    pub fn route_topk(&self, xs: &Tensor) -> Result<RoutePlan> {
        let (n_tokens, _) = xs.dims2()?;
        let logits = self.linear.forward(xs)?.to_dtype(DType::F32)?;
        let probs = softmax_last_dim(&logits)?;

        // Stable top-k on CPU.
        let probs_cpu = probs.to_device(&Device::Cpu)?;
        let probs_rows: Vec<Vec<f32>> = probs_cpu.to_vec2()?;

        let k = self.n_experts_per_tok.min(self.n_experts);
        let mut experts = Vec::with_capacity(n_tokens);
        let mut weights = Vec::with_capacity(n_tokens);

        for row in &probs_rows {
            let mut masked = vec![false; row.len()];
            let mut selected: Vec<(usize, f32)> = Vec::with_capacity(k);

            for _ in 0..k {
                // Strict `>` → first max wins → tie-break by lower expert index.
                let mut best_idx = 0usize;
                let mut best_val = f32::NEG_INFINITY;
                for (i, &p) in row.iter().enumerate() {
                    if !masked[i] && p > best_val {
                        best_val = p;
                        best_idx = i;
                    }
                }
                masked[best_idx] = true;
                selected.push((best_idx, best_val));
            }

            if self.norm_topk_prob {
                let sum: f32 = selected.iter().map(|(_, w)| *w).sum();
                if sum > 0.0 {
                    for (_, w) in &mut selected {
                        *w /= sum;
                    }
                }
            }

            experts.push(selected.iter().map(|(e, _)| *e).collect());
            weights.push(selected.iter().map(|(_, w)| *w).collect());
        }

        Ok(RoutePlan {
            experts,
            weights,
            logits,
            probs,
        })
    }
}

// ─── Packed experts ───────────────────────────────────────────────────────────

/// Packed routed expert weights (quantized, not dequantized at load time).
///
/// GGUF shapes (after dimension reversal on read):
/// - `gate`: `[n_experts, n_ff, n_embd]`
/// - `up`:   `[n_experts, n_ff, n_embd]`
/// - `down`: `[n_experts, n_embd, n_ff]`
pub struct PackedExperts {
    pub gate: Arc<QTensor>,
    pub up: Arc<QTensor>,
    pub down: Arc<QTensor>,
    pub n_experts: usize,
}

// ─── Shared expert ────────────────────────────────────────────────────────────

/// Sigmoid-gated shared SwiGLU expert.
///
/// llama.cpp qwen2moe pattern (qwen35moe shares it):
/// ```text
/// cur_gate = silu(mm(gate_inp, x)) / mm(gate_inp, x)   [= sigmoid, bit-exact]
/// cur_ffn  = down(silu(gate(x)) * up(x))
/// out      = cur_ffn * cur_gate
/// ```
pub struct SharedExpert {
    /// `ffn_gate_inp_shexp` → `[n_tokens, 1]` scalar gate.
    gate_inp: Linear,
    /// `ffn_gate_shexp`.
    gate: QMatMul,
    /// `ffn_up_shexp`.
    up: QMatMul,
    /// `ffn_down_shexp`.
    down: QMatMul,
}

impl SharedExpert {
    pub fn new(gate_inp: Linear, gate: QMatMul, up: QMatMul, down: QMatMul) -> Self {
        Self {
            gate_inp,
            gate,
            up,
            down,
        }
    }

    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let gate_logits = self.gate_inp.forward(xs)?; // [n_tokens, 1]
        let gate_scalar = silu_div(&gate_logits)?; // [n_tokens, 1]
        let gp = self.gate.forward(xs)?; // [n_tokens, n_ff_shexp]
        let up = self.up.forward(xs)?; // [n_tokens, n_ff_shexp]
        let act = gp.silu()?.mul(&up)?; // [n_tokens, n_ff_shexp]
        let ffn = self.down.forward(&act)?; // [n_tokens, n_embd]
        ffn.broadcast_mul(&gate_scalar)
    }
}

/// `silu(x) / x` — llama.cpp's exact gate computation (numerically = sigmoid).
///
/// At `x = 0` the limit is `0.5`; we return `0.5` to avoid `0/0` NaN.
/// Bit-exact with llama.cpp for all nonzero `x` (same op sequence: silu then div).
///
/// Gate logits are tiny (`[n_tokens, 1]`) — CPU round-trip is negligible.
fn silu_div(xs: &Tensor) -> Result<Tensor> {
    let device = xs.device();
    let (rows, cols) = xs.dims2()?;
    let xs_cpu = xs.to_device(&Device::Cpu)?.flatten_all()?;
    let vals: Vec<f32> = xs_cpu.to_vec1()?;
    let out: Vec<f32> = vals
        .iter()
        .map(|&x| {
            if x == 0.0 {
                0.5
            } else {
                let s = x / (1.0 + (-x).exp());
                s / x
            }
        })
        .collect();
    let t = Tensor::from_vec(out, (rows, cols), &Device::Cpu)?;
    t.to_device(device)
}

// ─── MoE block ────────────────────────────────────────────────────────────────

/// Qwen3.6 MoE feed-forward block.
pub struct Qwen35MoeBlock {
    router: MoeRouter,
    routed: PackedExperts,
    shared: SharedExpert,
    backend: MoeBackend,
    #[allow(dead_code)]
    cfg: Qwen35MoeConfig,
}

impl Qwen35MoeBlock {
    pub fn new(
        cfg: Qwen35MoeConfig,
        router: MoeRouter,
        routed: PackedExperts,
        shared: SharedExpert,
        backend: MoeBackend,
    ) -> Self {
        Self {
            router,
            routed,
            shared,
            backend,
            cfg,
        }
    }

    /// Forward: route → routed SwiGLU + shared expert → combine.
    ///
    /// `xs` shape: `[batch, seq_len, n_embd]` → output same shape.
    pub fn forward(&self, xs: &Tensor, mode: ForwardMode) -> Result<Tensor> {
        let (batch, seq_len, n_embd) = xs.dims3()?;
        let xs_2d = xs.reshape(((), n_embd))?;

        let route = self.router.route_topk(&xs_2d)?;
        let routed = self
            .backend
            .routed_swiglu(&xs_2d, &self.routed, &route, mode)?;
        let shared = self.shared.forward(&xs_2d)?;
        let combined = routed.broadcast_add(&shared)?;

        combined.reshape((batch, seq_len, n_embd))
    }

    /// Config accessor for diagnostics / adapter reporting.
    pub fn config(&self) -> &Qwen35MoeConfig {
        &self.cfg
    }
}

impl MoeBackend {
    fn routed_swiglu(
        &self,
        xs: &Tensor,
        experts: &PackedExperts,
        route: &RoutePlan,
        _mode: ForwardMode,
    ) -> Result<Tensor> {
        match self {
            Self::Reference => reference_routed_swiglu(xs, experts, route),
            Self::Ptx => candle_core::bail!(
                "PTX MoE backend not yet implemented (Phase 3). \
                 Set QWEN36_MOE_BACKEND=reference for diagnostics."
            ),
        }
    }
}

/// Reference routed expert SwiGLU: dequantize packed experts, batch per-expert.
///
/// Diagnostics/parity only — dequantizes the full packed tensor. Production
/// uses PTX sparse GEMM (Phase 3) which consumes packed storage directly.
///
/// Groups tokens by selected expert, batch-matmuls each expert's tokens,
/// weights the output, and scatter-adds (`index_add`) into the result.
fn reference_routed_swiglu(
    xs: &Tensor,
    experts: &PackedExperts,
    route: &RoutePlan,
) -> Result<Tensor> {
    let device = xs.device();
    let xs = xs.to_dtype(DType::F32)?;
    let (n_tokens, n_embd) = xs.dims2()?;
    let n_experts = experts.n_experts;

    // Dequantize packed experts to F32 on the compute device.
    let gate_f = experts.gate.dequantize(device)?.to_dtype(DType::F32)?;
    let up_f = experts.up.dequantize(device)?.to_dtype(DType::F32)?;
    let down_f = experts.down.dequantize(device)?.to_dtype(DType::F32)?;

    // Group tokens by selected expert: expert → [(token_idx, weight)].
    let mut expert_tokens: Vec<Vec<(usize, f32)>> = vec![Vec::new(); n_experts];
    for (t, (eks, ws)) in route.experts.iter().zip(route.weights.iter()).enumerate() {
        for (e, w) in eks.iter().zip(ws.iter()) {
            expert_tokens[*e].push((t, *w));
        }
    }

    let mut out = Tensor::zeros((n_tokens, n_embd), DType::F32, device)?;
    for e in 0..n_experts {
        let tokens = &expert_tokens[e];
        if tokens.is_empty() {
            continue;
        }
        let token_ids: Vec<u32> = tokens.iter().map(|(t, _)| *t as u32).collect();
        let weights: Vec<f32> = tokens.iter().map(|(_, w)| *w).collect();

        let idx = Tensor::from_vec(token_ids.clone(), token_ids.len(), device)?;
        let x_subset = xs.index_select(&idx, 0)?; // [n_sel, n_embd]

        let gate_w = gate_f.narrow(0, e, 1)?.squeeze(0)?; // [n_ff, n_embd]
        let up_w = up_f.narrow(0, e, 1)?.squeeze(0)?; // [n_ff, n_embd]
        let down_w = down_f.narrow(0, e, 1)?.squeeze(0)?; // [n_embd, n_ff]

        let gate_out = x_subset.matmul(&gate_w.t()?)?; // [n_sel, n_ff]
        let up_out = x_subset.matmul(&up_w.t()?)?; // [n_sel, n_ff]
        let act = gate_out.silu()?.mul(&up_out)?; // [n_sel, n_ff]
        let expert_out = act.matmul(&down_w.t()?)?; // [n_sel, n_embd]

        let w_len = weights.len();
        let w = Tensor::from_vec(weights, (w_len, 1), device)?;
        let weighted = expert_out.broadcast_mul(&w)?; // [n_sel, n_embd]

        out = out.index_add(&idx, &weighted, 0)?;
    }
    Ok(out)
}

// ─── Construction helpers (used by model_weights.rs and tests) ────────────────

/// Create an F32 `QTensor` from an F32 `Tensor` on CPU.
///
/// For synthetic test experts and small reference fixtures. The source tensor
/// **must** be on `Device::Cpu` (required by `QTensor::quantize_onto`).
pub fn f32_qtensor(t: &Tensor) -> Result<Arc<QTensor>> {
    use candle_core::quantized::GgmlDType;
    let qt = QTensor::quantize_onto(t, GgmlDType::F32, &Device::Cpu)?;
    Ok(Arc::new(qt))
}