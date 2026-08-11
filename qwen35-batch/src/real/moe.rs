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

        #[cfg(feature = "cuda")]
        if matches!(self.backend, MoeBackend::Ptx) && xs.device().is_cuda() {
            let combined = self.forward_ptx_cuda(&xs_2d)?;
            return combined.reshape((batch, seq_len, n_embd));
        }

        let route = self.router.route_topk(&xs_2d)?;
        let t0 = std::time::Instant::now();
        let routed = self
            .backend
            .routed_swiglu(&xs_2d, &self.routed, &route, mode)?;
        let shared = self.shared.forward(&xs_2d)?;
        if crate::scheduler::trace_on() && matches!(self.backend, MoeBackend::Ptx) {
            static N: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
            static TOTAL: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
            let n = N.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
            let tot = TOTAL.fetch_add(t0.elapsed().as_micros() as u64, std::sync::atomic::Ordering::Relaxed)
                + t0.elapsed().as_micros() as u64;
            if n % 40 == 0 {
                eprintln!("[moe] routed_swiglu avg {:.2}ms over {} calls", tot as f64 / n as f64 / 1000.0, n);
            }
        }
        let combined = routed.broadcast_add(&shared)?;

        combined.reshape((batch, seq_len, n_embd))
    }

    /// Полностью GPU-путь (CUDA): router → GPU softmax+topk kernel →
    /// dual indexed GEMM (gate+up) → SwiGLU → down → weighted sum.
    /// Без единого D2H/H2D round-trip (раньше: ~120 синков/шаг на 40 блоков).
    #[cfg(feature = "cuda")]
    fn forward_ptx_cuda(&self, xs: &Tensor) -> Result<Tensor> {
        let (n_tokens, n_embd) = xs.dims2()?;
        let k = self.router.n_experts_per_tok;
        let device = xs.device();
        let cuda_dev = device.as_cuda_device()?;

        // Router на GPU.
        let logits = self.router.linear.forward(xs)?.to_dtype(DType::F32)?.contiguous()?;
        let (ids_t, w_t) = gpu_softmax_topk(
            cuda_dev,
            &logits,
            self.router.n_experts,
            k,
            self.router.norm_topk_prob,
        )?;

        // x → [tokens, topk, n_embd] contiguous.
        let x3 = xs
            .to_dtype(DType::F32)?
            .unsqueeze(1)?
            .broadcast_as((n_tokens, k, n_embd))?
            .contiguous()?;

        // gate+up одним dual GEMM.
        let (gate, up) = self
            .routed
            .gate
            .indexed_moe_forward_dual_cuda(&self.routed.up, &x3, &ids_t)?;
        // ДИАГНОСТИКА (trace): сверка GPU-маршрута с CPU reference.
        if crate::scheduler::trace_on() {
            let route_ref = self.router.route_topk(xs)?;
            let gpu_ids: Vec<u32> = ids_t.to_vec2::<u32>()?;
            let gpu_w: Vec<f32> = w_t.to_vec2::<f32>()?;
            let cpu_ids: Vec<u32> = route_ref
                .experts
                .iter()
                .map(|r| r.iter().map(|&e| e as u32).collect::<Vec<_>>())
                .collect::<Vec<_>>()
                .concat();
            if gpu_ids != cpu_ids {
                eprintln!(
                    "[moe] ROUTE MISMATCH: gpu {:?} vs cpu {:?}",
                    &gpu_ids[..gpu_ids.len().min(16)],
                    &cpu_ids[..cpu_ids.len().min(16)]
                );
            } else {
                eprintln!("[moe] route OK; gpu_w[0..4]={:?}", &gpu_w[..gpu_w.len().min(4)]);
            }
        }
        let act = gate.silu()?.mul(&up)?.contiguous()?;
        let down = self
            .routed
            .down
            .indexed_moe_forward_cuda(&act, &ids_t)?; // [tokens, topk, n_embd]

        // Взвешивание GPU-весами + редукция по topk.
        let w = w_t.unsqueeze(candle_core::D::Minus1)?; // [tokens, topk, 1]
        let routed = down.broadcast_mul(&w)?.sum(candle_core::D::Minus2)?;
        let shared = self.shared.forward(xs)?;
        routed.broadcast_add(&shared)
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
            Self::Ptx => ptx_routed_swiglu(xs, experts, route),
        }
    }
}

/// Fused MoE путь (CUDA): indexed_moe_forward ядра — одна группа запусков на
/// проекцию вместо per-expert dequantize_rowslice (~1000 launches/шаг).
/// Работает для K-quants/Q8_0 экспертов (35B UD-Q4_K_M). IQ2_XXS — нет ядра.
#[cfg(feature = "cuda")]
fn ptx_routed_swiglu(xs: &Tensor, experts: &PackedExperts, route: &RoutePlan) -> Result<Tensor> {
    // UD-кванты динамические: часть тензоров может быть IQ (нет indexed ядра) —
    // такие блоки идём через reference (per-tensor fallback, не весь запрос).
    use candle_core::quantized::GgmlDType;
    let supported = |qt: &Arc<QTensor>| {
        matches!(
            qt.dtype(),
            GgmlDType::Q8_0
                | GgmlDType::Q2K
                | GgmlDType::Q3K
                | GgmlDType::Q4K
                | GgmlDType::Q5K
                | GgmlDType::Q6K
        )
    };
    if !(supported(&experts.gate) && supported(&experts.up) && supported(&experts.down)) {
        return reference_routed_swiglu(xs, experts, route);
    }
    let (n_tokens, n_embd) = xs.dims2()?;
    let topk = route.experts.first().map(|e| e.len()).unwrap_or(0);
    if n_tokens == 0 || topk == 0 {
        return Tensor::zeros((n_tokens, n_embd), DType::F32, xs.device());
    }
    let device = xs.device();

    // ids [n_tokens, topk] u32 на device.
    let ids_flat: Vec<u32> = route
        .experts
        .iter()
        .flat_map(|e| e.iter().map(|&x| x as u32))
        .collect();
    let ids_t = Tensor::from_vec(ids_flat, (n_tokens, topk), device)?;

    // xs → [n_tokens, topk, n_embd] contiguous (q8_1 quantize внутри ядра).
    let x3 = xs
        .to_dtype(DType::F32)?
        .unsqueeze(1)?
        .broadcast_as((n_tokens, topk, n_embd))?
        .contiguous()?;

    // gate/up: [n_tokens, topk, n_ff]; SwiGLU; down: [n_tokens, topk, n_embd].
    let gate = experts.gate.indexed_moe_forward_cuda(&x3, &ids_t)?;
    let up = experts.up.indexed_moe_forward_cuda(&x3, &ids_t)?;
    let act = gate.silu()?.mul(&up)?.contiguous()?;
    let down = experts.down.indexed_moe_forward_cuda(&act, &ids_t)?;

    // Взвешивание route weights и редукция по topk → [n_tokens, n_embd].
    let w_flat: Vec<f32> = route.weights.iter().flatten().copied().collect();
    let w = Tensor::from_vec(w_flat, (n_tokens, topk, 1), device)?;
    down.broadcast_mul(&w)?.sum(candle_core::D::Minus2)
}

#[cfg(not(feature = "cuda"))]
fn ptx_routed_swiglu(_xs: &Tensor, _experts: &PackedExperts, _route: &RoutePlan) -> Result<Tensor> {
    candle_core::bail!("PTX MoE backend requires cuda feature")
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

    // Per-expert dequantize on demand. The packed shapes are
    // gate/up: [n_experts, n_ff, n_embd], down: [n_experts, n_embd, n_ff].
    // For expert `e`, the rowslice [e, e+1) of the 2D view [n_experts, n_ff*n_embd]
    // gives a [1, n_ff*n_embd] f32 tensor; reshape to [n_ff, n_embd].
    // This avoids dequantizing all 256 experts to f32 at once (the OOM cause).
    // n_ff is recovered from the gate shape dims().
    let gate_dims = experts.gate.shape().dims();
    // Packed shape can be [n_experts, n_ff, n_embd] (3D) or a flat [n_experts*n_ff*n_embd].
    let n_ff = match gate_dims {
        [_, a, _b] => *a,                    // [n_experts, n_ff, n_embd]
        _ => candle_core::bail!(
            "reference_routed_swiglu: unexpected gate rank {:?}",
            gate_dims
        ),
    };
    // down shape is [n_experts, n_embd, n_ff] — n_ff is dims[2], not dims[1].
    let down_dims = experts.down.shape().dims();
    match down_dims {
        [_, _embd, ff] => {
            if *ff != n_ff {
                candle_core::bail!(
                    "reference_routed_swiglu: n_ff mismatch gate {} vs down {}",
                    n_ff,
                    ff
                );
            }
        }
        _ => candle_core::bail!(
            "reference_routed_swiglu: unexpected down rank {:?}",
            down_dims
        ),
    };
    let gate_k = n_ff * n_embd;      // cols of 2D view [n_experts, n_ff*n_embd]

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

        // Dequantize just expert e: rowslice [e, e+1) → [1, n_ff*n_embd] → squeeze [n_ff, n_embd].
        let gate_w = experts
            .gate
            .dequantize_rowslice(e, e + 1, gate_k, device)?
            .reshape((n_ff, n_embd))?;
        let up_w = experts
            .up
            .dequantize_rowslice(e, e + 1, gate_k, device)?
            .reshape((n_ff, n_embd))?;
        let down_w = experts
            .down
            .dequantize_rowslice(e, e + 1, gate_k, device)?
            .reshape((n_embd, n_ff))?;

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
// ─── GPU softmax+topk dispatch (CUDA, moe_router.cu) ─────────────────────────

/// Stable softmax+topk на GPU: logits [n_tokens, n_experts] F32 →
/// (ids [n_tokens, k] u32, weights [n_tokens, k] f32) — без D2H.
#[cfg(feature = "cuda")]
fn gpu_softmax_topk(
    dev: &candle_core::CudaDevice,
    logits: &Tensor,
    n_experts: usize,
    topk: usize,
    norm_topk_prob: bool,
) -> Result<(Tensor, Tensor)> {
    use cudarc::driver::{LaunchConfig, PushKernelArg};
    let (n_tokens, _) = logits.dims2()?;
    let (l_st, l_lay) = logits.storage_and_layout();
    let logits_view = match &*l_st {
        candle_core::Storage::Cuda(cs) => cs.as_cuda_slice::<f32>()?.slice(l_lay.start_offset()..),
        _ => candle_core::bail!("gpu_softmax_topk: logits not CUDA"),
    };

    let mut ids = unsafe { dev.alloc::<u32>(n_tokens * topk)? };
    let mut weights = unsafe { dev.alloc::<f32>(n_tokens * topk)? };

    let func = dev.get_or_load_func("moe_softmax_topk_kernel", &candle_kernels::MOE_ROUTER)?;
    let block = 256u32; // >= n_experts(256), power of 2
    let shared = (n_experts * 4 + n_experts) as u32; // probs f32 + masked u8
    let cfg = LaunchConfig {
        grid_dim: (n_tokens as u32, 1, 1),
        block_dim: (block, 1, 1),
        shared_mem_bytes: shared,
    };
    // kernel ждёт int32_t* для ids — transmute view u32→i32.
    let ids_i32 = unsafe { ids.transmute::<i32>(n_tokens * topk) }
        .ok_or_else(|| candle_core::Error::Msg("ids transmute".into()))?;
    let norm_flag: i32 = if norm_topk_prob { 1 } else { 0 };
    let n_experts_i = n_experts as i32;
    let topk_i = topk as i32;
    {
        let mut b = func.builder();
        b.arg(&logits_view);
        b.arg(&ids_i32);
        b.arg(&mut weights);
        b.arg(&n_experts_i);
        b.arg(&topk_i);
        b.arg(&norm_flag);
        unsafe { b.launch(cfg) }.map_err(candle_core::Error::wrap)?;
    }
    drop(l_st);

    let ids_storage = candle_core::CudaStorage::wrap_cuda_slice(ids, dev.clone());
    let ids_t = Tensor::from((candle_core::Storage::Cuda(ids_storage), (n_tokens, topk)));
    let w_storage = candle_core::CudaStorage::wrap_cuda_slice(weights, dev.clone());
    let w_t = Tensor::from((candle_core::Storage::Cuda(w_storage), (n_tokens, topk)));
    Ok((ids_t, w_t))
}
