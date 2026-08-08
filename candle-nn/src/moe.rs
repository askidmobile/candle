// Adapted from https://github.com/guoqingbao/attention.rs/blob/main/src/moe.rs
//
// T-331 Фаза 0 (dynamic-loading) удалила MoE CUDA-ядра (`libmoe.a`) из
// `candle-kernels/build.rs`, чтобы exe не имел link-time зависимости от
// `cudart64_*.dll` (см. комментарий там). Но FFI-функции `moe_gemm`/
// `moe_gemm_gguf` ниже линковались с символами из `libmoe.a` (moe_gemm_wmma/
// moe_gemm_gguf[_prefill]) → `LNK2019 unresolved external` при cuda-сборке,
// даже у dense-моделей (Qwen3.5-4B), которые MoE не вызывают (символы тянутся
// link-time через `candle-transformers::fused_moe`).
//
// Фикс: MoE CUDA-GEMM спрятан за feature `cuda_moe` (OFF по умолчанию) →
// дефолтная cuda-сборка берёт bail-версии и линкуется. Чтобы реально включить
// MoE на CUDA: feature `cuda_moe` + восстановить `libmoe.a` в candle-kernels
// под dynamic-loading-совместимой схемой (PTX-runtime, без cudart hard-link).
#[cfg(all(feature = "cuda", feature = "cuda_moe"))]
use candle::cuda_backend::kernels::ffi;
#[allow(unused_imports)]
use candle::quantized::{self, QTensor};
use candle::{Result, Tensor};

#[cfg(all(feature = "cuda", feature = "cuda_moe"))]
pub fn moe_gemm(
    input: &Tensor,
    weights: &Tensor,
    topk_weights: &Option<Tensor>,
    sorted_token_ids: &Tensor,
    experts_ids: &Tensor,
    topk: usize,
    is_prefill: bool,
) -> Result<Tensor> {
    use candle::cuda_backend::cudarc::driver::DevicePtr;
    use candle::DType;
    use half::{bf16, f16};

    fn cuda_fwd<
        T: candle::cuda_backend::CudaDType + candle::cuda_backend::cudarc::driver::DeviceRepr,
    >(
        input: &Tensor,
        weights: &Tensor,
        topk_weights: &Option<Tensor>,
        sorted_token_ids: &Tensor,
        experts_ids: &Tensor,
        topk: usize,
        is_prefill: bool,
    ) -> Result<Tensor> {
        let (mut size_m, size_k1) = input.dims2()?;
        if topk_weights.is_none() {
            size_m *= topk;
        }
        let (num_experts, size_n, size_k) = weights.dims3()?;
        assert!(
            size_k == size_k1,
            "input {:?} and weight {:?} last dim mismatch!",
            size_k1,
            size_k
        );
        let dev = input.device().as_cuda_device()?;
        let data_type = match input.dtype() {
            DType::F16 => 0,
            DType::BF16 => 1,
            _ => {
                candle::bail!("moe_gemm_wmma only accepts f16/bf16 inputs")
            }
        };

        let (input, _) = input.storage_and_layout();
        let input = match &*input {
            candle::Storage::Cuda(c) => c.as_cuda_slice::<T>()?,
            _ => candle::bail!("input must be a cuda tensor"),
        };

        let (weights, _) = weights.storage_and_layout();
        let weights = match &*weights {
            candle::Storage::Cuda(c) => c.as_cuda_slice::<T>()?,
            _ => candle::bail!("weight must be a cuda tensor"),
        };

        let (sorted_token_ids, _) = sorted_token_ids.storage_and_layout();
        let sorted_token_ids = match &*sorted_token_ids {
            candle::Storage::Cuda(c) => c.as_cuda_slice::<u32>()?,
            _ => candle::bail!("sorted_token_ids must be a cuda tensor"),
        };

        let (experts_ids, _) = experts_ids.storage_and_layout();
        let experts_ids = match &*experts_ids {
            candle::Storage::Cuda(c) => c.as_cuda_slice::<u32>()?,
            _ => candle::bail!("experts_ids must be a cuda tensor"),
        };

        let topk_weights_ptr = if let Some(topk_weights) = &topk_weights {
            let (topk_weights, _) = topk_weights.storage_and_layout();
            let topk_weights = match &*topk_weights {
                candle::Storage::Cuda(c) => c.as_cuda_slice::<f32>()?,
                _ => candle::bail!("topk_weights must be a cuda tensor"),
            };
            let weights_ptr = topk_weights.device_ptr(topk_weights.stream()).0 as *const f32;
            weights_ptr
        } else {
            std::ptr::null()
        };

        let output = unsafe { dev.alloc::<T>(size_m * size_n) }?;
        let expert_counts = unsafe { dev.alloc::<u32>(num_experts) }?;
        let expert_offsets = unsafe { dev.alloc::<u32>(num_experts + 1) }?;

        let stream = dev.cuda_stream().cu_stream() as i64;
        use core::ffi::c_void;

        unsafe {
            ffi::moe_gemm_wmma(
                input.device_ptr(input.stream()).0 as *const c_void, // [size_m, size_k]
                weights.device_ptr(weights.stream()).0 as *const c_void, // [num_experts, size_n, size_k]
                sorted_token_ids.device_ptr(sorted_token_ids.stream()).0 as *const i32,
                experts_ids.device_ptr(experts_ids.stream()).0 as *const i32,
                topk_weights_ptr,
                output.device_ptr(output.stream()).0 as *mut c_void, // [size_m, size_n]
                expert_counts.device_ptr(expert_counts.stream()).0 as *mut i32, // pre-allocated buffer [num_experts]
                expert_offsets.device_ptr(expert_offsets.stream()).0 as *mut i32, // pre-allocated buffer [num_experts + 1]
                num_experts as i32,
                topk as i32,
                size_m as i32,
                size_n as i32,
                size_k as i32,
                data_type as i32, // 0=float16, 1=bf16 (for input/output)
                is_prefill,
                stream,
            );
        }

        use candle::op::BackpropOp;
        let output = candle::CudaStorage::wrap_cuda_slice(output, dev.clone());
        let output = Tensor::from_storage(
            candle::Storage::Cuda(output),
            (size_m, size_n),
            BackpropOp::none(),
            false,
        );

        Ok(output)
    }

    match input.dtype() {
        DType::F16 => cuda_fwd::<f16>(
            input,
            weights,
            topk_weights,
            sorted_token_ids,
            experts_ids,
            topk,
            is_prefill,
        ),
        DType::BF16 => cuda_fwd::<bf16>(
            input,
            weights,
            topk_weights,
            sorted_token_ids,
            experts_ids,
            topk,
            is_prefill,
        ),
        _ => {
            candle::bail!("moe_gemm only accepts f16/bf16 inputs")
        }
    }
}

#[cfg(not(all(feature = "cuda", feature = "cuda_moe")))]
pub fn moe_gemm(
    _: &Tensor,
    _: &Tensor,
    _: &Option<Tensor>,
    _: &Tensor,
    _: &Tensor,
    _: usize,
    _: bool,
) -> Result<Tensor> {
    candle::bail!("moe_gemm is only implemented for the cuda backend")
}

#[cfg(all(feature = "cuda", feature = "cuda_moe"))]
#[allow(clippy::too_many_arguments)]
pub fn moe_gemm_gguf(
    input: &Tensor,
    weights: &QTensor,
    topk_weights: &Option<Tensor>,
    sorted_token_ids: &Tensor,
    experts_ids: &Tensor,
    topk: usize,
    is_prefill: bool,
    dtype: candle::DType,
) -> Result<Tensor> {
    use candle::cuda_backend::cudarc::driver::DevicePtr;
    use candle::quantized::GgmlDType;
    use candle::DType;
    use half::{bf16, f16};

    #[allow(clippy::too_many_arguments)]
    fn cuda_fwd(
        input: &Tensor,
        weights: &QTensor,
        topk_weights: &Option<Tensor>,
        sorted_token_ids: &Tensor,
        experts_ids: &Tensor,
        topk: usize,
        is_prefill: bool,
        dtype: DType,
    ) -> Result<Tensor> {
        let (mut size_m, size_k) = input.dims2()?;
        if topk_weights.is_none() {
            size_m *= topk;
        }
        let (num_experts, size_n, size_k1) = weights.shape().dims3()?;
        assert!(
            size_k == size_k1,
            "input {:?} and weight {:?} last dim mismatch!",
            size_k,
            size_k1,
        );
        let dev = input.device().as_cuda_device()?;

        // Q8_0: 0, Q4K: 1, Q2K: 2, Q3k: 3,  Q5K: 4, Q6K: 5
        let gguf_dtype = match weights.dtype() {
            GgmlDType::Q8_0 => 0,
            GgmlDType::Q4K => 1,
            GgmlDType::Q2K => 2,
            GgmlDType::Q3K => 3,
            GgmlDType::Q5K => 4,
            GgmlDType::Q6K => 5,
            _ => {
                candle::bail!(
                    "moe_gemm_gguf `ISQ` only accept q2k, q3k, q4k, q5k, q6k or q8_0 weights!"
                )
            }
        };

        let weight_ptr = weights.device_ptr()?;

        let topk_weights_ptr = if let Some(topk_weights) = &topk_weights {
            let (topk_weights, _) = topk_weights.storage_and_layout();
            let topk_weights = match &*topk_weights {
                candle::Storage::Cuda(c) => c.as_cuda_slice::<f32>()?,
                _ => candle::bail!("topk_weights must be a cuda tensor"),
            };
            let w_ptr = topk_weights.device_ptr(topk_weights.stream()).0 as *const f32;
            w_ptr
        } else {
            std::ptr::null()
        };

        let (sorted_token_ids, _) = sorted_token_ids.storage_and_layout();
        let sorted_token_ids = match &*sorted_token_ids {
            candle::Storage::Cuda(c) => c.as_cuda_slice::<u32>()?,
            _ => candle::bail!("sorted_token_ids must be a cuda tensor"),
        };
        let (experts_ids, _) = experts_ids.storage_and_layout();
        let experts_ids = match &*experts_ids {
            candle::Storage::Cuda(c) => c.as_cuda_slice::<u32>()?,
            _ => candle::bail!("experts_ids must be a cuda tensor"),
        };

        let output = unsafe { dev.alloc::<f32>(size_m * size_n) }?;
        let stream = dev.cuda_stream().cu_stream() as i64;
        use candle::op::BackpropOp;
        use core::ffi::c_void;

        assert!(size_k % 8 == 0, "size_k must divisible by 8");
        unsafe {
            if is_prefill {
                let input = input.to_dtype(dtype)?;
                let (input, _) = input.storage_and_layout();
                let (input_ptr, input_dtype) = match &*input {
                    candle::Storage::Cuda(c) => {
                        if dtype == DType::F16 {
                            let c = c.as_cuda_slice::<f16>()?;
                            (c.device_ptr(c.stream()).0 as *const c_void, 0)
                        } else {
                            let c = c.as_cuda_slice::<bf16>()?;
                            (c.device_ptr(c.stream()).0 as *const c_void, 1)
                        }
                    }
                    _ => candle::bail!("input must be a cuda tensor"),
                };
                ffi::moe_gemm_gguf_prefill(
                    input_ptr,  // [size_m or size_m/topk, size_k]
                    weight_ptr, // [num_experts, size_n, size_k]
                    sorted_token_ids.device_ptr(sorted_token_ids.stream()).0 as *const i32,
                    experts_ids.device_ptr(experts_ids.stream()).0 as *const i32,
                    topk_weights_ptr,
                    output.device_ptr(output.stream()).0 as *mut c_void, // [size_m, size_n]
                    num_experts as i32,
                    topk as i32,
                    size_m as i32,
                    size_n as i32,
                    size_k as i32,
                    input_dtype,
                    gguf_dtype as i32, // Q8_0: 0, Q4K: 1, Q2K: 2, Q3k: 3,  Q5K: 4, Q6K: 5 (for weight)
                    stream,
                );
            } else {
                let (input, _) = input.storage_and_layout();
                let input = match &*input {
                    candle::Storage::Cuda(c) => c.as_cuda_slice::<f32>()?,
                    _ => candle::bail!("input must be a cuda tensor"),
                };

                ffi::moe_gemm_gguf(
                    input.device_ptr(input.stream()).0 as *const f32, // [size_m or size_m/topk, size_k]
                    weight_ptr as *const c_void, // [num_experts, size_n, size_k]
                    sorted_token_ids.device_ptr(sorted_token_ids.stream()).0 as *const i32,
                    experts_ids.device_ptr(experts_ids.stream()).0 as *const i32,
                    topk_weights_ptr,
                    output.device_ptr(output.stream()).0 as *mut c_void, // [size_m, size_n]
                    num_experts as i32,
                    topk as i32,
                    size_m as i32,
                    size_n as i32,
                    size_k as i32,
                    gguf_dtype as i32, // Q8_0: 0, Q4K: 1, Q2K: 2, Q3k: 3,  Q5K: 4, Q6K: 5 (for weight)
                    stream,
                );
            }
        }

        let output = candle::CudaStorage::wrap_cuda_slice(output, dev.clone());
        let output = Tensor::from_storage(
            candle::Storage::Cuda(output),
            (size_m, size_n),
            BackpropOp::none(),
            false,
        );

        Ok(output)
    }

    match input.dtype() {
        DType::F32 => cuda_fwd(
            input,
            weights,
            topk_weights,
            sorted_token_ids,
            experts_ids,
            topk,
            is_prefill,
            dtype,
        ),
        _ => {
            candle::bail!("moe_gemm_gguf only accepts f32 inputs")
        }
    }
}

#[cfg(not(all(feature = "cuda", feature = "cuda_moe")))]
#[allow(clippy::too_many_arguments)]
pub fn moe_gemm_gguf(
    _: &Tensor,
    _: &QTensor,
    _: &Option<Tensor>,
    _: &Tensor,
    _: &Tensor,
    _: usize,
    _: bool,
    _: candle::DType,
) -> Result<Tensor> {
    candle::bail!("moe_gemm_gguf is only implemented for the cuda backend")
}

// ─── Phase 3: PTX MoE backend (cudarc dynamic-loading, no libmoe.a) ─────────────
//
// Fused K-quant (Q2_K, Q4_K) expert GEMM via PTX kernels in moe_quantized.cu.
// IQ types bail — caller falls back to reference backend.
// CPU flattens+sorts route plan by expert (PD-010), uploads sorted arrays to
// GPU, launches moe_gate_up_kernel then moe_down_proj_kernel (or prefill).
// No cudaMallocAsync in hot path (PD-011).

#[cfg(all(feature = "cuda", feature = "moe-cuda"))]
#[allow(clippy::too_many_arguments)]
pub fn moe_ptx_gguf(
    input: &Tensor,             // [n_tokens, n_embd] F32 on CUDA
    gate: &QTensor,             // [n_experts, n_ff, n_embd] packed quantized
    up: &QTensor,               // [n_experts, n_ff, n_embd] packed quantized
    down: &QTensor,             // [n_experts, n_embd, n_ff] packed quantized
    expert_ids: &[Vec<usize>],  // [n_tokens][topk] selected expert indices
    weights: &[Vec<f32>],      // [n_tokens][topk] normalized routing weights
    _n_experts: usize,
    n_ff: usize,
    is_prefill: bool,
) -> Result<Tensor> {
    use candle::cuda_backend::cudarc::driver::{
        DeviceRepr, LaunchConfig, PushKernelArg,
    };
    use candle::cuda_backend::{kernels, WrapErr};
    use candle::op::BackpropOp;
    use candle::quantized::GgmlDType;

    let dtype = gate.dtype();
    // Only Q2_K and Q4_K have fused PTX dispatch. IQ types and other K-quants
    // bail → caller (qwen35-batch) falls back to reference backend.
    // ponytail: add IQ grid tables to moe_quantized.cu to support IQ types here.
    // quant_type values match GgmlDType::to_u32() (Q2K=10, Q4K=12) and MOE_QTYPE_* in moe_dequant.cuh.
    let (quant_type, block_size, type_size) = match dtype {
        GgmlDType::Q2K => (10i32, dtype.block_size() as i32, dtype.type_size() as i32),
        GgmlDType::Q4K => (12i32, dtype.block_size() as i32, dtype.type_size() as i32),
        _ => {
            candle::bail!(
                "moe_ptx_gguf: dtype {:?} not supported by fused PTX kernels \
                 (only Q2K/Q4K). Use reference backend for this quant.",
                dtype
            )
        }
    };

    if up.dtype() != dtype || down.dtype() != dtype {
        candle::bail!(
            "moe_ptx_gguf: gate/up/down dtype mismatch: {:?} / {:?} / {:?}",
            dtype,
            up.dtype(),
            down.dtype()
        );
    }

    let (n_tokens, n_embd) = input.dims2()?;
    if expert_ids.len() != n_tokens || weights.len() != n_tokens {
        candle::bail!(
            "moe_ptx_gguf: route plan length {} != n_tokens {}",
            expert_ids.len(),
            n_tokens
        );
    }

    let dev = input.device().as_cuda_device()?;

    // Flatten + sort by expert on CPU (PD-010).
    let topk = expert_ids.first().map_or(0, |v| v.len());
    let mut triples: Vec<(i32, i32, f32)> = Vec::with_capacity(n_tokens * topk);
    for (t, (eks, ws)) in expert_ids.iter().zip(weights.iter()).enumerate() {
        if eks.len() != ws.len() {
            candle::bail!("moe_ptx_gguf: expert/weight length mismatch at token {}", t);
        }
        for (e, w) in eks.iter().zip(ws.iter()) {
            triples.push((*e as i32, t as i32, *w));
        }
    }
    triples.sort_by_key(|(e, _, _)| *e);

    let m_total = triples.len();
    if m_total == 0 {
        let zeros = Tensor::zeros((n_tokens, n_embd), candle::DType::F32, input.device())?;
        return Ok(zeros);
    }

    let sorted_expert_ids: Vec<i32> = triples.iter().map(|(e, _, _)| *e).collect();
    let sorted_token_ids: Vec<i32> = triples.iter().map(|(_, t, _)| *t).collect();
    let sorted_weights: Vec<f32> = triples.iter().map(|(_, _, w)| *w).collect();

    // Input as CudaSlice<f32>, sliced from layout start_offset.
    let (input_storage, input_layout) = input.storage_and_layout();
    let input_slice = match &*input_storage {
        candle::Storage::Cuda(c) => c.as_cuda_slice::<f32>()?,
        _ => candle::bail!("moe_ptx_gguf: input must be a cuda tensor"),
    };
    let input_view = input_slice.slice(input_layout.start_offset()..);

    // Raw device pointers for packed weights (PD-011: pass as &u64).
    let gate_ptr = gate.device_ptr()? as u64;
    let up_ptr = up.device_ptr()? as u64;
    let down_ptr = down.device_ptr()? as u64;

    // Upload sorted route arrays.
    let d_expert_ids = dev.clone_htod(&sorted_expert_ids)?;
    let d_token_ids = dev.clone_htod(&sorted_token_ids)?;
    let d_weights = dev.clone_htod(&sorted_weights)?;

    // Workspace: intermediate [m_total, n_ff] + output [n_tokens, n_embd].
    let mut intermediate = unsafe { dev.alloc::<f32>(m_total * n_ff)? };
    let mut output = dev.alloc_zeros::<f32>(n_tokens * n_embd)?;

    let block_dim: u32 = 256;

    // ── Launch moe_gate_up_kernel ───────────────────────────────────────────
    // Grid: (ceil(n_ff/256), m_total), Block: 256.
    let grid_gate = ((n_ff as u32 + block_dim - 1) / block_dim, m_total as u32, 1);
    let cfg_gate = LaunchConfig {
        grid_dim: grid_gate,
        block_dim: (block_dim, 1, 1),
        shared_mem_bytes: 0,
    };
    let gate_up_func = dev.get_or_load_func("moe_gate_up_kernel", &kernels::MOE_QUANTIZED)?;
    let mut b1 = gate_up_func.builder();
    b1.arg(&input_view);
    b1.arg(&gate_ptr);
    b1.arg(&up_ptr);
    b1.arg(&d_expert_ids);
    b1.arg(&d_token_ids);
    b1.arg(&mut intermediate);
    candle::builder_arg!(
        b1, n_embd as i32, n_ff as i32, m_total as i32, quant_type, block_size, type_size
    );
    // SAFETY: kernel args match moe_gate_up_kernel signature.
    unsafe { b1.launch(cfg_gate) }.w()?;

    // ── Launch moe_down_proj_kernel (decode or prefill variant) ─────────────
    // Grid: (ceil(n_embd/256), m_total), Block: 256.
    let down_name = if is_prefill {
        "moe_prefill_down_proj_kernel"
    } else {
        "moe_down_proj_kernel"
    };
    let grid_down = ((n_embd as u32 + block_dim - 1) / block_dim, m_total as u32, 1);
    let cfg_down = LaunchConfig {
        grid_dim: grid_down,
        block_dim: (block_dim, 1, 1),
        shared_mem_bytes: 0,
    };
    let down_func = dev.get_or_load_func(down_name, &kernels::MOE_QUANTIZED)?;
    let mut b2 = down_func.builder();
    b2.arg(&intermediate);
    b2.arg(&down_ptr);
    b2.arg(&d_expert_ids);
    b2.arg(&d_weights);
    b2.arg(&mut output);
    b2.arg(&d_token_ids);
    candle::builder_arg!(
        b2, n_ff as i32, n_embd as i32, m_total as i32, quant_type, block_size, type_size
    );
    // SAFETY: kernel args match moe_down_proj_kernel signature.
    unsafe { b2.launch(cfg_down) }.w()?;

    let out_storage = candle::CudaStorage::wrap_cuda_slice(output, dev.clone());
    let out_tensor = Tensor::from_storage(
        candle::Storage::Cuda(out_storage),
        (n_tokens, n_embd),
        BackpropOp::none(),
        false,
    );
    Ok(out_tensor)
}

#[cfg(not(all(feature = "cuda", feature = "moe-cuda")))]
#[allow(clippy::too_many_arguments)]
pub fn moe_ptx_gguf(
    _: &Tensor,
    _: &QTensor,
    _: &QTensor,
    _: &QTensor,
    _: &[Vec<usize>],
    _: &[Vec<f32>],
    _: usize,
    _: usize,
    _: bool,
) -> Result<Tensor> {
    candle::bail!("moe_ptx_gguf requires cuda + moe-cuda features")
}
