// delta_rule_cuda.rs — Rust bindings для CUDA delta_rule kernels (per-token decode).
//
// Структурный аналог metal/delta_rule_metal.rs для NVIDIA GPU (Windows/Linux).
// Сами ядра — candle-fork/candle-kernels/src/delta_rule.cu (зарегистрированы как
// `candle_kernels::DELTA_RULE`, имена функций без mangling через extern "C").
//
// ВЫПОЛНЯЕТ ВСЁ НА GPU — ноль GPU↔CPU sync на рекуррентном шаге (как Metal-путь).
// Decode работает в F32 (precision: F16 в decode даёт cumulative drift, см. .cu).
//
// ─────────────────────────────────────────────────────────────────────────────
// СБОРКА/ТЕСТ ТОЛЬКО на yttri-win (RTX 3060) с `--features gpu-cuda`.
// На macOS/Metal этот модуль cfg-выключен и в сборку не входит.
// Перед первой сборкой добавить в Cargo.toml под gpu-cuda:
//   candle-kernels = { git = "...askidmobile/candle", optional = true }
//   cudarc        = { version = "0.19", default-features = false, optional = true }
// и в feature gpu-cuda: "dep:candle-kernels", "dep:cudarc".
// На первом компиле проверить: точные публичные пути CudaStorage::wrap_cuda_slice,
// Tensor::from_storage(Storage::Cuda(..)), borrow-время storage-guard'ов.
// ─────────────────────────────────────────────────────────────────────────────

use candle_core::{CudaDevice, CudaStorage, Device, Result, Storage, Tensor};
use cudarc::driver::{CudaSlice, LaunchConfig, PushKernelArg};

/// Параметры DeltaNet слоя. Layout ДОЛЖЕН точно совпадать со `struct DeltaParams`
/// в delta_rule.cu (#[repr(C)], 11 полей u32/f32 = 44 байта). Передаётся в ядро
/// по значению через `builder.arg(&params)`.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct DeltaParams {
    pub n_k_heads: u32,
    pub n_v_heads: u32,
    pub head_k_dim: u32,
    pub head_v_dim: u32,
    pub key_dim: u32,
    pub value_dim: u32,
    pub channels: u32,
    pub conv_kernel: u32,
    pub q_scale: f32,
    pub rms_norm_eps: f32,
    pub heads_per_kv: u32,
}

// SAFETY: POD #[repr(C)] struct из u32/f32 — валидно как kernel-аргумент по значению.
unsafe impl cudarc::driver::DeviceRepr for DeltaParams {}

/// Persistent CUDA-буферы одного DeltaNet слоя (живут всё время жизни модели,
/// мутируются in-place device-side между токенами — НИКОГДА не уходят на host).
pub struct DeltaNetCudaState {
    /// SSM state: [n_v_heads * head_v_dim * head_v_dim] = 524288 f32
    pub ssm_state: CudaSlice<f32>,
    /// Conv1d state: [(conv_k - 1) * channels] = 24576 f32
    pub conv_state: CudaSlice<f32>,
    /// Conv1d веса: [channels * conv_k]
    pub conv_weights: CudaSlice<f32>,
    /// dt_bias: [n_v_heads]
    pub dt_bias: CudaSlice<f32>,
    /// ssm_a: [n_v_heads]
    pub ssm_a: CudaSlice<f32>,
    /// RMS norm веса: [head_v_dim]
    pub norm_weight: CudaSlice<f32>,
}

/// Временные (scratch) CUDA-буферы, переиспользуемые между слоями.
pub struct DeltaNetCudaTemp {
    pub qkv_conv: CudaSlice<f32>,     // [channels]
    pub q: CudaSlice<f32>,            // [n_v_heads * head_k_dim]
    pub k: CudaSlice<f32>,            // [n_v_heads * head_k_dim]
    pub v: CudaSlice<f32>,            // [n_v_heads * head_v_dim]
    pub beta: CudaSlice<f32>,         // [n_v_heads]
    pub gate: CudaSlice<f32>,         // [n_v_heads]
    pub delta_output: CudaSlice<f32>, // [n_v_heads * head_v_dim]
}

/// Создаёт persistent буферы слоя. Веса из GGUF копируются htod один раз.
pub fn create_layer_cuda_state(
    dev: &CudaDevice,
    params: &DeltaParams,
    conv_weights_data: &[f32],
    dt_bias_data: &[f32],
    ssm_a_data: &[f32],
    norm_weight_data: &[f32],
) -> Result<DeltaNetCudaState> {
    let hd = params.head_v_dim as usize;
    let n_v = params.n_v_heads as usize;
    let channels = params.channels as usize;
    let conv_k = params.conv_kernel as usize;

    let ssm_state = dev.alloc_zeros::<f32>(n_v * hd * hd)?;
    let conv_state = dev.alloc_zeros::<f32>((conv_k - 1) * channels)?;

    let conv_weights = dev.clone_htod(conv_weights_data)?;
    let dt_bias = dev.clone_htod(dt_bias_data)?;
    let ssm_a = dev.clone_htod(ssm_a_data)?;
    let norm_weight = dev.clone_htod(norm_weight_data)?;

    Ok(DeltaNetCudaState {
        ssm_state,
        conv_state,
        conv_weights,
        dt_bias,
        ssm_a,
        norm_weight,
    })
}

/// Создаёт scratch-буферы (один комплект на модель, переиспользуется слоями).
pub fn create_temp_buffers(dev: &CudaDevice, params: &DeltaParams) -> Result<DeltaNetCudaTemp> {
    let channels = params.channels as usize;
    let n_v = params.n_v_heads as usize;
    let hkd = params.head_k_dim as usize;
    let hvd = params.head_v_dim as usize;
    Ok(DeltaNetCudaTemp {
        qkv_conv: dev.alloc_zeros::<f32>(channels)?,
        q: dev.alloc_zeros::<f32>(n_v * hkd)?,
        k: dev.alloc_zeros::<f32>(n_v * hkd)?,
        v: dev.alloc_zeros::<f32>(n_v * hvd)?,
        beta: dev.alloc_zeros::<f32>(n_v)?,
        gate: dev.alloc_zeros::<f32>(n_v)?,
        delta_output: dev.alloc_zeros::<f32>(n_v * hvd)?,
    })
}

/// Строит `CudaView<f32>` (с учётом offset) из storage-guard'а F32 CUDA-тензора.
/// Guard передаётся caller'ом и обязан жить дольше возвращённого view (borrow).
fn cuda_slice_view<'a>(
    storage: &'a Storage,
    offset: usize,
) -> Result<cudarc::driver::CudaView<'a, f32>> {
    match storage {
        Storage::Cuda(cs) => Ok(cs.as_cuda_slice::<f32>()?.slice(offset..)),
        _ => candle_core::bail!("delta_rule_cuda: тензор не на CUDA device"),
    }
}

/// Полный delta_rule forward pass на CUDA GPU (4 ядра, ноль host-sync).
///
/// Входы — 4 результата QMatMul (уже на GPU): qkv [1,1,channels], z [1,1,value_dim],
/// beta [1,1,n_v_heads], alpha [1,1,n_v_heads]. Выход — [1,1,value_dim] на GPU.
pub fn dispatch_delta_rule(
    dev: &CudaDevice,
    state: &DeltaNetCudaState,
    temp: &DeltaNetCudaTemp,
    params: &DeltaParams,
    qkv_t: &Tensor,
    z_t: &Tensor,
    beta_t: &Tensor,
    alpha_t: &Tensor,
) -> Result<Tensor> {
    let channels = params.channels as usize;
    let n_v = params.n_v_heads as u32;
    let hkd = params.head_k_dim as u32;
    let hvd = params.head_v_dim as u32;
    let value_dim = params.value_dim as usize;

    // Decode-path требует F32 contiguous (как Metal). QMatMul выдаёт F32.
    let qkv_f = ensure_f32(qkv_t)?;
    let z_f = ensure_f32(z_t)?;
    let beta_f = ensure_f32(beta_t)?;
    let alpha_f = ensure_f32(alpha_t)?;

    // Storage-guard'ы держим живыми до конца всех launch'ей (view'ы их заимствуют).
    let (qkv_st, qkv_lay) = qkv_f.storage_and_layout();
    let (z_st, z_lay) = z_f.storage_and_layout();
    let (beta_st, beta_lay) = beta_f.storage_and_layout();
    let (alpha_st, alpha_lay) = alpha_f.storage_and_layout();
    let qkv_v = cuda_slice_view(&qkv_st, qkv_lay.start_offset())?;
    let z_v = cuda_slice_view(&z_st, z_lay.start_offset())?;
    let beta_v = cuda_slice_view(&beta_st, beta_lay.start_offset())?;
    let alpha_v = cuda_slice_view(&alpha_st, alpha_lay.start_offset())?;

    let p = *params;

    // ── Kernel 1: delta_conv1d_prep ──
    // grid=(ceil(channels/256),1,1), block=(256,1,1)
    {
        let func = dev.get_or_load_func("delta_conv1d_prep", &candle_kernels::DELTA_RULE)?;
        let cfg = LaunchConfig {
            grid_dim: ((channels as u32).div_ceil(256), 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut b = func.builder();
        b.arg(&qkv_v); // qkv_raw
        b.arg(&beta_v); // beta_raw
        b.arg(&alpha_v); // alpha_raw
        b.arg(&state.conv_weights);
        b.arg(&state.dt_bias);
        b.arg(&state.ssm_a);
        b.arg(&state.conv_state); // read-write persistent
        b.arg(&temp.qkv_conv); // out
        b.arg(&temp.beta); // out
        b.arg(&temp.gate); // out
        b.arg(&p);
        unsafe { b.launch(cfg) }.map_err(candle_core::Error::wrap)?;
    }

    // ── Kernel 2: delta_l2_norm_expand ──
    // grid=(n_v,1,1), block=(head_k_dim,1,1)
    {
        let func = dev.get_or_load_func("delta_l2_norm_expand", &candle_kernels::DELTA_RULE)?;
        let cfg = LaunchConfig {
            grid_dim: (n_v, 1, 1),
            block_dim: (hkd, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut b = func.builder();
        b.arg(&temp.qkv_conv);
        b.arg(&temp.q);
        b.arg(&temp.k);
        b.arg(&temp.v);
        b.arg(&p);
        unsafe { b.launch(cfg) }.map_err(candle_core::Error::wrap)?;
    }

    // ── Kernel 3: delta_rule_kernel ──
    // grid=(n_v,1,1), block=(head_v_dim,1,1)
    {
        let func = dev.get_or_load_func("delta_rule_kernel", &candle_kernels::DELTA_RULE)?;
        let cfg = LaunchConfig {
            grid_dim: (n_v, 1, 1),
            block_dim: (hvd, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut b = func.builder();
        b.arg(&temp.q);
        b.arg(&temp.k);
        b.arg(&temp.v);
        b.arg(&temp.beta);
        b.arg(&temp.gate);
        b.arg(&state.ssm_state); // read-write persistent
        b.arg(&temp.delta_output); // out
        b.arg(&p);
        unsafe { b.launch(cfg) }.map_err(candle_core::Error::wrap)?;
    }

    // ── Kernel 4: delta_norm_gate_kernel ──
    // Свежий output-буфер (per-token 4096 f32 = 16 КБ), оборачиваем в Tensor.
    let gated = unsafe { dev.alloc::<f32>(value_dim)? };
    {
        let func = dev.get_or_load_func("delta_norm_gate_kernel", &candle_kernels::DELTA_RULE)?;
        let cfg = LaunchConfig {
            grid_dim: (n_v, 1, 1),
            block_dim: (hvd, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut b = func.builder();
        b.arg(&temp.delta_output);
        b.arg(&z_v);
        b.arg(&state.norm_weight);
        b.arg(&gated); // out
        b.arg(&p);
        unsafe { b.launch(cfg) }.map_err(candle_core::Error::wrap)?;
    }

    // Zero-copy: CudaSlice → CudaStorage → Tensor [1,1,value_dim].
    let storage = CudaStorage::wrap_cuda_slice(gated, dev.clone());
    let output = Tensor::from_storage(
        Storage::Cuda(storage),
        (1, 1, value_dim),
        candle_core::op::BackpropOp::none(),
        false,
    );
    Ok(output)
}

/// Cast в F32 contiguous (как Metal-путь).
fn ensure_f32(t: &Tensor) -> Result<Tensor> {
    if t.dtype() == candle_core::DType::F32 {
        if t.is_contiguous() {
            Ok(t.clone())
        } else {
            t.contiguous()
        }
    } else {
        t.to_dtype(candle_core::DType::F32)?.contiguous()
    }
}

/// Snapshot GPU state → CPU (T-274 prompt-cache parity). dtoh-копия двух буферов.
pub fn snapshot_cuda_state(
    dev: &CudaDevice,
    state: &DeltaNetCudaState,
) -> Result<(Vec<f32>, Vec<f32>)> {
    let ssm = dev.clone_dtoh(&state.ssm_state)?;
    let conv = dev.clone_dtoh(&state.conv_state)?;
    Ok((ssm, conv))
}

/// Восстановление GPU state из snapshot. htod-копия в существующие буферы.
pub fn restore_cuda_state(
    dev: &CudaDevice,
    state: &mut DeltaNetCudaState,
    ssm: &[f32],
    conv: &[f32],
) -> Result<()> {
    dev.memcpy_htod(ssm, &mut state.ssm_state)?;
    dev.memcpy_htod(conv, &mut state.conv_state)?;
    Ok(())
}

/// Сброс state (новая беседа): зануляем оба persistent-буфера через htod нулей.
pub fn clear_cuda_state(dev: &CudaDevice, state: &mut DeltaNetCudaState) -> Result<()> {
    let ssm_len = state.ssm_state.len();
    let conv_len = state.conv_state.len();
    dev.memcpy_htod(&vec![0.0f32; ssm_len], &mut state.ssm_state)?;
    dev.memcpy_htod(&vec![0.0f32; conv_len], &mut state.conv_state)?;
    Ok(())
}

/// Достаёт `&CudaDevice` из CUDA-тензора (для caller'а, который держит Tensor).
pub fn cuda_device_of(t: &Tensor) -> Result<CudaDevice> {
    match t.device() {
        Device::Cuda(d) => Ok(d.clone()),
        _ => candle_core::bail!("delta_rule_cuda: ожидался CUDA device"),
    }
}
