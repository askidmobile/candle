// Mixture-of-Experts (MoE) GEMM operations on CUDA via dynamic PTX runtime.
//
// Uses WMMA grouped kernels for prefill and vectorized MMVQ for decode.
// All kernels are dynamically loaded through cudarc without any link-time dependency on libmoe.a or cudart.

#[allow(unused_imports)]
use candle::quantized::{self, QTensor};
use candle::{Result, Tensor};

#[cfg(feature = "cuda")]
mod cuda {
    use super::*;
    use candle::cuda_backend::cudarc::driver::{CudaSlice, DevicePtr, DeviceRepr, LaunchConfig};
    use candle::cuda_backend::CudaDType;
    use candle::quantized::GgmlDType;
    use candle::{CudaDevice, CudaStorage, DType, Storage};
    use half::{bf16, f16};

    const CEILDIV: fn(usize, usize) -> usize = |x, y| (x + y - 1) / y;

    fn calculate_expert_offsets(
        dev: &CudaDevice,
        expert_ids: &CudaSlice<i32>,
        size_m: usize,
        num_experts: usize,
    ) -> Result<CudaSlice<i32>> {
        let expert_counts = dev.alloc_zeros::<i32>(num_experts)?;
        let expert_offsets = dev.alloc_zeros::<i32>(num_experts + 1)?;

        // 1. Count tokens per expert
        let threads = 256;
        let blocks = CEILDIV(size_m, threads);
        let count_func = dev.get_or_load_func("count_tokens_per_expert_kernel", &candle_kernels::MOE)?;
        let count_cfg = LaunchConfig {
            grid_dim: (blocks as u32, 1, 1),
            block_dim: (threads as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            count_func.launch(count_cfg, (expert_ids, &expert_counts, size_m as i32))
        }?;

        // 2. Prefix sum to get expert offsets
        let mut scan_threads = num_experts;
        if scan_threads < 32 {
            scan_threads = 32;
        } else if scan_threads > 1024 {
            candle::bail!("MoE prefix sum supports up to 1024 experts, got {num_experts}");
        }
        let smem_size = (scan_threads * std::mem::size_of::<i32>()) as u32;
        let scan_func = dev.get_or_load_func("expert_prefix_sum_kernel", &candle_kernels::MOE)?;
        let scan_cfg = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (scan_threads as u32, 1, 1),
            shared_mem_bytes: smem_size,
        };
        unsafe {
            scan_func.launch(scan_cfg, (&expert_counts, &expert_offsets, num_experts as i32))
        }?;

        Ok(expert_offsets)
    }

    pub fn moe_gemm_cuda(
        input: &Tensor,
        weights: &Tensor,
        topk_weights: &Option<Tensor>,
        sorted_token_ids: &Tensor,
        expert_ids: &Tensor,
        topk: usize,
        is_prefill: bool,
    ) -> Result<Tensor> {
        fn cuda_fwd<T: CudaDType + DeviceRepr>(
            input: &Tensor,
            weights: &Tensor,
            topk_weights: &Option<Tensor>,
            sorted_token_ids: &Tensor,
            expert_ids: &Tensor,
            topk: usize,
            is_prefill: bool,
            is_bf16: bool,
        ) -> Result<Tensor> {
            let (mut size_m, size_k1) = input.dims2()?;
            if topk_weights.is_none() {
                size_m *= topk;
            }
            let (num_experts, size_n, size_k) = weights.dims3()?;
            if size_k != size_k1 {
                candle::bail!(
                    "input size_k ({size_k1}) and weight size_k ({size_k}) mismatch!"
                );
            }
            let dev = input.device().as_cuda_device()?;

            let (input_storage, _) = input.storage_and_layout();
            let input_slice = match &*input_storage {
                Storage::Cuda(c) => c.as_cuda_slice::<T>()?,
                _ => candle::bail!("input must be a cuda tensor"),
            };

            let (weights_storage, _) = weights.storage_and_layout();
            let weights_slice = match &*weights_storage {
                Storage::Cuda(c) => c.as_cuda_slice::<T>()?,
                _ => candle::bail!("weight must be a cuda tensor"),
            };

            let (sorted_ids_storage, _) = sorted_token_ids.storage_and_layout();
            let sorted_ids_slice = match &*sorted_ids_storage {
                Storage::Cuda(c) => c.as_cuda_slice::<i32>()?,
                _ => candle::bail!("sorted_token_ids must be a cuda tensor"),
            };

            let (expert_ids_storage, _) = expert_ids.storage_and_layout();
            let expert_ids_slice = match &*expert_ids_storage {
                Storage::Cuda(c) => c.as_cuda_slice::<i32>()?,
                _ => candle::bail!("expert_ids must be a cuda tensor"),
            };

            let expert_offsets = calculate_expert_offsets(dev, expert_ids_slice, size_m, num_experts)?;

            let output_slice = unsafe { dev.alloc::<T>(size_m * size_n) }?;

            let kernel_name = match (is_bf16, is_prefill) {
                (false, true) => "moe_gemm_wmma_f16_prefill",
                (false, false) => "moe_gemm_wmma_f16_decode",
                (true, true) => "moe_gemm_wmma_bf16_prefill",
                (true, false) => "moe_gemm_wmma_bf16_decode",
            };

            let func = dev.get_or_load_func(kernel_name, &candle_kernels::MOE)?;

            let grid_n = CEILDIV(size_n, 32);
            let grid = (num_experts as u32, grid_n as u32, 1);
            let block = (128, 1, 1);

            let a_sh_bytes = 32 * 16 * 2;
            let b_sh_bytes = 32 * 16 * 2;
            let c_sh_bytes = 32 * 32 * std::mem::size_of::<f32>();
            let ab_bytes = a_sh_bytes + b_sh_bytes;
            let pad = (16 - (ab_bytes % 16)) % 16;
            let smem_bytes = (ab_bytes + pad + c_sh_bytes) as u32;

            let cfg = LaunchConfig {
                grid_dim: grid,
                block_dim: block,
                shared_mem_bytes: smem_bytes,
            };

            let topk_w_slice = if let Some(tw) = topk_weights {
                let (tw_storage, _) = tw.storage_and_layout();
                match &*tw_storage {
                    Storage::Cuda(c) => Some(c.as_cuda_slice::<f32>()?),
                    _ => candle::bail!("topk_weights must be a cuda tensor"),
                }
            } else {
                None
            };

            unsafe {
                if let Some(tw) = topk_w_slice {
                    func.launch(
                        cfg,
                        (
                            input_slice,
                            weights_slice,
                            sorted_ids_slice,
                            &expert_offsets,
                            tw,
                            &output_slice,
                            num_experts as i32,
                            topk as i32,
                            size_m as i32,
                            size_n as i32,
                            size_k as i32,
                        ),
                    )
                } else {
                    let null_ptr: u64 = 0;
                    func.launch(
                        cfg,
                        (
                            input_slice,
                            weights_slice,
                            sorted_ids_slice,
                            &expert_offsets,
                            null_ptr,
                            &output_slice,
                            num_experts as i32,
                            topk as i32,
                            size_m as i32,
                            size_n as i32,
                            size_k as i32,
                        ),
                    )
                }
            }?;

            let output = CudaStorage::wrap_cuda_slice(output_slice, dev.clone());
            Ok(Tensor::from_storage(
                Storage::Cuda(output),
                (size_m, size_n),
                candle::op::BackpropOp::none(),
                false,
            ))
        }

        match input.dtype() {
            DType::F16 => cuda_fwd::<f16>(
                input, weights, topk_weights, sorted_token_ids, expert_ids, topk, is_prefill, false,
            ),
            DType::BF16 => cuda_fwd::<bf16>(
                input, weights, topk_weights, sorted_token_ids, expert_ids, topk, is_prefill, true,
            ),
            dtype => candle::bail!("moe_gemm only accepts f16/bf16 inputs, got {dtype:?}"),
        }
    }

    pub fn moe_gemm_gguf_cuda(
        input: &Tensor,
        weights: &QTensor,
        topk_weights: &Option<Tensor>,
        sorted_token_ids: &Tensor,
        expert_ids: &Tensor,
        topk: usize,
        is_prefill: bool,
        dtype: DType,
    ) -> Result<Tensor> {
        let (mut size_m, size_k) = input.dims2()?;
        if topk_weights.is_none() {
            size_m *= topk;
        }
        let (num_experts, size_n, size_k1) = weights.shape().dims3()?;
        if size_k != size_k1 {
            candle::bail!(
                "input size_k ({size_k}) and weight size_k ({size_k1}) mismatch!"
            );
        }
        if size_k % 8 != 0 {
            candle::bail!("size_k must be divisible by 8, got {size_k}");
        }

        let dev = input.device().as_cuda_device()?;
        let weight_ptr = weights.device_ptr()?;

        let (sorted_ids_storage, _) = sorted_token_ids.storage_and_layout();
        let sorted_ids_slice = match &*sorted_ids_storage {
            Storage::Cuda(c) => c.as_cuda_slice::<i32>()?,
            _ => candle::bail!("sorted_token_ids must be a cuda tensor"),
        };

        let (expert_ids_storage, _) = expert_ids.storage_and_layout();
        let expert_ids_slice = match &*expert_ids_storage {
            Storage::Cuda(c) => c.as_cuda_slice::<i32>()?,
            _ => candle::bail!("expert_ids must be a cuda tensor"),
        };

        let topk_w_slice = if let Some(tw) = topk_weights {
            let (tw_storage, _) = tw.storage_and_layout();
            match &*tw_storage {
                Storage::Cuda(c) => Some(c.as_cuda_slice::<f32>()?),
                _ => candle::bail!("topk_weights must be a cuda tensor"),
            }
        } else {
            None
        };

        let output_slice = unsafe { dev.alloc::<f32>(size_m * size_n) }?;

        let quant_name = match weights.dtype() {
            GgmlDType::Q8_0 => "q8_0",
            GgmlDType::Q4K => "q4_k",
            GgmlDType::Q2K => "q2_k",
            GgmlDType::Q3K => "q3_k",
            GgmlDType::Q5K => "q5_k",
            GgmlDType::Q6K => "q6_k",
            d => candle::bail!("moe_gemm_gguf does not support weight dtype {d:?}"),
        };

        if is_prefill {
            let expert_offsets = calculate_expert_offsets(dev, expert_ids_slice, size_m, num_experts)?;
            let input_act = input.to_dtype(dtype)?;
            let (act_storage, _) = input_act.storage_and_layout();

            let type_str = if dtype == DType::F16 { "half" } else { "__nv_bfloat16" };
            let kernel_name = format!("moe_gemm_gguf_prefill_{type_str}_{quant_name}");
            let func = dev.get_or_load_func(&kernel_name, &candle_kernels::MOE)?;

            let grid = (num_experts as u32, CEILDIV(size_n, 32) as u32, 1);
            let wrap_size = if matches!(weights.dtype(), GgmlDType::Q8_0 | GgmlDType::Q4K) { 32 } else { 64 };
            let block = (wrap_size as u32, 4, 1);

            let block_size_bytes = weights.dtype().type_size();
            let qk = weights.dtype().block_size();
            let a_sh_bytes = 32 * qk * (if dtype == DType::F16 { 2 } else { 2 });
            let b_sh_bytes = 32 * qk * 2;
            let b_quant_sh_bytes = 32 * block_size_bytes;
            let c_sh_bytes = 32 * 32 * std::mem::size_of::<f32>();
            let smem_bytes = (a_sh_bytes + b_sh_bytes + b_quant_sh_bytes + 16 + c_sh_bytes) as u32;

            let cfg = LaunchConfig {
                grid_dim: grid,
                block_dim: block,
                shared_mem_bytes: smem_bytes,
            };

            let weight_dev_ptr = weight_ptr as u64;

            unsafe {
                match &*act_storage {
                    Storage::Cuda(c) => {
                        let act_ptr = c.as_cuda_slice::<u8>()?.device_ptr(&c.stream()).0;
                        if let Some(tw) = topk_w_slice {
                            func.launch(
                                cfg,
                                (
                                    act_ptr,
                                    weight_dev_ptr,
                                    sorted_ids_slice,
                                    &expert_offsets,
                                    tw,
                                    &output_slice,
                                    num_experts as i32,
                                    topk as i32,
                                    size_m as i32,
                                    size_n as i32,
                                    size_k as i32,
                                ),
                            )
                        } else {
                            let null_ptr: u64 = 0;
                            func.launch(
                                cfg,
                                (
                                    act_ptr,
                                    weight_dev_ptr,
                                    sorted_ids_slice,
                                    &expert_offsets,
                                    null_ptr,
                                    &output_slice,
                                    num_experts as i32,
                                    topk as i32,
                                    size_m as i32,
                                    size_n as i32,
                                    size_k as i32,
                                ),
                            )
                        }
                    }
                    _ => candle::bail!("input must be a cuda tensor"),
                }
            }?;
        } else {
            // Decode path: Quantize input to Q8_1
            let (input_storage, _) = input.storage_and_layout();
            let input_slice = match &*input_storage {
                Storage::Cuda(c) => c.as_cuda_slice::<f32>()?,
                _ => candle::bail!("input must be a cuda tensor"),
            };

            let matrix_row_padding = 512;
            let k_padded = (size_k + matrix_row_padding - 1) / matrix_row_padding * matrix_row_padding;
            let m_quant = if topk_weights.is_some() { size_m } else { size_m / topk };
            let q8_1_size = m_quant * (k_padded / 32 * std::mem::size_of::<candle::quantized::k_quants::BlockQ8_1>());
            let mut y_q8_1 = dev.alloc_zeros::<u8>(q8_1_size)?;

            let quant_func = dev.get_or_load_func("quantize_q8_1", &candle_kernels::QUANTIZED)?;
            let num_blocks = (k_padded + 255) / 256;
            let quant_cfg = LaunchConfig {
                grid_dim: (num_blocks as u32, m_quant as u32, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            };
            unsafe {
                quant_func.launch(quant_cfg, (input_slice, &mut y_q8_1, size_k as i32, k_padded as i32))
            }?;

            let kernel_name = format!("moe_gemm_gguf_{quant_name}");
            let func = dev.get_or_load_func(&kernel_name, &candle_kernels::MOE)?;

            let n_warps = 4;
            let grid_dim = (CEILDIV(size_n, n_warps) as u32, size_m as u32, 1);
            let block_dim = (32, n_warps as u32, 1);
            let block_size_bytes = weights.dtype().type_size();
            let qk = weights.dtype().block_size();
            let shared_bytes = (size_k / qk * block_size_bytes * n_warps + 1024) as u32;

            let cfg = LaunchConfig {
                grid_dim,
                block_dim,
                shared_mem_bytes: shared_bytes,
            };

            let weight_dev_ptr = weight_ptr as u64;

            unsafe {
                if let Some(tw) = topk_w_slice {
                    func.launch(
                        cfg,
                        (
                            weight_dev_ptr,
                            &y_q8_1,
                            sorted_ids_slice,
                            expert_ids_slice,
                            tw,
                            &output_slice,
                            num_experts as i32,
                            topk as i32,
                            size_m as i32,
                            size_n as i32,
                            size_k as i32,
                            k_padded as i32,
                        ),
                    )
                } else {
                    let null_ptr: u64 = 0;
                    func.launch(
                        cfg,
                        (
                            weight_dev_ptr,
                            &y_q8_1,
                            sorted_ids_slice,
                            expert_ids_slice,
                            null_ptr,
                            &output_slice,
                            num_experts as i32,
                            topk as i32,
                            size_m as i32,
                            size_n as i32,
                            size_k as i32,
                            k_padded as i32,
                        ),
                    )
                }
            }?;
        }

        let output = CudaStorage::wrap_cuda_slice(output_slice, dev.clone());
        Ok(Tensor::from_storage(
            Storage::Cuda(output),
            (size_m, size_n),
            candle::op::BackpropOp::none(),
            false,
        ))
    }
}

pub fn moe_gemm(
    input: &Tensor,
    weights: &Tensor,
    topk_weights: &Option<Tensor>,
    sorted_token_ids: &Tensor,
    expert_ids: &Tensor,
    topk: usize,
    is_prefill: bool,
) -> Result<Tensor> {
    #[cfg(feature = "cuda")]
    {
        if input.device().is_cuda() {
            return cuda::moe_gemm_cuda(
                input,
                weights,
                topk_weights,
                sorted_token_ids,
                expert_ids,
                topk,
                is_prefill,
            );
        }
    }
    let _ = (weights, topk_weights, sorted_token_ids, expert_ids, topk, is_prefill);
    candle::bail!("moe_gemm (dense MoE) is only supported on CUDA")
}

#[allow(clippy::too_many_arguments)]
pub fn moe_gemm_gguf(
    input: &Tensor,
    weights: &QTensor,
    topk_weights: &Option<Tensor>,
    sorted_token_ids: &Tensor,
    expert_ids: &Tensor,
    topk: usize,
    is_prefill: bool,
    dtype: candle::DType,
) -> Result<Tensor> {
    #[cfg(feature = "cuda")]
    {
        if input.device().is_cuda() {
            return cuda::moe_gemm_gguf_cuda(
                input,
                weights,
                topk_weights,
                sorted_token_ids,
                expert_ids,
                topk,
                is_prefill,
                dtype,
            );
        }
    }
    let _ = (weights, topk_weights, sorted_token_ids, expert_ids, topk, is_prefill, dtype);
    candle::bail!("moe_gemm_gguf is only supported on CUDA")
}
