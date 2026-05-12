use crate::utils::EncoderProvider;
use crate::{set_params, Buffer, ComputeCommandEncoder, Device, Kernels, MetalKernelError, Source};
use objc2_metal::{MTLResourceUsage, MTLSize};

#[derive(Debug, Clone, Copy)]
pub enum GgmlDType {
    Q4_0,
    Q4_1,
    Q5_0,
    Q5_1,
    Q8_0,
    Q8_1,
    Q2K,
    Q3K,
    Q4K,
    Q5K,
    Q6K,
    Q8K,
    IQ2XXS,
    IQ2XS,
    IQ3XXS,
    IQ1S,
    IQ4NL,
    IQ3S,
    IQ2S,
    IQ4XS,
    IQ1M,
    F16,
    F32,
    BF16,
}

#[allow(clippy::too_many_arguments)]
pub fn call_quantized_matmul_mv_t(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    dtype: GgmlDType,
    input_is_f16: bool,
    (b, m, n, k): (usize, usize, usize, usize),
    lhs: &Buffer,
    lhs_offset: usize,
    rhs: &Buffer,
    rhs_offset: usize,
    dst_offset: usize,
    dst: &Buffer,
) -> Result<(), MetalKernelError> {
    // Everything is in reverse
    let ne00 = k as i64;
    let ne01 = n as i64;
    let ne02 = b as i64;
    let ne03 = 1i64;

    let nb00 = 0i64;
    let nb01 = 0i64;
    let nb02 = 0i64;

    let ne10 = k as i64;
    let ne11 = m as i64;
    let ne12 = b as i64;
    let ne13 = 1i64;

    let nb10 = 0i64;
    let nb11 = 0i64;
    let nb12 = 0i64;

    let ne0 = n as i64;
    let ne1 = m as i64;
    let r2: u32 = (ne12 / ne02) as u32;
    let r3: u32 = (ne13 / ne03) as u32;

    let (nth0, nth1, align) = match dtype {
        GgmlDType::Q4_0
        | GgmlDType::Q4_1
        | GgmlDType::Q5_0
        | GgmlDType::Q5_1
        | GgmlDType::Q8_0
        | GgmlDType::Q8_1 => {
            let nth0 = 8;
            let nth1 = 8;
            let align = 8;
            (nth0, nth1, align)
        }
        GgmlDType::Q2K => {
            // Fixing a bug in Metal for GGML
            // https://github.com/ggerganov/llama.cpp/blob/b8109bc0139f15a5b321909f47510b89dca47ffc/ggml-metal.m#L1576
            let nth0 = 2;
            let nth1 = 32;
            let align = 4;
            (nth0, nth1, align)
        }
        GgmlDType::Q4K => {
            let nth0 = 4;
            let nth1 = 8;
            let align = 4;
            (nth0, nth1, align)
        }
        GgmlDType::Q3K | GgmlDType::Q5K => {
            let nth0 = 2;
            let nth1 = 32;
            let align = 4;
            (nth0, nth1, align)
        }
        GgmlDType::Q6K => {
            let nth0 = 2;
            let nth1 = 32;
            let align = 2;
            (nth0, nth1, align)
        }
        GgmlDType::F16 | GgmlDType::BF16 | GgmlDType::Q8K => {
            // Original implem uses rows
            let nth0 = 32;
            let nth1 = 1;
            let align = 8;
            (nth0, nth1, align)
        }
        GgmlDType::IQ2XXS
        | GgmlDType::IQ2XS
        | GgmlDType::IQ3XXS
        | GgmlDType::IQ1S
        | GgmlDType::IQ4NL
        | GgmlDType::IQ3S
        | GgmlDType::IQ2S
        | GgmlDType::IQ4XS
        | GgmlDType::IQ1M => {
            let nth0 = 8;
            let nth1 = 8;
            let align = 8;
            (nth0, nth1, align)
        }
        GgmlDType::F32 => {
            let nth0 = 32;
            let nth1 = 1;
            let align = 8;
            (nth0, nth1, align)
        }
    };
    let thread_groups_count = MTLSize {
        width: divide(ne01 as usize, align),
        height: ne11 as usize,
        depth: (ne12 * ne13) as usize,
    };
    let threads_per_threadgroup = MTLSize {
        width: nth0,
        height: nth1,
        depth: 1,
    };
    // F16 input variants пока поддержаны только для базовых Q-типов
    // (Phase 1-2 плана). IQ кванты, Q8_1, Q8K, F32/F16/BF16 weights —
    // F32 input only (Phase 7 опц.).
    let name: &str = if input_is_f16 {
        match dtype {
            GgmlDType::Q4_0 => "kernel_mul_mv_q4_0_f16",
            GgmlDType::Q4_1 => "kernel_mul_mv_q4_1_f16",
            GgmlDType::Q5_0 => "kernel_mul_mv_q5_0_f16",
            GgmlDType::Q5_1 => "kernel_mul_mv_q5_1_f16",
            GgmlDType::Q8_0 => "kernel_mul_mv_q8_0_f16",
            GgmlDType::Q2K => "kernel_mul_mv_q2_K_f16",
            GgmlDType::Q3K => "kernel_mul_mv_q3_K_f16",
            GgmlDType::Q4K => "kernel_mul_mv_q4_K_f16",
            GgmlDType::Q5K => "kernel_mul_mv_q5_K_f16",
            GgmlDType::Q6K => "kernel_mul_mv_q6_K_f16",
            _ => {
                return Err(MetalKernelError::UnsupportedDTypeForOp(
                    "non-Q4_0..Q6_K dtype + F16 input",
                    "qmatmul_mv",
                ))
            }
        }
    } else {
        match dtype {
            GgmlDType::Q4_0 => "kernel_mul_mv_q4_0_f32",
            GgmlDType::Q4_1 => "kernel_mul_mv_q4_1_f32",
            GgmlDType::Q5_0 => "kernel_mul_mv_q5_0_f32",
            GgmlDType::Q5_1 => "kernel_mul_mv_q5_1_f32",
            GgmlDType::Q8_0 => "kernel_mul_mv_q8_0_f32",
            GgmlDType::Q8_1 => "kernel_mul_mv_q8_1_f32",
            GgmlDType::Q2K => "kernel_mul_mv_q2_K_f32",
            GgmlDType::Q3K => "kernel_mul_mv_q3_K_f32",
            GgmlDType::Q4K => "kernel_mul_mv_q4_K_f32",
            GgmlDType::Q5K => "kernel_mul_mv_q5_K_f32",
            GgmlDType::Q6K => "kernel_mul_mv_q6_K_f32",
            GgmlDType::Q8K => "kernel_mul_mv_q8_K_f32",
            GgmlDType::IQ2XXS => "kernel_mul_mv_iq2_xxs_f32",
            GgmlDType::IQ2XS => "kernel_mul_mv_iq2_xs_f32",
            GgmlDType::IQ3XXS => "kernel_mul_mv_iq3_xxs_f32",
            GgmlDType::IQ1S => "kernel_mul_mv_iq1_s_f32",
            GgmlDType::IQ4NL => "kernel_mul_mv_iq4_nl_f32",
            GgmlDType::IQ3S => "kernel_mul_mv_iq3_s_f32",
            GgmlDType::IQ2S => "kernel_mul_mv_iq2_s_f32",
            GgmlDType::IQ4XS => "kernel_mul_mv_iq4_xs_f32",
            GgmlDType::IQ1M => "kernel_mul_mv_iq1_m_f32",
            GgmlDType::F16 => "kernel_mul_mv_f16_f32",
            GgmlDType::BF16 => "kernel_mul_mv_bf16_f32",
            GgmlDType::F32 => "kernel_mul_mv_f32_f32",
        }
    };

    let pipeline = kernels.load_pipeline(device, Source::Quantized, name)?;
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);

    set_params!(
        encoder,
        (
            (rhs, rhs_offset),
            (lhs, lhs_offset),
            (dst, dst_offset),
            ne00,
            ne01,
            ne02,
            nb00,
            nb01,
            nb02,
            ne10,
            ne11,
            ne12,
            nb10,
            nb11,
            nb12,
            ne0,
            ne1,
            r2,
            r3
        )
    );
    encoder.use_resource(lhs, MTLResourceUsage::Read);
    encoder.use_resource(rhs, MTLResourceUsage::Read);
    encoder.use_resource(dst, MTLResourceUsage::Write);

    if matches!(
        dtype,
        GgmlDType::IQ2XXS
            | GgmlDType::IQ2XS
            | GgmlDType::IQ3XXS
            | GgmlDType::IQ1S
            | GgmlDType::IQ4NL
            | GgmlDType::IQ3S
            | GgmlDType::IQ2S
            | GgmlDType::IQ4XS
            | GgmlDType::IQ1M
    ) {
    encoder.set_threadgroup_memory_length(0, 16384); // 12KB для MPP, 8KB для legacy
    }

    encoder.dispatch_thread_groups(thread_groups_count, threads_per_threadgroup);
    Ok(())
}

/// - src0 is usually weight
/// - src1 is usually xs
#[allow(clippy::too_many_arguments)]
pub fn call_quantized_matmul_mm_t(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    dtype: GgmlDType,
    input_is_f16: bool,
    src0_shape: &[usize],
    src0_stride: &[usize],
    src0: &Buffer,
    src0_offset: usize,
    src1_shape: &[usize],
    src1_stride: &[usize],
    src1: &Buffer,
    src1_offset: usize,
    dst_shape: &[usize],
    dst_offset: usize,
    dst: &Buffer,
) -> Result<(), MetalKernelError> {
    // Everything is in reverse
    let ne00 = src0_shape[src0_shape.len() - 1] as i64;
    let ne01 = src0_shape[src0_shape.len() - 2] as i64;
    let ne02 = src0_shape[src0_shape.len() - 3] as i64;
    let ne03 = src0_shape[src0_shape.len() - 4] as i64;

    let nb01 = src0_stride[src0_stride.len() - 2] as i64;
    let nb02 = src0_stride[src0_stride.len() - 3] as i64;
    let nb03 = src0_stride[src0_stride.len() - 4] as i64;

    let ne11 = src1_shape[src1_shape.len() - 2] as i64;
    let ne12 = src1_shape[src1_shape.len() - 3] as i64;
    let ne13 = src1_shape[src1_shape.len() - 4] as i64;

    let nb10 = src1_stride[src1_stride.len() - 1] as i64;
    let nb11 = src1_stride[src1_stride.len() - 2] as i64;
    let nb12 = src1_stride[src1_stride.len() - 3] as i64;
    let nb13 = src1_stride[src1_stride.len() - 4] as i64;

    let ne0 = dst_shape[dst_shape.len() - 1] as i64;
    let ne1 = dst_shape[dst_shape.len() - 2] as i64;
    let r2 = (ne12 / ne02) as u32;
    let r3 = (ne13 / ne03) as u32;

    let threads_per_threadgroup = MTLSize {
        width: 128,
        height: 1,
        depth: 1,
    };
    // Проверка alignment для fast-path kernel (без bounds checking).
    // Fast-path variants доступны для Q4K и Q6K когда M%64==0 и N%32==0.
    let use_fast_path = matches!(dtype, GgmlDType::Q4K | GgmlDType::Q6K)
        && (ne01 % 64 == 0)
        && (ne11 % 32 == 0);
    let name: &str = if use_fast_path {
        if input_is_f16 {
            match dtype {
                GgmlDType::Q4K => "kernel_mul_mm_q4_K_f16_fast",
                GgmlDType::Q6K => "kernel_mul_mm_q6_K_f16_fast",
                _ => unreachable!(),
            }
        } else {
            match dtype {
                GgmlDType::Q4K => "kernel_mul_mm_q4_K_f32_fast",
                GgmlDType::Q6K => "kernel_mul_mm_q6_K_f32_fast",
                _ => unreachable!(),
            }
        }
    } else if input_is_f16 {
        match dtype {
            GgmlDType::Q4_0 => "kernel_mul_mm_q4_0_f16",
            GgmlDType::Q4_1 => "kernel_mul_mm_q4_1_f16",
            GgmlDType::Q5_0 => "kernel_mul_mm_q5_0_f16",
            GgmlDType::Q5_1 => "kernel_mul_mm_q5_1_f16",
            GgmlDType::Q8_0 => "kernel_mul_mm_q8_0_f16",
            GgmlDType::Q2K => "kernel_mul_mm_q2_K_f16",
            GgmlDType::Q3K => "kernel_mul_mm_q3_K_f16",
            GgmlDType::Q4K => "kernel_mul_mm_q4_K_f16",
            GgmlDType::Q5K => "kernel_mul_mm_q5_K_f16",
            GgmlDType::Q6K => "kernel_mul_mm_q6_K_f16",
            _ => {
                return Err(MetalKernelError::UnsupportedDTypeForOp(
                    "non-Q4_0..Q6_K dtype + F16 input",
                    "qmatmul_mm",
                ))
            }
        }
    } else {
        match dtype {
            GgmlDType::Q4_0 => "kernel_mul_mm_q4_0_f32",
            GgmlDType::Q4_1 => "kernel_mul_mm_q4_1_f32",
            GgmlDType::Q5_0 => "kernel_mul_mm_q5_0_f32",
            GgmlDType::Q5_1 => "kernel_mul_mm_q5_1_f32",
            GgmlDType::Q8_0 => "kernel_mul_mm_q8_0_f32",
            GgmlDType::Q2K => "kernel_mul_mm_q2_K_f32",
            GgmlDType::Q3K => "kernel_mul_mm_q3_K_f32",
            GgmlDType::Q4K => "kernel_mul_mm_q4_K_f32",
            GgmlDType::Q5K => "kernel_mul_mm_q5_K_f32",
            GgmlDType::Q6K => "kernel_mul_mm_q6_K_f32",
            GgmlDType::IQ2XXS => "kernel_mul_mm_iq2_xxs_f32",
            GgmlDType::IQ2XS => "kernel_mul_mm_iq2_xs_f32",
            GgmlDType::IQ3XXS => "kernel_mul_mm_iq3_xxs_f32",
            GgmlDType::IQ1S => "kernel_mul_mm_iq1_s_f32",
            GgmlDType::IQ4NL => "kernel_mul_mm_iq4_nl_f32",
            GgmlDType::IQ3S => "kernel_mul_mm_iq3_s_f32",
            GgmlDType::IQ2S => "kernel_mul_mm_iq2_s_f32",
            GgmlDType::IQ4XS => "kernel_mul_mm_iq4_xs_f32",
            GgmlDType::IQ1M => "kernel_mul_mm_iq1_m_f32",
            GgmlDType::F16 => "kernel_mul_mm_f16_f32",
            GgmlDType::BF16 => "kernel_mul_mm_bf16_f32",
            GgmlDType::F32 => "kernel_mul_mm_f32_f32",
            GgmlDType::Q8_1 => Err(MetalKernelError::UnsupportedDTypeForOp("Q8_1", "qmatmul"))?,
            GgmlDType::Q8K => Err(MetalKernelError::UnsupportedDTypeForOp("Q8K", "qmatmul"))?,
        }
    };

    // MPP kernel — только на M5+/A19+ (на M4 не даёт ускорения, см. llama.cpp device.m)
    // llama.cpp benchmark: M2 Ultra +5% slower, M4/M4 Max no significant difference.
    // Причина: hardware tensor-ускорение только с M5/A19.
    let try_mpp = false; // отключено до M5 — см. GGML_METAL_TENSOR_ENABLE
    // input_is_f16 && matches!(dtype, GgmlDType::Q4K | GgmlDType::Q6K);
    let (pipeline, is_mpp) = if try_mpp {
        let mpp_name = match dtype {
            GgmlDType::Q4K => "kernel_mul_mm_mpp_q4_K_f16",
            GgmlDType::Q6K => "kernel_mul_mm_mpp_q6_K_f16",
            _ => unreachable!(),
        };
        if let Ok(p) = kernels.load_pipeline(device, Source::Quantized, mpp_name) {
            (p, true)
        } else {
            (kernels.load_pipeline(device, Source::Quantized, name)?, false)
        }
    } else {
        (kernels.load_pipeline(device, Source::Quantized, name)?, false)
    };

    let thread_groups_count = if is_mpp {
        MTLSize { width: divide(ne11 as usize, 64), height: divide(ne01 as usize, 128), depth: (ne12 * ne13) as usize }
    } else {
        MTLSize { width: divide(ne11 as usize, 32), height: divide(ne01 as usize, 64), depth: (ne12 * ne13) as usize }
    };
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);

    set_params!(
        encoder,
        (
            (src0, src0_offset),
            (src1, src1_offset),
            (dst, dst_offset),
            ne00,
            ne02,
            nb01,
            nb02,
            nb03,
            ne12,
            nb10,
            nb11,
            nb12,
            nb13,
            ne0,
            ne1,
            r2,
            r3
        )
    );
    encoder.use_resource(src0, MTLResourceUsage::Read);
    encoder.use_resource(src1, MTLResourceUsage::Read);
    encoder.use_resource(dst, MTLResourceUsage::Write);

    encoder.set_threadgroup_memory_length(0, 8192);

    encoder.dispatch_thread_groups(thread_groups_count, threads_per_threadgroup);
    Ok(())
}

fn divide(m: usize, b: usize) -> usize {
    m.div_ceil(b)
}

/// T-275 Phase 3: dispatch для optimized Q4_K_M + F32 input matmul kernel.
///
/// Использует `kernel_mul_mm_q4_K_f32_opt` с дополнительным `scales_repacked`
/// buffer (16 f16 per Q4_K block, layout V1 от `Q4KOptMetadata`).
///
/// Предусловия (вызывающий код должен гарантировать):
/// - `dtype == GgmlDType::Q4K` — kernel specialized только для Q4_K_M
/// - Input dtype = F32 — kernel specialized для F32 activations (НЕ F16)
/// - `ne01 % 64 == 0 && ne11 % 32 == 0` — FAST_PATH alignment требование
/// - `scales_repacked` содержит `Q4KOptMetadata.data` upload'нутый в Metal buffer
///   с источником из того же `src0` Q4_K_M tensor (identity guard на стороне caller'а)
///
/// При нарушении предусловий — caller должен fall back на `call_quantized_matmul_mm_t`.
#[allow(clippy::too_many_arguments)]
pub fn call_quantized_matmul_mm_q4k_opt(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    src0_shape: &[usize],
    src0_stride: &[usize],
    src0: &Buffer,
    src0_offset: usize,
    scales_repacked: &Buffer,
    src1_shape: &[usize],
    src1_stride: &[usize],
    src1: &Buffer,
    src1_offset: usize,
    dst_shape: &[usize],
    dst_offset: usize,
    dst: &Buffer,
) -> Result<(), MetalKernelError> {
    // Identical layout extraction что и в `call_quantized_matmul_mm_t`.
    let ne00 = src0_shape[src0_shape.len() - 1] as i64;
    let ne01 = src0_shape[src0_shape.len() - 2] as i64;
    let ne02 = src0_shape[src0_shape.len() - 3] as i64;
    let ne03 = src0_shape[src0_shape.len() - 4] as i64;

    let nb01 = src0_stride[src0_stride.len() - 2] as i64;
    let nb02 = src0_stride[src0_stride.len() - 3] as i64;
    let nb03 = src0_stride[src0_stride.len() - 4] as i64;

    let ne11 = src1_shape[src1_shape.len() - 2] as i64;
    let ne12 = src1_shape[src1_shape.len() - 3] as i64;
    let ne13 = src1_shape[src1_shape.len() - 4] as i64;

    let nb10 = src1_stride[src1_stride.len() - 1] as i64;
    let nb11 = src1_stride[src1_stride.len() - 2] as i64;
    let nb12 = src1_stride[src1_stride.len() - 3] as i64;
    let nb13 = src1_stride[src1_stride.len() - 4] as i64;

    let ne0 = dst_shape[dst_shape.len() - 1] as i64;
    let ne1 = dst_shape[dst_shape.len() - 2] as i64;
    let r2 = (ne12 / ne02) as u32;
    let r3 = (ne13 / ne03) as u32;

    // Sanity check FAST_PATH alignment. Caller должен это проверить заранее, но
    // дублируем как defensive check на случай wrong dispatch.
    if ne01 % 64 != 0 || ne11 % 32 != 0 {
        return Err(MetalKernelError::UnsupportedDTypeForOp(
            "kernel_mul_mm_q4_K_f32_opt requires ne01%64==0 && ne11%32==0",
            "qmatmul_mm_opt",
        ));
    }

    let threads_per_threadgroup = MTLSize {
        width: 128,
        height: 1,
        depth: 1,
    };
    let thread_groups_count = MTLSize {
        width: divide(ne11 as usize, 32),
        height: divide(ne01 as usize, 64),
        depth: (ne12 * ne13) as usize,
    };

    let pipeline =
        kernels.load_pipeline(device, Source::Quantized, "kernel_mul_mm_q4_K_f32_opt")?;

    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);

    // Параметры в порядке, заданном в kernel_mul_mm_q4_K_f32_opt signature
    // (см. quantized.metal). scales_repacked идёт ВТОРЫМ buffer (buffer(1)),
    // остальные сдвинуты на +1 относительно vanilla kernel_mul_mm.
    set_params!(
        encoder,
        (
            (src0, src0_offset),
            (scales_repacked, 0_usize),
            (src1, src1_offset),
            (dst, dst_offset),
            ne00,
            ne02,
            nb01,
            nb02,
            nb03,
            ne12,
            nb10,
            nb11,
            nb12,
            nb13,
            ne0,
            ne1,
            r2,
            r3
        )
    );
    encoder.use_resource(src0, MTLResourceUsage::Read);
    encoder.use_resource(scales_repacked, MTLResourceUsage::Read);
    encoder.use_resource(src1, MTLResourceUsage::Read);
    encoder.use_resource(dst, MTLResourceUsage::Write);

    encoder.set_threadgroup_memory_length(0, 8192);

    encoder.dispatch_thread_groups(thread_groups_count, threads_per_threadgroup);
    Ok(())
}

/// T-278 Phase 0: bridge для kernel_mul_mm_q4_K_f32_v3.
///
/// SKELETON STATE: функционально идентичен `call_quantized_matmul_mm_q4k_opt`,
/// единственное отличие — pipeline state name. Будет переписан в Фазе 1 когда
/// `kernel_mul_mm_q4_K_f32_v3` получит threadgroup tile cache.
///
/// Predicate / contract — точно те же что у `call_quantized_matmul_mm_q4k_opt`:
/// - Q4_K_M weights с pre-packed `scales_repacked` metadata
/// - F32 activations
/// - FAST_PATH alignment: `ne01 % 64 == 0 && ne11 % 32 == 0`
#[allow(clippy::too_many_arguments)]
pub fn call_quantized_matmul_mm_q4k_v3(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    src0_shape: &[usize],
    src0_stride: &[usize],
    src0: &Buffer,
    src0_offset: usize,
    scales_repacked: &Buffer,
    src1_shape: &[usize],
    src1_stride: &[usize],
    src1: &Buffer,
    src1_offset: usize,
    dst_shape: &[usize],
    dst_offset: usize,
    dst: &Buffer,
) -> Result<(), MetalKernelError> {
    let ne00 = src0_shape[src0_shape.len() - 1] as i64;
    let ne01 = src0_shape[src0_shape.len() - 2] as i64;
    let ne02 = src0_shape[src0_shape.len() - 3] as i64;
    let ne03 = src0_shape[src0_shape.len() - 4] as i64;

    let nb01 = src0_stride[src0_stride.len() - 2] as i64;
    let nb02 = src0_stride[src0_stride.len() - 3] as i64;
    let nb03 = src0_stride[src0_stride.len() - 4] as i64;

    let ne11 = src1_shape[src1_shape.len() - 2] as i64;
    let ne12 = src1_shape[src1_shape.len() - 3] as i64;
    let ne13 = src1_shape[src1_shape.len() - 4] as i64;

    let nb10 = src1_stride[src1_stride.len() - 1] as i64;
    let nb11 = src1_stride[src1_stride.len() - 2] as i64;
    let nb12 = src1_stride[src1_stride.len() - 3] as i64;
    let nb13 = src1_stride[src1_stride.len() - 4] as i64;

    let ne0 = dst_shape[dst_shape.len() - 1] as i64;
    let ne1 = dst_shape[dst_shape.len() - 2] as i64;
    let r2 = (ne12 / ne02) as u32;
    let r3 = (ne13 / ne03) as u32;

    if ne01 % 64 != 0 || ne11 % 32 != 0 {
        return Err(MetalKernelError::UnsupportedDTypeForOp(
            "kernel_mul_mm_q4_K_f32_v3 requires ne01%64==0 && ne11%32==0",
            "qmatmul_mm_v3",
        ));
    }

    let threads_per_threadgroup = MTLSize {
        width: 128,
        height: 1,
        depth: 1,
    };
    let thread_groups_count = MTLSize {
        width: divide(ne11 as usize, 32),
        height: divide(ne01 as usize, 64),
        depth: (ne12 * ne13) as usize,
    };

    let pipeline =
        kernels.load_pipeline(device, Source::Quantized, "kernel_mul_mm_q4_K_f32_v3")?;

    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);

    set_params!(
        encoder,
        (
            (src0, src0_offset),
            (scales_repacked, 0_usize),
            (src1, src1_offset),
            (dst, dst_offset),
            ne00,
            ne02,
            nb01,
            nb02,
            nb03,
            ne12,
            nb10,
            nb11,
            nb12,
            nb13,
            ne0,
            ne1,
            r2,
            r3
        )
    );
    encoder.use_resource(src0, MTLResourceUsage::Read);
    encoder.use_resource(scales_repacked, MTLResourceUsage::Read);
    encoder.use_resource(src1, MTLResourceUsage::Read);
    encoder.use_resource(dst, MTLResourceUsage::Write);

    encoder.set_threadgroup_memory_length(0, 8192);

    encoder.dispatch_thread_groups(thread_groups_count, threads_per_threadgroup);
    Ok(())
}

/// Полная дequantization квантованного буфера в F16 (half) на GPU.
///
/// Используется при `CANDLE_DEQUANTIZE_ALL_F16=1` для материализации весов в F16
/// без промежуточной F32-копии. Каждый поток обрабатывает один блок из 16 элементов
/// (один half4x4). Размер `dst` буфера должен быть `elem_count * 2` байт.
#[allow(clippy::too_many_arguments)]
pub fn call_dequantize_q_to_half(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    dtype: GgmlDType,
    src: &Buffer,
    src_offset: usize,
    dst: &Buffer,
    elem_count: usize,
) -> Result<(), MetalKernelError> {
    let name = match dtype {
        GgmlDType::Q4_0 => "kernel_dequantize_q4_0_f16",
        GgmlDType::Q4_1 => "kernel_dequantize_q4_1_f16",
        GgmlDType::Q5_0 => "kernel_dequantize_q5_0_f16",
        GgmlDType::Q5_1 => "kernel_dequantize_q5_1_f16",
        GgmlDType::Q8_0 => "kernel_dequantize_q8_0_f16",
        GgmlDType::Q2K => "kernel_dequantize_q2_K_f16",
        GgmlDType::Q3K => "kernel_dequantize_q3_K_f16",
        GgmlDType::Q4K => "kernel_dequantize_q4_K_f16",
        GgmlDType::Q5K => "kernel_dequantize_q5_K_f16",
        GgmlDType::Q6K => "kernel_dequantize_q6_K_f16",
        GgmlDType::IQ2XXS => "kernel_dequantize_iq2_xxs_f16",
        GgmlDType::IQ2XS => "kernel_dequantize_iq2_xs_f16",
        GgmlDType::IQ3XXS => "kernel_dequantize_iq3_xxs_f16",
        GgmlDType::IQ1S => "kernel_dequantize_iq1_s_f16",
        GgmlDType::IQ4NL => "kernel_dequantize_iq4_nl_f16",
        GgmlDType::IQ3S => "kernel_dequantize_iq3_s_f16",
        GgmlDType::IQ2S => "kernel_dequantize_iq2_s_f16",
        GgmlDType::IQ4XS => "kernel_dequantize_iq4_xs_f16",
        GgmlDType::IQ1M => "kernel_dequantize_iq1_m_f16",
        GgmlDType::F16 | GgmlDType::F32 | GgmlDType::BF16 => {
            return Err(MetalKernelError::UnsupportedDTypeForOp(
                "F16/F32/BF16",
                "dequantize_q_to_half",
            ))
        }
        GgmlDType::Q8_1 => {
            return Err(MetalKernelError::UnsupportedDTypeForOp(
                "Q8_1",
                "dequantize_q_to_half",
            ))
        }
        GgmlDType::Q8K => {
            return Err(MetalKernelError::UnsupportedDTypeForOp(
                "Q8K",
                "dequantize_q_to_half",
            ))
        }
    };

    let pipeline = kernels.load_pipeline(device, Source::Quantized, name)?;
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);

    // Каждый поток обрабатывает 16 элементов (одну half4x4 структуру).
    let total_threads = elem_count.div_ceil(16);
    let threads_per_tg = 64usize;
    let tg_count = total_threads.div_ceil(threads_per_tg);

    let ne = elem_count as u64;
    set_params!(encoder, ((src, src_offset), (dst, 0usize), ne));

    encoder.use_resource(src, MTLResourceUsage::Read);
    encoder.use_resource(dst, MTLResourceUsage::Write);

    let thread_groups = MTLSize {
        width: tg_count,
        height: 1,
        depth: 1,
    };
    let threads_per_threadgroup = MTLSize {
        width: threads_per_tg,
        height: 1,
        depth: 1,
    };
    encoder.dispatch_thread_groups(thread_groups, threads_per_threadgroup);
    Ok(())
}
