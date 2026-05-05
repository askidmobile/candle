//! Numerical correctness тесты F16 input vs F32 input для Metal QMatMul.
//!
//! Покрывает Phase 1-3 плана F16 QMatMul Metal kernel:
//! - mm path (prefill, m>1) — kernel_mul_mm template T_input
//! - mv path (decode, m=1) — kernel_mul_mv_q* template impl
//!
//! Тест запускается только под `--features metal` на Apple Silicon.
//! diff threshold = 1e-2 — соответствует F16 precision (mantissa 11 bit).

#![cfg(feature = "metal")]

use candle_core::{
    quantized::{GgmlDType, QMatMul, QTensor},
    DType, Device, Module, Result, Tensor,
};
use std::sync::Arc;

fn run_f16_vs_f32_qmatmul(
    dtype: GgmlDType,
    n: usize,
    k: usize,
    m: usize,
) -> Result<f32> {
    let device = Device::new_metal(0)?;

    // Random weight (n, k) и random input (m, k).
    let weight = Tensor::randn(0f32, 1f32, (n, k), &device)?;
    let input = Tensor::randn(0f32, 1f32, (m, k), &device)?;

    let qweight = QTensor::quantize(&weight, dtype)?;
    let qmm = QMatMul::from_arc(Arc::new(qweight))?;

    // F32 path
    let input_f32 = input.to_dtype(DType::F32)?;
    let out_f32 = qmm.forward(&input_f32)?;

    // F16 path
    let input_f16 = input.to_dtype(DType::F16)?;
    let out_f16 = qmm.forward(&input_f16)?;

    // Output обоих kernel — F32 (kernel пишет float dst).
    // Считаем max abs diff.
    let out_f32 = out_f32.flatten_all()?.to_vec1::<f32>()?;
    let out_f16 = out_f16.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;

    assert_eq!(out_f32.len(), out_f16.len());
    let max_abs_diff = out_f32
        .iter()
        .zip(out_f16.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);

    // RMS значения F32 output — используем как scale для нормализации diff'а.
    // Это устраняет ложные срабатывания, когда отдельные элементы output
    // близки к нулю (relative diff к 1e-3 даёт неинформативный 0.7+).
    let out_rms = (out_f32.iter().map(|v| v * v).sum::<f32>() / out_f32.len() as f32).sqrt();

    let rel_to_rms = max_abs_diff / out_rms.max(1e-6);

    println!(
        "qmatmul {dtype:?} n={n} k={k} m={m}: max_abs_diff={max_abs_diff:.5} out_rms={out_rms:.5} rel_to_rms={rel_to_rms:.5}"
    );

    // F16 mantissa = 11 bit → ~5e-4 relative precision per element.
    // Накопление через k элементов: noise ~ sqrt(k) * eps на result element.
    // Допуск к RMS output — порядка 1% (k=128) до 5% (k=512).
    assert!(
        rel_to_rms < 5e-2,
        "F16 vs F32 qmatmul rel-to-RMS diff {rel_to_rms} exceeds 5% threshold \
         (dtype={dtype:?}, n={n}, k={k}, m={m}, max_abs_diff={max_abs_diff}, out_rms={out_rms})"
    );

    Ok(max_abs_diff)
}

// mv path (m=1) — decode. Q4_0 — legacy template `mul_vec_q_n_f32_impl`.
#[test]
fn qmatmul_f16_q4_0_mv() -> Result<()> {
    run_f16_vs_f32_qmatmul(GgmlDType::Q4_0, 64, 128, 1)?;
    Ok(())
}

// mv path (m=1) — Q8_0 — отдельный non-template impl.
#[test]
fn qmatmul_f16_q8_0_mv() -> Result<()> {
    run_f16_vs_f32_qmatmul(GgmlDType::Q8_0, 64, 128, 1)?;
    Ok(())
}

// mv path (m=1) — Q4_K — K-quants отдельный impl, block_size=256.
#[test]
fn qmatmul_f16_q4_k_mv() -> Result<()> {
    run_f16_vs_f32_qmatmul(GgmlDType::Q4K, 64, 512, 1)?;
    Ok(())
}

// mv path — Q6_K (полный test K-quants).
#[test]
fn qmatmul_f16_q6_k_mv() -> Result<()> {
    run_f16_vs_f32_qmatmul(GgmlDType::Q6K, 64, 512, 1)?;
    Ok(())
}

// mm path (m>1) — prefill, kernel_mul_mm template.
// Размеры выбраны так, чтобы попасть в mm-вариант (m=64 > 1, BLOCK_SIZE_M=64).
#[test]
fn qmatmul_f16_q4_0_mm() -> Result<()> {
    run_f16_vs_f32_qmatmul(GgmlDType::Q4_0, 64, 128, 64)?;
    Ok(())
}

#[test]
fn qmatmul_f16_q4_k_mm() -> Result<()> {
    run_f16_vs_f32_qmatmul(GgmlDType::Q4K, 64, 512, 64)?;
    Ok(())
}
