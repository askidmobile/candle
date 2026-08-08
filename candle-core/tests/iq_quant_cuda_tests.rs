//! Isolated CUDA tests for IQ quant types (IQ3XXS, IQ2S, IQ3S, IQ2XS, IQ4XS).
//!
//! These types cannot use `QTensor::quantize` (CPU `from_float` panics for
//! `RawQuantizedType`). Instead we construct QTensors from raw bytes via
//! `QStorage::from_data` on a CUDA device — the same path used by the GGUF
//! loader — and verify that `QMatMul::forward` dispatches to `cuda_fwd`
//! (not `cpu_fwd`, which bails with "CPU matmul is not implemented for IQ3XXS").
//!
//! Each block is 256 elements (QK_K). We use d=1.0 (f16) and zeroed qs, which
//! produces a deterministic dequantized value per element. The matmul is
//! compared against a CPU reference computed from the same dequantized
//! weights (via `QTensor::dequantize` on CPU construction).
//!
//! Run on yttri-win: `cargo test --features cuda --package candle-core --test iq_quant_cuda_tests`

#![cfg(feature = "cuda")]

use candle_core::quantized::{GgmlDType, QMatMul, QStorage, QTensor};
use candle_core::{DType, Device, Module, Result, Tensor};
use half::f16;
use std::sync::Arc;

const QK_K: usize = 256;

/// Build a zeroed block of raw bytes for the given IQ dtype, with d set to 1.0 (f16).
///
/// The block layout follows the CUDA struct definitions in quantized.cu:
/// - IQ3XXS: half d + 3*QK_K/8 bytes qs
/// - IQ2S:   half d + QK_K/4 qs + QK_K/32 qh + QK_K/32 scales
/// - IQ3S:   half d + QK_K/4 qs + QK_K/32 qh + QK_K/8 signs + QK_K/64 scales
/// - IQ2XS:  half d + QK_K/8 * uint16 qs + QK_K/32 scales
/// - IQ4XS:  half d + uint16 scales_h + QK_K/64 scales_l + QK_K/2 qs
fn make_zero_block_bytes(dtype: GgmlDType) -> Vec<u8> {
    let type_size = dtype.type_size();
    let block_size = dtype.block_size();
    assert_eq!(block_size, QK_K, "IQ types have block_size == QK_K == 256");
    let mut bytes = vec![0u8; type_size];
    // Set d (first 2 bytes) to f16(1.0).
    let d_f16: f16 = f16::from_f32(1.0);
    let d_bytes = d_f16.to_le_bytes();
    bytes[0] = d_bytes[0];
    bytes[1] = d_bytes[1];
    bytes
}

/// Number of blocks needed for `n_elems` elements.
fn n_blocks(n_elems: usize, dtype: GgmlDType) -> usize {
    n_elems / dtype.block_size()
}

/// Build a QTensor on the given device from zeroed blocks (d=1.0).
///
/// `n_rows` x `n_cols` where n_cols must be a multiple of QK_K.
fn make_iq_qtensor(
    n_rows: usize,
    n_cols: usize,
    dtype: GgmlDType,
    device: &Device,
) -> Result<QTensor> {
    assert!(
        n_cols % QK_K == 0,
        "n_cols must be multiple of QK_K, got {n_cols}"
    );
    let blocks_per_row = n_cols / QK_K;
    let total_blocks = n_rows * blocks_per_row;
    let type_size = dtype.type_size();
    let total_bytes = total_blocks * type_size;
    // Build raw data: each block is zeroed with d=1.0.
    let block_template = make_zero_block_bytes(dtype);
    let mut raw = Vec::with_capacity(total_bytes);
    for _ in 0..total_blocks {
        raw.extend_from_slice(&block_template);
    }
    let storage = QStorage::from_data(
        std::borrow::Cow::Borrowed(&raw),
        device,
        dtype,
    )?;
    QTensor::new(storage, (n_rows, n_cols))
}

/// Dequantize the IQ QTensor to a CPU f32 Tensor for reference computation.
///
/// On CUDA, `QTensor::dequantize` calls `QCudaStorage::dequantize` which runs
/// the CUDA dequantize kernel and returns a CudaStorage, then `.to_device(Cpu)`
/// copies it to CPU. This also validates the dequantize kernel itself.
fn dequantize_to_cpu_f32(qt: &QTensor) -> Result<Tensor> {
    let dev = qt.device();
    let t = qt.dequantize(&dev)?;
    t.to_device(&Device::Cpu)?.to_dtype(DType::F32)
}

/// Run QMatMul::forward on a CUDA device with a CUDA input tensor, and compare
/// against a CPU reference matmul using the dequantized weights.
///
/// This catches the "CPU matmul is not implemented for IQ3XXS" bug: if the
/// dispatch erroneously routes to `cpu_fwd`, the test fails with that error
/// instead of producing a result.
fn test_iq_matmul(dtype: GgmlDType) -> Result<()> {
    let device = Device::new_cuda(0)?;

    // Dimensions: m=2, k=256 (1 block), n=4.
    // Weight is [n_rows=n, n_cols=k] (transposed in QMatMul: n first, then k).
    let m = 2;
    let k = QK_K; // single block per row
    let n = 4;

    // Build IQ weight tensor on CUDA: shape (n, k) — n rows, k cols.
    // Wrap in Arc so we can both feed QMatMul and dequantize the same weights.
    let qt = Arc::new(make_iq_qtensor(n, k, dtype, &device)?);
    let qmatmul = QMatMul::from_arc(qt.clone())?;

    // Input: (m, k) f32 on CUDA.
    let lhs_data: Vec<f32> = (0..m * k).map(|i| (i as f32) / (m * k) as f32).collect();
    let lhs = Tensor::from_slice(&lhs_data, (m, k), &device)?.to_dtype(DType::F32)?;

    // Run quantized matmul on CUDA. If dispatch is broken, this bails with
    // "CPU matmul is not implemented for IQ3XXS".
    let res = qmatmul.forward(&lhs)?;
    assert!(
        res.device().same_device(&device),
        "result should be on CUDA for {dtype:?}, got {:?}",
        res.device()
    );
    assert_eq!(res.dtype(), DType::F32, "result dtype for {dtype:?}");
    assert_eq!(
        res.shape().dims(),
        [m, n],
        "result shape for {dtype:?}"
    );

    // Reference: dequantize weights to f32 on CPU, matmul.
    let w_cpu = dequantize_to_cpu_f32(&qt)?;
    let lhs_cpu = Tensor::from_slice(&lhs_data, (m, k), &Device::Cpu)?.to_dtype(DType::F32)?;
    // w_cpu shape is (n, k), need (k, n) for lhs @ w_t.
    let ref_mm = lhs_cpu.matmul(&w_cpu.t()?)?;

    // Compare.
    let res_cpu = res.to_device(&Device::Cpu)?;
    let diff = (&res_cpu - &ref_mm)?.abs()?.max_all()?.to_scalar::<f32>()?;
    // With zeroed quant data and d=1.0, dequant values are deterministic but
    // may be nonzero (grid index 0 has nonzero entries). Allow generous
    // tolerance — we mainly care that CUDA dispatch works and produces
    // finite results close to the CUDA-dequantized reference (same kernel
    // path, so should match closely).
    assert!(
        diff.is_finite(),
        "non-finite diff {diff} for {dtype:?}"
    );
    assert!(
        diff < 1e-2,
        "diff {diff} too large for {dtype:?} (CUDA vs CPU-ref)"
    );

    Ok(())
}

#[test]
fn iq3xxs_cuda_matmul() -> Result<()> {
    test_iq_matmul(GgmlDType::IQ3XXS)
}

#[test]
fn iq2s_cuda_matmul() -> Result<()> {
    test_iq_matmul(GgmlDType::IQ2S)
}

#[test]
fn iq3s_cuda_matmul() -> Result<()> {
    test_iq_matmul(GgmlDType::IQ3S)
}

#[test]
fn iq2xs_cuda_matmul() -> Result<()> {
    test_iq_matmul(GgmlDType::IQ2XS)
}

#[test]
fn iq4xs_cuda_matmul() -> Result<()> {
    test_iq_matmul(GgmlDType::IQ4XS)
}

/// Verify that the dequantize kernel produces finite values for each IQ type.
/// This isolates dequantize correctness from the matmul dispatch.
fn test_iq_dequantize_finite(dtype: GgmlDType) -> Result<()> {
    let device = Device::new_cuda(0)?;
    let n_rows = 1;
    let n_cols = QK_K;
    let qt = make_iq_qtensor(n_rows, n_cols, dtype, &device)?;
    let w = dequantize_to_cpu_f32(&qt)?;
    let vals = w.flatten_all()?.to_vec1::<f32>()?;
    assert_eq!(vals.len(), n_rows * n_cols, "len for {dtype:?}");
    for (i, &v) in vals.iter().enumerate() {
        assert!(v.is_finite(), "non-finite dequant value at {i} for {dtype:?}: {v}");
    }
    // With d=1.0 and zeroed qs, every element should be 0.0 (all quant bytes
    // are zero → grid index 0, signs 0, scale factor from zeroed scales).
    // This gives a known-good reference: all zeros.
    for (i, &v) in vals.iter().enumerate() {
        assert_eq!(
            v, 0.0,
            "expected 0.0 at {i} for {dtype:?} with d=1.0 and zeroed qs, got {v}"
        );
    }
    Ok(())
}

#[test]
fn iq3xxs_cuda_dequantize() -> Result<()> {
    test_iq_dequantize_finite(GgmlDType::IQ3XXS)
}

#[test]
fn iq2s_cuda_dequantize() -> Result<()> {
    test_iq_dequantize_finite(GgmlDType::IQ2S)
}

#[test]
fn iq3s_cuda_dequantize() -> Result<()> {
    test_iq_dequantize_finite(GgmlDType::IQ3S)
}

#[test]
fn iq2xs_cuda_dequantize() -> Result<()> {
    test_iq_dequantize_finite(GgmlDType::IQ2XS)
}

#[test]
fn iq4xs_cuda_dequantize() -> Result<()> {
    test_iq_dequantize_finite(GgmlDType::IQ4XS)
}

/// Multi-block test: k = 2*QK_K (two blocks per row), exercises the
/// dequantize kernel grid with >1 block.
fn test_iq_matmul_multiblock(dtype: GgmlDType) -> Result<()> {
    let device = Device::new_cuda(0)?;
    let m = 2;
    let k = 2 * QK_K;
    let n = 4;

    let qt = Arc::new(make_iq_qtensor(n, k, dtype, &device)?);
    let qmatmul = QMatMul::from_arc(qt.clone())?;

    let lhs_data: Vec<f32> = (0..m * k).map(|i| (i as f32) / (m * k) as f32).collect();
    let lhs = Tensor::from_slice(&lhs_data, (m, k), &device)?.to_dtype(DType::F32)?;

    let res = qmatmul.forward(&lhs)?;
    assert_eq!(res.shape().dims(), [m, n], "multiblock shape for {dtype:?}");
    assert!(
        res.device().same_device(&device),
        "multiblock device for {dtype:?}, got {:?}",
        res.device()
    );

    // With zeroed weights (all dequant to 0.0), result should be all zeros.
    let res_cpu = res.to_device(&Device::Cpu)?;
    let vals = res_cpu.flatten_all()?.to_vec1::<f32>()?;
    for (i, &v) in vals.iter().enumerate() {
        assert_eq!(v, 0.0, "multiblock result at {i} for {dtype:?}: {v}");
    }
    Ok(())
}

#[test]
fn iq3xxs_cuda_matmul_multiblock() -> Result<()> {
    test_iq_matmul_multiblock(GgmlDType::IQ3XXS)
}

#[test]
fn iq4xs_cuda_matmul_multiblock() -> Result<()> {
    test_iq_matmul_multiblock(GgmlDType::IQ4XS)
}