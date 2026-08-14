// Adapted from https://github.com/guoqingbao/attention.rs/blob/main/src/moe.rs
//
// FFI CUDA-GEMM path (`moe_gemm_wmma`/`moe_gemm_gguf[_prefill]`) is removed:
// host symbols lived in `libmoe.a`, which is not built under dynamic-loading
// (see candle-kernels/build.rs). Any build linking this FFI failed at link time
// with unresolved external symbol. The working MoE path on CUDA is
// `QTensor::indexed_moe_forward` (PTX runtime, dynamic-loading-compatible;
// see candle-core/src/quantized/cuda.rs).
// Quantized GGUF MoE: see FusedMoeGGUF in candle-transformers/src/fused_moe.rs.
//
// Dense MoE (`moe_gemm`) is removed: there is no alternative PTX path for dense
// MoE, and the naive expert-loop via matmul works on any backend and is used in
// qwen3_moe (Qwen3SparseMoeBlock). `moe_gemm_gguf` is kept as a bail stub for
// backward compatibility -- the working quantized path via FusedMoeGGUF uses
// QTensor::indexed_moe_forward directly.
#[allow(unused_imports)]
use candle::quantized::{self, QTensor};
use candle::{Result, Tensor};

/// Quantized GGUF MoE GEMM (bail stub).
///
/// The CUDA FFI path (`moe_gemm_gguf[_prefill]`) is removed: `libmoe.a` is not
/// built under dynamic-loading. The working quantized MoE on CUDA is
/// `QTensor::indexed_moe_forward` (PTX runtime). `FusedMoeGGUF::forward` uses it
/// directly rather than this function. The bail is kept for backward
/// compatibility with code that may call this function directly.
#[allow(clippy::too_many_arguments)]
pub fn moe_gemm_gguf(
    _input: &Tensor,
    _weights: &QTensor,
    _topk_weights: &Option<Tensor>,
    _sorted_token_ids: &Tensor,
    _experts_ids: &Tensor,
    _topk: usize,
    _is_prefill: bool,
    _dtype: candle::DType,
) -> Result<Tensor> {
    candle::bail!(
        "moe_gemm_gguf is not implemented in this build: the CUDA FFI path \
         requires libmoe.a which is not built under dynamic-loading. \
         Use QTensor::indexed_moe_forward (PTX runtime) via FusedMoeGGUF instead."
    )
}