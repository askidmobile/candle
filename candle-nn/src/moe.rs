// Adapted from https://github.com/guoqingbao/attention.rs/blob/main/src/moe.rs
//
// FFI CUDA-GEMM путь (`moe_gemm_wmma`/`moe_gemm_gguf[_prefill]`) удалён:
// host-символы жили в `libmoe.a`, которая не собирается с T-331 dynamic-loading
// (см. candle-kernels/build.rs). Любая сборка с этим FFI падала на линке с
// unresolved external symbol. Рабочий MoE-путь на CUDA — `QTensor::indexed_moe_forward`
// (PTX-рантайм, dynamic-loading-совместимый, см. candle-core/src/quantized/cuda.rs).
// Quantized GGUF MoE: см. FusedMoeGGUF в candle-transformers/src/fused_moe.rs.
//
// Dense MoE (`moe_gemm`) удалён: альтернативного PTX-пути для dense MoE нет,
// а naive expert-loop через matmul работает на любом backend и используется
// в qwen3_moe (Qwen3SparseMoeBlock). `moe_gemm_gguf` оставлен как bail-заглушка
// для обратной совместимости — рабочий quantized путь через FusedMoeGGUF
// использует QTensor::indexed_moe_forward напрямую.
#[allow(unused_imports)]
use candle::quantized::{self, QTensor};
use candle::{Result, Tensor};

/// Quantized GGUF MoE GEMM (bail-заглушка).
///
/// CUDA FFI-путь (`moe_gemm_gguf[_prefill]`) удалён: `libmoe.a` не собирается
/// под dynamic-loading. Рабочий quantized MoE на CUDA —
/// `QTensor::indexed_moe_forward` (PTX-рантайм). `FusedMoeGGUF::forward`
/// использует его напрямую, а не эту функцию. Bail оставлен для обратной
/// совместимости с кодом, который мог бы вызывать эту функцию напрямую.
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