// Adapted from https://github.com/guoqingbao/attention.rs/blob/main/src/moe.rs
//
// FFI CUDA-GEMM путь (`moe_gemm_wmma`/`moe_gemm_gguf[_prefill]`) удалён:
// host-символы жили в `libmoe.a`, которая не собирается с T-331 dynamic-loading
// (см. candle-kernels/build.rs). Любая сборка с этим FFI падала на линке с
// unresolved external symbol. Рабочий MoE-путь на CUDA — `QTensor::indexed_moe_forward`
// (PTX-рантайм, dynamic-loading-совместимый, см. candle-core/src/quantized/cuda.rs).
// Quantized GGUF MoE: см. FusedMoeGGUF в candle-transformers/src/fused_moe.rs.
//
// `moe_gemm`/`moe_gemm_gguf` здесь — явные bail-заглушки. Они нужны, потому что
// candle-transformers::fused_moe всё ещё ссылается на них link-time для dense
// FusedMoe и FusedMoeGGUF. Пока dense MoE не переведён на рабочий путь, любой
// вызов честно сообщает об отсутствии реализации вместо падения на линке.
#[allow(unused_imports)]
use candle::quantized::{self, QTensor};
use candle::{Result, Tensor};

/// Dense MoE GEMM (f16/bf16 weights).
///
/// CUDA FFI-путь (`moe_gemm_wmma`) удалён: `libmoe.a` не собирается под
/// dynamic-loading. Dense MoE на CUDA временно не поддерживается. Для
/// quantized GGUF MoE используйте `QTensor::indexed_moe_forward` через
/// `FusedMoeGGUF`.
pub fn moe_gemm(
    _input: &Tensor,
    _weights: &Tensor,
    _topk_weights: &Option<Tensor>,
    _sorted_token_ids: &Tensor,
    _experts_ids: &Tensor,
    _topk: usize,
    _is_prefill: bool,
) -> Result<Tensor> {
    candle::bail!(
        "moe_gemm (dense MoE) is not implemented in this build: the CUDA FFI \
         path requires libmoe.a which is not built under dynamic-loading. \
         Use the quantized GGUF MoE path (FusedMoeGGUF) instead."
    )
}

/// Quantized GGUF MoE GEMM.
///
/// CUDA FFI-путь (`moe_gemm_gguf[_prefill]`) удалён: `libmoe.a` не собирается
/// под dynamic-loading. Рабочий quantized MoE на CUDA —
/// `QTensor::indexed_moe_forward` (PTX-рантайм). `FusedMoeGGUF::forward`
/// должен использовать его напрямую, а не эту функцию.
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