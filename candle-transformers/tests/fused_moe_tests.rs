use candle::{DType, Device, Result, Tensor};
use candle_nn::Activation;
use candle_transformers::fused_moe::{FusedMoe, MoeCfg};
use candle_nn::VarBuilder;

// FusedMoe (dense) на любом backend вызывает moe::moe_gemm (bail-заглушка без
// cuda_moe). Проверяем что forward возвращает Err, не panic, не linker failure.
// Раньше с feature cuda_moe это было linker failure; сейчас bail в рантайме.
// ponytail: полный CUDA/GGUF-тест требует GPU + GGUF weights — вне scope self-check.
// Когда появится CUDA CI: добавить тест на Q4K MoE с CPU reference comparison
// через FusedMoeGGUF + QTensor::indexed_moe_forward.
#[test]
fn fused_moe_dense_forward_bails_not_panics() -> Result<()> {
    let device = Device::Cpu;
    let dtype = DType::F32;
    let vb = VarBuilder::zeros(dtype, &device);

    let cfg = MoeCfg {
        hidden_size: 16,
        num_experts: 2,
        num_experts_per_tok: 1,
        moe_intermediate_size: 16,
        norm_topk_prob: true,
        act: Activation::Silu,
        decoder_sparse_step: None,
    };

    let moe = FusedMoe::new(&cfg, vb, dtype)?;
    let xs = Tensor::zeros((1, 1, 16), dtype, &device)?;
    let res = moe.forward(&xs, false);
    assert!(
        res.is_err(),
        "FusedMoe::forward (dense) must Err on CPU — moe_gemm is CUDA-only bail"
    );
    let msg = format!("{}", res.unwrap_err());
    assert!(
        msg.contains("moe_gemm") || msg.contains("not implemented"),
        "unexpected error: {msg}"
    );
    Ok(())
}