use candle::{DType, Device, Result, Tensor};
use candle_nn::Activation;
use candle_transformers::fused_moe::{FusedMoe, MoeCfg};
use candle_nn::VarBuilder;

#[test]
fn fused_moe_dense_cpu_bails_not_panics() -> Result<()> {
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
        "FusedMoe::forward (dense) must Err on CPU -- moe_gemm is CUDA-only"
    );
    let msg = format!("{}", res.unwrap_err());
    assert!(
        msg.contains("moe_gemm") || msg.contains("CUDA"),
        "unexpected error: {msg}"
    );
    Ok(())
}
