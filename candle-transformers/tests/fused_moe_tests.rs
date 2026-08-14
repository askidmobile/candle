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

#[cfg(feature = "cuda")]
#[test]
fn test_fused_moe_dense_cuda_execution() -> Result<()> {
    if !candle::utils::cuda_is_available() {
        println!("CUDA not available, skipping CUDA MoE test");
        return Ok(());
    }
    let device = Device::new_cuda(0)?;
    let dtype = DType::F16;
    let vb = VarBuilder::zeros(dtype, &device);

    let cfg = MoeCfg {
        hidden_size: 64,
        num_experts: 4,
        num_experts_per_tok: 2,
        moe_intermediate_size: 64,
        norm_topk_prob: true,
        act: Activation::Silu,
        decoder_sparse_step: None,
    };

    let moe = FusedMoe::new(&cfg, vb, dtype)?;

    // 1. Test decode (batch=1, seq=1)
    let xs_decode = Tensor::randn(0.0f32, 1.0f32, (1, 1, 64), &device)?.to_dtype(dtype)?;
    let out_decode = moe.forward(&xs_decode, false)?;
    assert_eq!(out_decode.dims(), &[1, 1, 64]);
    println!("CUDA Dense MoE decode forward succeeded: shape {:?}", out_decode.dims());

    // 2. Test prefill (batch=2, seq=4)
    let xs_prefill = Tensor::randn(0.0f32, 1.0f32, (2, 4, 64), &device)?.to_dtype(dtype)?;
    let out_prefill = moe.forward(&xs_prefill, true)?;
    assert_eq!(out_prefill.dims(), &[2, 4, 64]);
    println!("CUDA Dense MoE prefill forward succeeded: shape {:?}", out_prefill.dims());

    Ok(())
}
