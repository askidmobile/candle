use cudaforge::{KernelBuilder, Result};
use std::env;
use std::path::PathBuf;

fn main() -> Result<()> {
    println!("cargo::rerun-if-changed=build.rs");
    println!("cargo::rerun-if-changed=src/compatibility.cuh");
    println!("cargo::rerun-if-changed=src/cuda_utils.cuh");
    println!("cargo::rerun-if-changed=src/binary_op_macros.cuh");

    // Build for PTX
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let ptx_path = out_dir.join("ptx.rs");
    let mut builder = KernelBuilder::new();
    builder = builder
        .source_dir("src") // Scan src/ for .cu files
        .exclude(&["moe/*", "mmvq_gguf.cu", "mmq_*.cu"]) // Exclude statically compiled kernels from ptx build
        .arg("--expt-relaxed-constexpr")
        .arg("-std=c++17")
        .arg("-O3");

    if let Ok(target) = std::env::var("TARGET") {
        if target.contains("msvc") {
            builder = builder
                .arg("-DCCCL_IGNORE_MSVC_TRADITIONAL_PREPROCESSOR_WARNING")
                .arg("-Xcompiler")
                .arg("/Zc:preprocessor");
        }
    }

    let bindings = builder.build_ptx()?;

    bindings.write(&ptx_path)?;

    // Phase 0 (dynamic-loading): MoE kernels (libmoe.a) and the associated
    // `rustc-link-lib=dylib=cudart` are REMOVED. The cudart link here made the exe
    // hard-dependent on cudart64_*.dll at startup (STATUS_ENTRYPOINT_NOT_FOUND on
    // machines without CUDA), which defeats the goal of dynamic-loading (cudarc
    // loads CUDA libs at runtime via LoadLibrary). The quantized MoE path
    // (`indexed_moe_forward`) loads kernels via `get_or_load_func` (PTX runtime),
    // NOT via FFI into libmoe.a, so there is no link-time dependency on libmoe.a.
    //
    // upstream in #3855 and related commits added new MoE kernels (moe_gguf,
    // moe_wmma, mmq_quantize, etc.) into libmoe.a. This builder is DISABLED.
    // Accordingly the FFI wrappers `moe_gemm_wmma`/`moe_gemm_gguf[_prefill]`
    // (candle-kernels `src/ffi.rs`) are unavailable: their host symbols lived ONLY
    // in libmoe.a (the PTX runtime does not provide them). `candle-nn::moe::{moe_gemm,
    // moe_gemm_gguf}` are therefore replaced with bail stubs (the `cuda_moe` feature
    // is removed). The working quantized MoE on CUDA is `QTensor::indexed_moe_forward`
    // (PTX runtime, dynamic-loading-compatible).
    //
    // TODO(upstream-sync): when re-enabling MoE, add the missing kernels from the
    // upstream builder above (moe_gguf, moe_wmma_gguf, mmq_quantize, mmq_instance_q*_k)
    // -- see the file list in the git history of upstream/main candle-kernels/build.rs.
    Ok(())
}
