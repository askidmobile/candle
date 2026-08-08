use cudaforge::{KernelBuilder, Result};
use std::env;
use std::path::PathBuf;

fn main() -> Result<()> {
    println!("cargo::rerun-if-changed=build.rs");
    println!("cargo::rerun-if-changed=src/compatibility.cuh");
    println!("cargo::rerun-if-changed=src/cuda_utils.cuh");
    println!("cargo::rerun-if-changed=src/binary_op_macros.cuh");
    println!("cargo::rerun-if-changed=src/moe_dequant.cuh");

    // Build for PTX.
    // Exclude the old reference MoE kernels under src/moe/ (they use the CUDA Runtime
    // API and are not PTX-compatible). The new device-only kernels moe_router.cu and
    // moe_quantized.cu live directly under src/ and must be compiled.
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let ptx_path = out_dir.join("ptx.rs");
    let bindings = KernelBuilder::new()
        .source_dir("src")
        .exclude(&["moe/*"])
        .arg("--expt-relaxed-constexpr")
        .arg("-Xcompiler")
        .arg("/Zc:preprocessor")
        .arg("-std=c++17")
        .arg("-O3")
        .build_ptx()?;

    bindings.write(&ptx_path)?;
    Ok(())
}
