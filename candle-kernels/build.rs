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
    let bindings = KernelBuilder::new()
        .source_dir("src") // Scan src/ for .cu files
        .exclude(&["moe_*.cu"]) // Exclude moe kernels for ptx build
        .arg("--expt-relaxed-constexpr")
        .arg("-std=c++17")
        .arg("-O3")
        .build_ptx()?;

    bindings.write(&ptx_path)?;

    // T-331 / Фаза 0 (dynamic-loading): MoE-ядра (libmoe.a) и сопутствующий
    // `rustc-link-lib=dylib=cudart` УДАЛЕНЫ. cudart-линк здесь делал exe жёстко
    // зависимым от cudart64_*.dll на старте (STATUS_ENTRYPOINT_NOT_FOUND на
    // машинах без CUDA), что подрывает цель dynamic-loading (cudarc грузит CUDA-
    // либы в рантайме через LoadLibrary). MoE-путь (`indexed_moe_forward`) грузит
    // ядра через `get_or_load_func` (PTX-рантайм), НЕ через FFI в libmoe.a, так что
    // link-time зависимости от libmoe.a нет. Целевые модели Yttri — dense (Qwen3.5-4B),
    // не MoE. Если MoE на CUDA понадобится — вернуть под dynamic-loading-совместимой схемой.
    Ok(())
}
