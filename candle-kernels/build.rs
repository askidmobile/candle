use cudaforge::{KernelBuilder, Result};
use std::env;
use std::path::PathBuf;

fn main() -> Result<()> {
    println!("cargo::rerun-if-changed=build.rs");
    println!("cargo::rerun-if-changed=src/compatibility.cuh");
    println!("cargo::rerun-if-changed=src/cuda_utils.cuh");
    println!("cargo::rerun-if-changed=src/binary_op_macros.cuh");
    println!("cargo::rerun-if-changed=src/moe_dequant.cuh");
    // Явно следим за ВСЕМИ .cu (новый файл без этого не пересобирает ptx.rs —
    // поймано: FLASH_DECODE отсутствовал в сгенерированном модуле).
    for entry in std::fs::read_dir("src").unwrap() {
        let p = entry.unwrap().path();
        if p.extension().and_then(|s| s.to_str()) == Some("cu") {
            println!("cargo::rerun-if-changed={}", p.display());
        }
    }

    // Build for PTX.
    // Exclude the old reference MoE kernels under src/moe/ (they use the CUDA Runtime
    // API and are not PTX-compatible). The new device-only kernels moe_router.cu and
    // moe_quantized.cu live directly under src/ and must be compiled.
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let ptx_path = out_dir.join("ptx.rs");
    let mut builder = KernelBuilder::new()
        .source_dir("src")
        .exclude(&["moe/*"])
        .arg("--expt-relaxed-constexpr");
    // /Zc:preprocessor — только MSVC; на Linux gcc примет флаг за имя файла
    // и упадёт с "-o with multiple files" (урок сборки на A100 2026-08-11).
    #[cfg(target_os = "windows")]
    {
        builder = builder.arg("-Xcompiler").arg("/Zc:preprocessor");
    }
    let bindings = builder
        .arg("-std=c++17")
        .arg("-O3")
        .build_ptx()?;

    bindings.write(&ptx_path)?;
    Ok(())
}
