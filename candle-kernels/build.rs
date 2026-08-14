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
        .exclude(&["moe_*.cu", "mmvq_gguf.cu", "mmq_*.cu"]) // Exclude statically compiled kernels from ptx build
        .arg("--expt-relaxed-constexpr")
        .arg("-std=c++17")
        .arg("-O3")
        .build_ptx()?;

    bindings.write(&ptx_path)?;

    // T-331 / Фаза 0 (dynamic-loading): MoE-ядра (libmoe.a) и сопутствующий
    // `rustc-link-lib=dylib=cudart` УДАЛЕНЫ. cudart-линк здесь делал exe жёстко
    // зависимым от cudart64_*.dll на старте (STATUS_ENTRYPOINT_NOT_FOUND на
    // машинах без CUDA), что подрывает цель dynamic-loading (cudarc грузит CUDA-
    // либы в рантайме через LoadLibrary). Quantized MoE-путь (`indexed_moe_forward`)
    // грузит ядра через `get_or_load_func` (PTX-рантайм), НЕ через FFI в libmoe.a,
    // так что link-time зависимости от libmoe.a нет.
    //
    // upstream в #3855 и связанных коммитах добавил новые MoE-ядра (moe_gguf,
    // moe_wmma, mmq_quantize и т.д.) в libmoe.a. Мы этот builder ОТКЛЮЧЕН.
    // Соответственно FFI-обёртки `moe_gemm_wmma`/`moe_gemm_gguf[_prefill]`
    // (candle-kernels `src/ffi.rs`) недоступны: их host-символы жили ТОЛЬКО в
    // libmoe.a (PTX-рантайм их НЕ даёт). `candle-nn::moe::{moe_gemm,
    // moe_gemm_gguf}` поэтому заменены на bail-заглушки (feature `cuda_moe`
    // удалена). Рабочий quantized MoE на CUDA — `QTensor::indexed_moe_forward`
    // (PTX-рантайм, dynamic-loading-совместимый).
    //
    // TODO(upstream-sync): при реэндейле MoE добавить недостающие ядра из upstream
    // builder выше (moe_gguf, moe_wmma_gguf, mmq_quantize, mmq_instance_q*_k) —
    // список файлов см. в git history upstream/main candle-kernels/build.rs.
    Ok(())
}
