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
    // не MoE.
    //
    // ⚠️ ЛАТЕНТНАЯ МИНА (review T-331): СУЩЕСТВУЕТ ВТОРОЙ MoE-путь — `extern "C"`
    // FFI `moe_gemm_wmma`/`moe_gemm_gguf`/`moe_gemm_gguf_prefill` (candle-kernels
    // `src/ffi.rs`), вызываемый из `candle-nn::moe` (#[cfg(feature="cuda")]) и далее
    // `candle-transformers::fused_moe` → `quantized_qwen3_moe`. Host-символы этих
    // функций жили ТОЛЬКО в удалённой libmoe.a (PTX-рантайм их НЕ даёт). Сейчас
    // сборка зелёная только из-за dead-code elimination: Yttri-бинарь не ссылается
    // ни на один из этих символов. ЛЮБОЙ бинарь с feature="cuda", инстанцирующий
    // `FusedMoeGGUF::forward` / `moe_gemm_*`, упадёт на финальном линке с unresolved
    // external symbol. Прежде чем включать MoE на CUDA — вернуть libmoe.a под
    // dynamic-loading-совместимой схемой (cudart через cudarc, без hard-link) ЛИБО
    // cfg-выключить cuda-ветку `candle-nn::moe` + `ffi.rs`.
    Ok(())
}
