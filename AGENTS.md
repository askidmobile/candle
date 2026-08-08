# Project Rules — candle-fork-qwen35-batch

Auto-applied by Warp every conversation. Operational lessons + project conventions. Update when a mistake repeats or a new hard-won fact is learned.

## Repo layout

- Fork of HuggingFace candle (v0.9.2), branch `feat/qwen35-batching`.
- `candle-core/` — quantized CUDA/Metal/CPU dispatch. IQ quant work lives here (`src/quantized/`).
- `candle-kernels/src/quantized.cu` — CUDA dequantize kernels + lookup tables.
- `qwen35-batch/` — continuous batching prototype + real GGUF model (`src/real/model_weights.rs`).
- `qwen36-server/` (on yttri-win only, `D:\Projects\yttri-build\qwen36-server`) — inference server binary.
- Remote (fork): `origin https://github.com/askidmobile/candle.git`. Upstream: `huggingface/candle`.
- Push flow: local commit → `git push origin feat/qwen35-batching` → `git pull` on yttri-win.

## yttri-win (Windows build/test machine) — SSH + shell

- SSH host alias: `yttri-win` (192.168.2.89, User Askid). See `~/.ssh/config`.
- **Default shell is PowerShell.** `&&` is NOT valid in PowerShell — errors with «'&&' is not recognized». Always wrap commands in `cmd /c "..."`.
- **Nested quotes break PowerShell parsing of `(x86)` paths.** `C:\Program Files (x86)\...` inside a PS `-Command` string triggers `ObjectNotFound: (x86:String)`. The `(x86)` parenthesised token gets parsed as a command. Workarounds:
  - For `Test-Path`/file checks: use `powershell -NoProfile -Command "Test-Path 'C:\Program Files (x86)\...\file.bat'"` (single quotes inside double-quoted `-Command`).
  - For commands needing `(x86)` paths + `&&`: **use `cmd /c` instead of PowerShell**, OR run a `.bat` file that contains the `(x86)` `call` line (PowerShell never sees the path).
  - Avoid `&&` entirely inside PS; chain with `;` or separate commands.
- **PATH for CUDA**: `set "PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin;...;%PATH%"` only inside a `.bat`. CUDA is v12.4 on yttri-win (NOT v13.2 — the env var auto-detect picks 13.2 if not overridden).
- **MSVC for nvcc**: nvcc needs `cl.exe` in PATH. Must `call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64` before `cargo build/test --features cuda`. Without it: `nvcc fatal: Cannot find compiler 'cl.exe' in PATH`.
- **Build candle with CUDA on yttri-win**: use a `.bat` that sets VsDevCmd + CUDA env then runs cargo. Pattern: `D:\Projects\yttri-build\run_iq_tests.bat`. Do NOT run bare `cargo test --features cuda` over SSH — it lacks MSVC env.
- **Run IQ CUDA tests**: `ssh yttri-win "cmd /c \"D:\\Projects\\yttri-build\\run_iq_tests.bat\""`. The bat does `cd /d D:\Projects\yttri-build\candle-fork-qwen35-batch && cargo test --features cuda --package candle-core --test iq_quant_cuda_tests`.
- Repo on yttri-win: `D:\Projects\yttri-build\candle-fork-qwen35-batch` (separate from `candle-fork` which is the older copy).

## qwen36-server (Q2_K_XL inference) — launch + diagnostics

- Binary: `D:\Projects\yttri-build\qwen36-server\target\release\qwen36-server.exe`.
- **Server needs env vars** (`QWEN36_MODEL`, `QWEN36_SLOTS`, `QWEN36_CTX`, `QWEN36_PORT`, `QWEN36_API_KEY`, `PATH` with CUDA bin). Launch via a `.bat` that sets them (`run_server_q2.bat`, `start_q2_test.bat`, `start_q2_live.bat`).
- **`Start-Process ... -RedirectStandardOutput` does NOT capture the server's stdout** — the exe likely writes to its own log or buffers. Observed: process starts, GPU stays at 375 MiB (baseline), logs stay 0 bytes, process exits silently. Use the `run_test.bat` pattern instead (background `start /b`, poll for "loaded:" in a logfile, curl test, then kill).
- Model: `D:\Models\unsloth\Qwen3.6-27B-GGUF\Qwen3.6-27B-UD-Q2_K_XL.gguf` (~10 GB Q2_K_XL, 27B). Load takes >90s; 12 GB VRAM (RTX 3060).
- Server logs: `server_q2.log`, `server_q2_iq3.log`, etc. in `D:\Projects\yttri-build\`. All observed 0 bytes → server writes elsewhere or crashes before flush. Check `RUST_BACKTRACE=1` + run foreground with `ssh -t` to see the panic.
- Known failure: inference bails with `"CPU matmul is not implemented for IQ3XXS"` — see IQ quant section below.

## IQ quant CUDA (IQ3XXS, IQ2S, IQ3S, IQ2XS, IQ4XS)

- **Isolated CUDA tests**: `candle-core/tests/iq_quant_cuda_tests.rs`. Run via `run_iq_tests.bat` on yttri-win. All 12 tests PASS (matmul dispatches to `cuda_fwd`, dequantize finite, multiblock correct) as of 2026-08-08.
- **The isolated test path works.** Bug `"CPU matmul is not implemented for IQ3XXS"` is NOT in the candle dispatch — it's in how `qwen35-batch` model passes input tensors to `QMatMul::forward`. The error fires in `cpu_fwd` (`candle-core/src/quantized/mod.rs:530`) which only runs when BOTH: input `xs` is on CPU (`Storage::Cpu`) AND weight is `QStorage::Cpu`. So in the server, either the weight landed on CPU (wrong load path) or the input `xs` is on CPU (engine passes CPU tensor to `QMatMul::forward`).
- **Next investigation target**: `qwen35-batch/src/real/model_weights.rs` — `forward_inner` (line ~5013) calls `self.tok_embeddings.forward(x)` which returns CPU f32 (`QuantizedEmbedding::forward` → `Tensor::from_vec(..., &Device::Cpu)`), then `.to_device(x.device())`. If `x.device()` is CPU (not CUDA), the whole pipeline stays on CPU and IQ weights on CPU hit the bail. Check what device the server passes as the model device and whether `forward_inner`'s `layer_in` ever reaches CUDA before `QMatMul::forward`.

### candle API gotchas (learned the hard way — do NOT repeat)

- **`QTensor` does NOT impl `Clone`.** Use `Arc<QTensor>` + `QMatMul::from_arc(arc.clone())` when you need the same weights for both `QMatMul` and `dequantize`.
- **`Device` does NOT impl `PartialEq`.** `assert_eq!(res.device(), device)` fails to compile. Use `res.device().same_device(&device)` (returns bool) or compare `location()`.
- **`f32` does not impl `Try`.** `((... )?)` with extra parens around a terminal `to_scalar::<f32>()?` makes the outer `?` apply to `f32` → `E0277: the ? operator cannot only be applied to values that implement Try`. Write `let diff = (&a - &b)?.abs()?.max_all()?.to_scalar::<f32>()?;` — no wrapping parens around the final `?`.
- **Zeroed IQ quant blocks dequantize to NONZERO values.** Grid lookup tables have nonzero entries at index 0 (e.g. `iq3xxs_grid[0] = 0x04040404`, `kvalues_iq4nl_f[0] = -127`). With `d=1.0` (f16) and zeroed qs, IQ3XXS/IQ2S/IQ3S/IQ2XS dequantize to ~1.0, IQ4XS to ~4064. Do NOT assert `== 0.0`. Assert finiteness, or compare CUDA matmul against a CUDA-dequantized reference (same kernel path → close match).
- **`from_float` panics for `RawQuantizedType` (IQ types) on CPU.** Can't use `QTensor::quantize` for IQ types. Construct from raw bytes via `QStorage::from_data(Cow::Borrowed(&raw), device, dtype)` — same path as the GGUF loader.

### CUDA kernel struct sizes (quantized.cu)

- `block_iq3_xxs`: `half d` + `uint8_t qs[3*QK_K/8]` = 2 + 96 = 98 bytes.
- `block_iq2_s`: `half d` + `qs[QK_K/4]` + `qh[QK_K/32]` + `scales[QK_K/32]` = 2 + 64 + 8 + 8 = 82 bytes.
- `block_iq3_s`: `half d` + `qs[QK_K/4]` + `qh[QK_K/32]` + `signs[QK_K/8]` + `scales[QK_K/64]` = 2 + 64 + 8 + 32 + 4 = 110 bytes.
- `block_iq2_xs`: `half d` + `uint16_t qs[QK_K/8]` + `scales[QK_K/32]` = 2 + 64 + 8 = 74 bytes.
- `block_iq4_xs`: `half d` + `uint16_t scales_h` + `scales_l[QK_K/64]` + `qs[QK_K/2]` = 2 + 2 + 4 + 128 = 136 bytes.
- `QK_K = 256` for all IQ types. Block size = 256.

## Conventions

- Respond in Russian (per Global Rule).
- Do NOT add `Co-Authored-By` to commits unless explicitly told to (per Global Rule).
- Commit co-author line only on explicit user request.
