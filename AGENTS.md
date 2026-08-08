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
- **Build qwen36-server with CUDA**: `ssh -t yttri-win "cmd /c D:\Projects\yttri-build\build_cuda124.bat"` (28s incremental, foreground). The bat calls VsDevCmd + sets CUDA 12.4 PATH then `cargo build --release --features cuda` inside the qwen36-server dir. Do NOT build without `--features cuda` — the binary falls back to CPU device (see qwen36-server section).

## qwen36-server (Q2_K_XL inference) — launch + diagnostics

- Binary: `D:\Projects\yttri-build\qwen36-server\target\release\qwen36-server.exe`.
- **Server needs env vars** (`QWEN36_MODEL`, `QWEN36_SLOTS`, `QWEN36_CTX`, `QWEN36_PORT`, `QWEN36_API_KEY`, `PATH` with CUDA bin). Launch via a `.bat` that sets them (`run_server_q2.bat`, `start_q2_test.bat`, `start_q2_live.bat`).
- **Foreground launch (диагностика):** `ssh -t yttri-win "cmd /c D:\Projects\yttri-build\start_q2_live.bat"` — показывает stdout/stderr в реальном времени. Сессия жива пока сервер работает. После Ctrl+C — сервер умираст (exe в foreground).
- **Detached launch через SSH не виснет** только если launcher .bat запускает `start /b` и сразу `exit /b 0`. Прямой `ssh yttri-win "cmd /c start /b cmd /c ..."` виснет — SSH ждёт фоновый процесс. Паттерн: `launch_server_detached.bat` → `start /b cmd /c "run_live_inner.bat >live.log 2>&1"` → `exit /b 0`. `run_live_inner.bat` задаёт env vars и запускает exe.
- **curl на yttri-win требует `--noproxy "*"`** — иначе прокси перехватывает localhost запросы, ответ пустой. Рабочий паттерн: `curl -s -m 60 --noproxy "*" http://127.0.0.1:18099/v1/chat/completions -H "Content-Type: application/json" -H "Authorization: Bearer test" -d @file.json`. JSON-body через `@file.json` — inline JSON ломается PowerShell кавычками.
- **`Start-Process ... -RedirectStandardOutput` does NOT capture the server's stdout** — the exe likely writes to its own log or buffers. Observed: process starts, GPU stays at 375 MiB (baseline), logs stay 0 bytes, process exits silently. Use the `run_test.bat` pattern instead (background `start /b`, poll for "loaded:" in a logfile, curl test, then kill).
- Model: `D:\Models\unsloth\Qwen3.6-27B-GGUF\Qwen3.6-27B-UD-Q2_K_XL.gguf` (~10 GB Q2_K_XL, 27B). Load takes >90s; 12 GB VRAM (RTX 3060).
- Server logs: `server_q2.log`, `server_q2_iq3.log`, etc. in `D:\Projects\yttri-build\`. All observed 0 bytes → server writes elsewhere or crashes before flush. Check `RUST_BACKTRACE=1` + run foreground with `ssh -t` to see the panic.
- **Known failure RESOLVED (2026-08-08):** inference bailed with `"CPU matmul is not implemented for IQ3XXS"`. Root cause: `qwen36-server.exe` was built WITHOUT `--features cuda`. `select_device()` (engine.rs:381-394) has `#[cfg(feature = "cuda")]` → `Device::new_cuda(0)`, else falls through to `Ok(Device::Cpu)`. Without the cuda feature, device = CPU → all weights load as `QStorage::Cpu` → `cpu_fwd` → `matmul_t` → bail for RawQuantizedType (IQ types). **Fix: always rebuild via `build_cuda124.bat` (`cargo build --release --features cuda`)** — see build section above.
- **How to verify the binary has cuda:** `dumpbin /dependents qwen36-server.exe`. With cuda, size jumps from ~8.7 MB (no cuda) to ~23.4 MB (cuda kernels embedded). NOTE: `cudart`/`cublas` do NOT appear in dumpbin — candle uses `cudarc` with dynamic-loading (CUDA loaded via `LoadLibrary` at runtime, not statically linked). Size is the reliable indicator.
- **After rebuilding with cuda feature**: server loads Q2_K_XL on CUDA, inference works end-to-end. First curl returned valid chat completion (`"Hello! How can I help you today"`, 8 completion / 13 prompt tokens).

## Prefill performance (2026-08-08)

- **Prefill НЕ виснет — медленный.** 3000-токенный промпт: 15 чанков × ~39s = ~583s (600s timeout). Время растёт линейно (11s → 43s → 83s → ...).
- **Chunked prefill** (engine.rs:240-276, `PREFILL_CHUNK=256`): prompt разбит на чанки по 256 токенов, `model.forward` вызывается для каждого. KV cache накапливается.
- **DeltaNet `forward_prefill` CUDA path** (model_weights.rs:2081-2103): token-by-token loop по seq_len, 4 kernel launches + sync alloc на токен × 32 слоя = 4096 GPU syncs/chunk. CPU fallback (model_weights.rs:2105-2322) медленнее (33s vs 18s/chunk). Оставляем CUDA.
- **Attention `forward_attn` prefill** (model_weights.rs:2710-2851): chunked F32 manual matmul, scores [256 × kv_len] растут O(n²) с KV cache. На 3000 токенов kv_len=3000 → scores 256×3000×4=3MB, но matmul cuBLAS дороже с ростом.
- **Tiled dequantize matmul** (cuda.rs:175-213, 923-964): commit `2fdd7a1e`. Снижает dequant buffer peak с 357MB до 35.7MB. IQ tests 12/12 pass.
- **Короткие промпты (<500 токенов):** ~2 чанка × ~11s = ~22s — приемлемо для live chat.
- **Длинные промпты (>2000 токенов):** >300s — оптимизация fused batch DeltaNet kernel (как Metal GDN на macOS) отдельная задача.
- **eprintln timing markers** в engine.rs prefill loop: `[prefill] chunk start={} end={} elapsed={:.1}ms`. Использовать для диагностики. qwen36-server не имеет логгера — `log::debug!`/`log::info!` не выводятся. Только `eprintln!` попадает в stdout/redirect-лог.

## IQ quant CUDA (IQ3XXS, IQ2S, IQ3S, IQ2XS, IQ4XS)

- **Isolated CUDA tests**: `candle-core/tests/iq_quant_cuda_tests.rs`. Run via `run_iq_tests.bat` on yttri-win. All 12 tests PASS (matmul dispatches to `cuda_fwd`, dequantize finite, multiblock correct) as of 2026-08-08.
- **The candle dispatch is CORRECT.** `QMatMul::forward` (mod.rs:1037-1060) → `xs.apply_op1_no_bwd(t)` → `Storage::apply_op1` (storage.rs:205-220) dispatches by INPUT storage → `cuda_fwd` for Cuda, `cpu_fwd` for Cpu. `QStorage::from_data` (mod.rs:87-153) routes IQ types to `cuda::load_quantized_bytes` for CUDA device. Loading path verified correct. The only failure mode is building without the cuda feature (see qwen36-server section above).
- **Dispatch path** (confirmed correct): `QMatMul::forward` → `cuda_fwd` (mod.rs:952-1035 CustomOp1) → for IQ types, fallback to `dequantize + cuBLAS` matmul in `cuda.rs:765`.
- **Model forward path** (`model_weights.rs:5009-5024`): `forward_inner` → `emb_cpu = tok_embeddings.forward(x)` (returns CPU f32) → `layer_in = emb_cpu.to_device(x.device())`. If `x.device()` = CUDA → `layer_in` on CUDA → all `QMatMul` weights already on CUDA (loaded via `load_heavy` with CUDA device).

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
