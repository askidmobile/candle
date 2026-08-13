# IQ MoE CUDA on RTX 3060

**Date:** 2026-08-13
**Source:** Phase 3 implementation for Qwen3.6-35B-A3B UD-IQ2_XXS

## What happened

Filename suffix `IQ2_XXS` did not describe all routed expert tensors. Real GGUF routed projections contained 80 `IQ2_XXS`, 37 `IQ2_S`, and 3 `IQ3_S` tensors. Enabling only `IQ2_XXS` failed first on `IQ2_S`, then `IQ3_S`.

Initial GPU router normalized only rank 0. `weight_sum` was a thread-local register updated by thread 0, while threads 1–7 observed zero and skipped division. This produced plausible text but wrong MoE composition.

A fixed four-slot grouping kernel was accidentally reachable from prefill grids larger than four. Fixed arrays then had invalid capacity. Shared-input layout also needs `current_batch`, not zero, when batch exceeds grouping capacity.

## Fix

- Dispatch direct packed F32-activation sparse kernels for exact routed dtype set: `IQ2_XXS`, `IQ2_S`, `IQ3_S`.
- Share selected top-k weight sum through block shared memory.
- Restrict cross-slot route grouping to grid batch 2–4; B=1 and prefill-shaped B>4 use independent task path.
- Test B=1, B=4, B=5, `input_dim1=1`, varied expert IDs, multiple quant blocks, and GPU router parity.

## Results

Windows CUDA 12.4, RTX 3060 12 GB, 128 output tokens, warmed server:

- B=1: PTX 9.13 tok/s; reference 7.52 tok/s.
- B=4: PTX 21.09 aggregate tok/s; reference 19.67 aggregate tok/s.
- CUDA projection tests: 21 passed.
- GPU router parity test: passed.

Full-model exact greedy diverges between reference and PTX. CPU-router/PTX-projection probe also diverges, localizing remaining drift to projection/reduction accumulation order rather than routing. Candle teacher-forced comparison found 5/128 argmax divergences, all at reference margins below 0.183.

Pinned llama.cpp commit `8e7f22b67ef4667b4ddd50230771287f328cfb3f` was built on RTX 3060 with MSVC 19.44, CUDA 12.4, NMake, and SM86. Visual Studio CMake generator was unsuitable: installed VS CUDA integration forced 13.2 despite `CMAKE_CUDA_COMPILER` pointing at 12.4. NMake respected the requested toolkit. Parallel NMake attempts left orphan compiler processes that locked `.obj` files; stop orphan build processes and resume serially after an interrupted SSH command.

An exact-token diagnostic probe fed the same 33 prompt tokens and 128 teacher-forced continuation tokens to all three paths. Results:

- all three agreed on 123/128 argmax decisions;
- PTX matched llama.cpp on disputed steps 16 and 45;
- Candle reference matched llama.cpp on disputed steps 50, 92, and 111;
- free greedy matched llama.cpp for 43 tokens with PTX, versus 16 with Candle reference;
- at five disputed states, PTX nRMSE versus llama.cpp was 0.0272–0.0661 and max absolute error was at most 1.205.

Exact greedy cannot serve as a binary correctness gate for this 2-bit model: llama.cpp itself selects a mix of the two Candle accumulation outcomes near tied logits. Gate fixed teacher-forced states instead: 128 contiguous steps, full logits at 16/45/50/92/111, cosine at least 0.997, nRMSE at most 0.07, max absolute error at most 1.3, and no more than five argmax differences with external margin at most 0.30. Keep free greedy as a drift diagnostic.

PTX then passed clean 4×8K stability on committed CUDA build: all four streams returned HTTP 200 and 8140 completion tokens (`52 + 8140 = 8192`), `finish_reason=length`, one `[DONE]`, no malformed JSON, and bit-exact content in 1232.3 seconds. Post-run reuse returned HTTP 200 in 1.71 seconds. GPU shared usage stayed at 76 MiB, committed GPU memory peaked at 11143 MiB, and logs contained no CUDA/error markers.

PTX is now CUDA default only when all routed gate/up/down dtypes have validated kernels. Explicit `QWEN36_MOE_BACKEND=reference` remains diagnostic rollback. Unsupported dtype combinations and invalid backend values fail during model load instead of silently falling back.

## References

- `candle-kernels/src/quantized.cu`
- `candle-kernels/src/moe_router.cu`
- `candle-core/src/quantized/cuda.rs`
- `candle-core/tests/iq_quant_cuda_tests.rs`
- `qwen35-batch/src/real/moe.rs`
- `qwen35-batch/src/bin/qwen35moe_logits.rs`
- `qwen35-batch/src/bin/qwen35moe_compare.rs`
- `qwen35-batch/tools/llama-logits.cpp`
