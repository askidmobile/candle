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

Full-model exact greedy still diverges between reference and PTX. CPU-router/PTX-projection probe also diverges, localizing remaining drift to projection/reduction accumulation order rather than routing. JSONL teacher-forced comparison found 5/128 argmax divergences, all at reference margins below 0.183; first divergence had cosine 0.999357, nRMSE 0.03695, and max absolute logit error 0.618. Runtime default remains `reference` until llama.cpp comparison identifies which accumulation path is closer to the external reference and tolerance policy is approved.

## References

- `candle-kernels/src/quantized.cu`
- `candle-kernels/src/moe_router.cu`
- `candle-core/src/quantized/cuda.rs`
- `candle-core/tests/iq_quant_cuda_tests.rs`
- `qwen35-batch/src/real/moe.rs`
