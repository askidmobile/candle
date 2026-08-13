# Qwen3.5 dense CUDA: Yttri comparison and VRAM diagnosis

Date: 2026-08-13

GPU: RTX 3060 12 GB, Windows WDDM, CUDA 12.4, SM86

Models: Qwen3.5-4B/9B Q4_K_M

Reference: llama.cpp `8e7f22b67ef4667b4ddd50230771287f328cfb3f`

## Result

Observed 12 GiB VRAM saturation was benchmark-harness error, not dense runtime allocation. Two ignored tests ran concurrently and each loaded adapters; one test also loaded separate B=1/B=2/B=4 adapters. WDDM paged duplicate 9B weights into shared system memory.

Benchmark now uses one `Qwen35BatchAdapter` with capacity 4. B changes only through active prompt count. Test command must include `--ignored --exact --nocapture --test-threads=1`.

Permanent guard rejects:

- model VRAM delta greater than GGUF size + 1024 MiB;
- B=1 to B=4 growth greater than 256 MiB.

## Isolated memory

Clean GPU baseline: 453 MiB used.

| Model | B=1 total | B=4 total | Runtime delta | Post-exit |
|---|---:|---:|---:|---:|
| 4B | 3448 MiB | 3480 MiB | 2880–2912 MiB | 453 MiB |
| 9B | 5688 MiB | 5720 MiB | 5120–5152 MiB | 453 MiB |

Fresh-server WDDM telemetry stayed resident:

| Model | Loaded dedicated | After B=4 dedicated | Shared |
|---|---:|---:|---:|
| 4B | 3002.9 MiB | 3034.9 MiB | 76.0 MiB |
| 9B | 5210.9 MiB | 5274.9 MiB | 76.0 MiB |

No runtime paging found.

## Yttri prefill comparison

Yttri solved two relevant problems:

1. CUDA prefill attention drift: commit `952bb5d18` moved manual CUDA attention from F16 to F32. Current fork already contains this fallback and defaults to FlashAttention 2 with F32 accumulation (`ec13b1e4`).
2. Repeated prompts: commits `70ab76a30` and `bd7e4b230` place recurrent snapshots at observed common-prefix boundaries. This accelerates repeated shared prefixes, not first unique prompt prefill. Server prefix cache remains disabled until its final-logits/restore contract is corrected.

Yttri CUDA Gated DeltaNet still used a Rust-side per-token loop. Current fork is newer: fused sequence kernels move token loop into CUDA and keep recurrent state in registers (`0b3b5e2f`, `048463cc`). Porting Yttri compute path would regress performance.

Comparable cold 2191-token 4B prefill:

| Path | Time | Throughput |
|---|---:|---:|
| Yttri CUDA | about 5.0 s | 434–439 tok/s |
| Current fork, one forward | 1.593 s | 1375.7 tok/s |
| Current fork, server chunks 512 | 1.789 s | 1224.4 tok/s |
| Current fork, chunks 2048 | 1.647 s | 1330.2 tok/s |

Full/512/2048 produced identical final logits:

```text
argmax   248046
margin   3.3532981872558594
checksum dba55b1794191e1d
```

Chunk 2048 improves 4B throughput only 8.6% over server default 512. Default stays 512 because it bounds 35B transient VRAM and preserves decode interleaving. No model-specific scheduler policy added for this small gain.

## Synchronized Candle vs llama.cpp timing

Both probes use identical 33 prompt tokens and 127 teacher-forced decode calls. llama.cpp timing calls `llama_synchronize(ctx)` after each successful `llama_decode`; enqueue-only timing is invalid.

| Model | Metric | Candle | llama.cpp | Candle/reference |
|---|---|---:|---:|---:|
| 4B | ready-to-prefill load | 1.094 s | 1.674 s | 0.654× |
| 4B | 33-token prefill | 251.0 tok/s | 618.9 tok/s | 0.406× |
| 4B | B=1 decode | 56.57 tok/s | 88.47 tok/s | 0.639× |
| 9B | ready-to-prefill load | 1.856 s | 2.589 s | 0.717× |
| 9B | 33-token prefill | 188.9 tok/s | 521.9 tok/s | 0.362× |
| 9B | B=1 decode | 41.63 tok/s | 55.66 tok/s | 0.748× |

The 33-token prefill row measures launch latency and must not be presented as long-prefill throughput.

## Correctness status

- 4B: 128/128 teacher-forced argmax equal.
- 9B: 126/128 argmax equal; two divergences have llama.cpp margins 0.0585 and 0.0127.
- 4B step 111 remains unresolved: direct quantized path nRMSE 0.4567, max abs 6.3868. Full dequantization plus cuBLAS reduces it to nRMSE 0.1123, max abs 1.9841, locating most drift in quantized matvec. It is diagnostic-only at 7.39 tok/s and is not a production fallback.

Dense models remain unpromoted until the token-specific numerical spike receives a calibrated gate or runtime fix.
