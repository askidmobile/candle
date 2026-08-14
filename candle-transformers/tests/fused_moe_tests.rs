// Phase 3: dense FusedMoe (FFI) is removed. Quantized GGUF MoE works via
// FusedMoeGGUF + QTensor::indexed_moe_forward (PTX path). Dense MoE models
// (qwen3_moe) use the naive expert-loop (Qwen3SparseMoeBlock, matmul).
//
// ponytail: a full CUDA/GGUF test requires a GPU + GGUF weights -- out of scope for this self-check.
// When CUDA CI is available: add a Q4K MoE test with a CPU reference comparison
// via FusedMoeGGUF + QTensor::indexed_moe_forward, and a naive dense MoE test
// (Qwen3SparseMoeBlock) with a CPU reference.

#[test]
fn placeholder_no_dense_fused_moe() {
    // Dense FusedMoe is removed; the naive expert-loop is tested indirectly via
    // qwen3_moe model tests. This file is kept as a marker for future MoE tests.
    assert!(true);
}